"""TreeWM training entry point.

    torchrun --nproc_per_node=2 scripts/train.py experiment=pointmaze_treewm seed=0

Runs single-process too (no torchrun) with identical semantics. Only rank 0 creates the
TensorBoard writer, saves checkpoints, renders plots and prints progress; every logged
scalar is reduced across ranks first (spec section 23).
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import json
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
from treewm.data.retrieval_index import LatentIndex, compute_endpoint_cells, sample_key_indices
from treewm.data.samplers import InfiniteLoader, build_dataloader, to_device
from treewm.evaluation import diagnostics as diag
from treewm.evaluation import tree_stats as tstats
from treewm.evaluation import tree_viz as tv
from treewm.tree.frontier import GOAL_AWARE_SCORERS
from treewm.evaluation.rollout import EvaluationInterrupted, evaluate
from treewm.evaluation.tasks import build_tasks, describe_tasks
from treewm.evaluation.coverage import StateQuantizer
from treewm.data.maze_utils import MazeSpec
from treewm.logging.metrics import MetricTracker
from treewm.logging.tensorboard import TreeWMLogger
from treewm.losses.expansion_losses import novelty_gain_loss
from treewm.losses.recursive_losses import multi_step_recursive_loss, scheduled_sampling_schedule
from treewm.losses.total import (
    assemble_loss_terms,
    compute_branch_losses,
    compute_expansion_gain_loss,
    loss_term_metrics,
)
from treewm.models.baselines import build_model, tree_config_for
from treewm.planning.goal_planner import GoalPlanner
from treewm.utils import config as cfg_utils
from treewm.utils.checkpoint import (
    GRACEFUL_EXIT_CODE,
    CheckpointManager,
    StopController,
    atomic_json_dump,
    load_checkpoint,
)
from treewm.utils.distributed import (
    all_reduce_mean,
    any_rank_true,
    barrier,
    cleanup_distributed,
    gather_rank_objects,
    is_distributed,
    setup_distributed,
)
from treewm.utils.meta import build_run_dir, count_parameters, env_summary, git_commit, hostname
from treewm.utils.provenance import runtime_fingerprint, trainer_code_fingerprint
from treewm.utils.rng import RngStreams, make_generator
from treewm.utils.seeding import get_rng_state, seed_everything, set_rng_state


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
    configured_horizon = cfg.train.get("scheduler_total_steps")
    total = max(
        warmup + 1,
        int(configured_horizon) if configured_horizon is not None else int(cfg.train.steps),
    )
    floor = float(cfg.train.min_lr_scale)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@contextmanager
def preserve_global_rng_state(*, strict_cuda: bool = False):
    """Make stochastic diagnostics observational rather than training interventions.

    Validation calls the same branch objective as training and therefore samples
    control anchors and recursive nodes.  Its cadence must not change the subsequent
    optimizer trajectory (in particular, a short pilot may checkpoint more often than
    the formal run).  Snapshot every global Python/NumPy/Torch stream and restore it
    before a checkpoint can capture rank-local state.
    """
    state = get_rng_state()
    try:
        yield
    finally:
        set_rng_state(state, strict_cuda=strict_cuda)


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def required_formal_provenance_hashes(
    objective_version: str,
    *,
    protocol_sha256: str | None,
    code_sha256: str | None,
    runtime_sha256: str | None,
    calibration_sha256: str | None,
    future_recipe_sha256: str | None,
) -> dict[str, str | None]:
    required = {
        "TREEWM_PROTOCOL_SHA256": protocol_sha256,
        "TREEWM_CODE_SHA256": code_sha256,
        "TREEWM_RUNTIME_SHA256": runtime_sha256,
    }
    if objective_version == "treewm_v2_rms_rank_v1":
        required["TREEWM_CALIBRATION_SHA256"] = calibration_sha256
        required["TREEWM_FUTURE_RECIPE_SHA256"] = future_recipe_sha256
    return required


def malformed_sha256_names(values: Mapping[str, str | None]) -> list[str]:
    return [
        name
        for name, value in values.items()
        if not value
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ]


def _finite_metrics(values: dict[str, float]) -> dict[str, float]:
    """JSON-stable metric payload; NaN would make two identical artifacts compare unequal."""
    return {key: float(value) for key, value in values.items() if np.isfinite(float(value))}


def gradients_finite(parameters) -> bool:
    """Return false on the first non-finite gradient (parameters without grads are inert)."""
    checks = [
        torch.isfinite(parameter.grad).all()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return not checks or bool(torch.stack(checks).all().item())


def objective_finite(loss: torch.Tensor) -> bool:
    """Scalar/tensor objective guard used collectively before any backward call."""
    return bool(torch.isfinite(loss).all().item())


def gradient_l2_norm(parameters) -> float:
    """Unclipped global L2 norm for a disjoint parameter collection."""
    squares = [
        parameter.grad.detach().float().pow(2).sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().item())


def gradient_parameter_groups(model, include_branch_prior: bool) -> tuple[list, list]:
    """Disjoint world/contextual-gain paths used for optimiser clipping and telemetry."""
    gain_parameters = list(model.gain_head.parameters())
    if include_branch_prior:
        gain_parameters.extend(model.heads.gain_head.parameters())
    gain_ids = {id(parameter) for parameter in gain_parameters}
    world_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in gain_ids
    ]
    gain_parameters = [parameter for parameter in gain_parameters if parameter.requires_grad]
    return world_parameters, gain_parameters


def module_gradient_norms(model, include_branch_prior: bool) -> dict[str, float]:
    modules = {
        "encoder": model.encoder,
        "branch_transformer": model.branch_transformer,
        "dynamics": model.dynamics,
        "controllability": model.controllability,
        "contextual_gain": model.gain_head,
    }
    if include_branch_prior:
        modules["branch_gain_prior"] = model.heads.gain_head
    return {
        f"train/grad_norm_module/{name}": gradient_l2_norm(module.parameters())
        for name, module in modules.items()
    }


def formal_v2_objective_contract(
    model,
    loss_cfg,
    match_cfg,
    future_cfg,
    tree_cfg,
    *,
    separate_gain_clip: bool,
) -> dict[str, bool]:
    """Pure, testable formal-v2 contract evaluated before optimiser construction."""
    return {
        "one_q_scale": int(model.controllability.num_scales) == 1,
        # The bounded q-distance and normalized matching formulas are valid only for
        # an L2-normalized, single-scale q representation.  The remaining switches
        # ensure the heads trained by v2 are exactly the ones consumed by inference.
        "normalized_q": bool(model.cfg.normalize_q)
        and bool(model.controllability.normalize),
        "q_novelty_space": str(model.cfg.novelty_space) == "q",
        "tree_context_enabled": bool(model.cfg.use_tree_context)
        and bool(model.gain_head.use_context),
        "learned_horizon_selection": str(model.cfg.horizon_mode) == "learned",
        "formal_horizon_values": tuple(int(value) for value in model.cfg.horizons)
        == (4, 8, 16, 32, 64),
        "formal_horizon_width": int(model.cfg.h_max) == 64,
        "model_tree_depth_aligned": int(model.cfg.max_depth)
        == int(tree_cfg.max_depth)
        == 16,
        "depth_embedding_disabled": not bool(model.cfg.use_depth_embedding)
        and not bool(model.branch_transformer.use_depth_embedding),
        "set_aware_gain": bool(loss_cfg.gain_set_context),
        "legacy_gain_disabled": not any(
            parameter.requires_grad for parameter in model.gain_head.net.parameters()
        ),
        "tree_signature_disabled": not any(
            parameter.requires_grad for parameter in model.tree_signature.parameters()
        ),
        "metric_endpoint": str(loss_cfg.control_endpoint_key) == "fut_metric_endpoint",
        "endpoint_fallback_disabled": not bool(loss_cfg.control_allow_endpoint_fallback),
        "single_scale_guard": bool(loss_cfg.control_require_single_scale),
        "bounded_rms_control": str(loss_cfg.control_target_transform) == "rms_tanh",
        "unit_future_scale": float(loss_cfg.future_scale) == 1.0,
        "future_set_control": str(loss_cfg.control_objective) == "future_set",
        "control_metric_active": float(loss_cfg.control_metric_weight) > 0,
        "control_rank_active": float(loss_cfg.control_rank_weight) > 0,
        "novelty_gain_target": str(loss_cfg.gain_target) == "novelty",
        "gain_rank_active": float(loss_cfg.gain_rank_weight) > 0,
        "gain_calibration_active": float(loss_cfg.gain_calibration_weight) > 0,
        "mass_objective_disabled": not loss_cfg.on("mass"),
        "mass_head_disabled": not any(
            parameter.requires_grad for parameter in model.heads.mass_head.parameters()
        ),
        "branch_prior_weight_zero": float(loss_cfg.gain_branch_prior_weight) == 0.0,
        "branch_prior_disabled": not any(
            parameter.requires_grad for parameter in model.heads.gain_head.parameters()
        ),
        "detached_world_targets": bool(loss_cfg.detach_world_targets),
        "matching_rms_v2": str(match_cfg.normalization_version) == "rms_v2",
        "horizon_cardinality": int(match_cfg.num_horizons)
        == int(future_cfg.num_horizons)
        == len(model.horizons),
        "future_metric_rms_v2": str(future_cfg.metric_mode) == "rms_v2",
        "keep_threshold": tree_cfg.keep_threshold is not None,
        "separate_gain_grad_clip": bool(separate_gain_clip),
    }


class TrainingStepModule(torch.nn.Module):
    """Put the complete differentiable training graph behind one DDP forward.

    Calling custom methods on the module wrapped by DistributedDataParallel bypasses
    DDP's forward bookkeeping. The old trainer did exactly that, so its gradients were
    rank-local. Auxiliary losses must also be returned by this forward; otherwise
    ``find_unused_parameters`` can mark their parameters ready before their gradients
    arrive.
    """

    def __init__(
        self,
        model,
        loss_cfg,
        match_cfg,
        gain_tree_cfg,
        latent_index,
        quantizer,
        train_cfg,
        model_cfg,
        losses_cfg,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_cfg = loss_cfg
        self.match_cfg = match_cfg
        self.gain_tree_cfg = gain_tree_cfg
        self.latent_index = latent_index
        self.quantizer = quantizer
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.losses_cfg = losses_cfg

    def forward(
        self,
        batch,
        step: int,
        planner_generator: torch.Generator,
        return_loss_terms: bool = False,
    ):
        branch_loss, metrics, artifacts, branch_terms = compute_branch_losses(
            self.model,
            batch,
            self.loss_cfg,
            self.match_cfg,
            step=step,
            return_loss_terms=True,
        )
        raw_terms = dict(branch_terms.raw)

        if self.loss_cfg.on("multistep"):
            p_ss = scheduled_sampling_schedule(
                step,
                float(self.loss_cfg.scheduled_sampling_p),
                int(self.loss_cfg.scheduled_sampling_warmup),
            )
            ms_loss, ms_metrics = multi_step_recursive_loss(
                self.model,
                batch,
                scheduled_sampling_p=p_ss,
                depth_weights=self.loss_cfg.multistep_depth_weights or None,
            )
            raw_terms["multistep"] = ms_loss
            metrics.update(ms_metrics)

        gain_active = self.loss_cfg.on("expand") and (
            step % int(self.train_cfg.gain_loss_every) == 0
        )
        if self.loss_cfg.on("expand"):
            gain_loss = branch_loss * 0.0
        if gain_active:
            n_gain = min(int(self.train_cfg.gain_batch_size), batch["obs"].shape[0])
            if str(self.losses_cfg.gain_target) == "novelty":
                gain_loss, gain_metrics = novelty_gain_loss(
                    self.model,
                    artifacts["z"][:n_gain],
                    self.gain_tree_cfg,
                    space=str(self.model_cfg.novelty_space),
                    generator=planner_generator,
                    rank_weight=float(self.loss_cfg.gain_rank_weight),
                    calibration_weight=float(self.loss_cfg.gain_calibration_weight),
                    branch_prior_weight=float(self.loss_cfg.gain_branch_prior_weight),
                )
            else:
                gain_loss, gain_metrics = compute_expansion_gain_loss(
                    self.model,
                    artifacts["z"][:n_gain],
                    self.gain_tree_cfg,
                    self.latent_index,
                    self.quantizer,
                )
            metrics.update(gain_metrics)
        if self.loss_cfg.on("expand"):
            raw_terms["expand"] = gain_loss

        terms = assemble_loss_terms(raw_terms, self.loss_cfg, step)
        loss = terms.total
        metrics.update(loss_term_metrics(terms))
        metrics["train/loss_total_branch"] = float(branch_loss.detach().item())
        metrics["train/loss_total_backward"] = float(loss.detach().item())
        metrics["train/gain_active"] = float(gain_active)
        metrics["train/gain_active_fraction"] = float(gain_active)

        if return_loss_terms:
            return loss, metrics, artifacts, terms
        return loss, metrics, artifacts



def resource_metrics() -> dict[str, float]:
    """Peak VRAM / host RSS, emitted in a form scripts/fleet.py can parse from the log.

    The fleet bin-packs jobs from these numbers, so they must come from the process
    itself rather than from nvidia-smi, which cannot attribute memory per process when
    eight jobs share a GPU.
    """
    import resource

    out = {"host_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0}
    if torch.cuda.is_available():
        out["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1073741824.0
        out["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1073741824.0
    print("  ".join(f"{k}={v:.3f}" for k, v in out.items()), flush=True)
    return {f"resource/{k}": v for k, v in out.items()}


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    dist_info = setup_distributed()
    stop = StopController()
    stop.install()
    seed_everything(int(cfg.seed), rank=dist_info.rank)
    device = torch.device(
        f"cuda:{dist_info.local_rank}" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    )

    total_steps = int(cfg.train.steps)
    if total_steps <= 0:
        raise ValueError("train.steps must be positive")
    configured_scheduler_horizon = cfg.train.get("scheduler_total_steps")
    scheduler_total_steps = int(
        configured_scheduler_horizon
        if configured_scheduler_horizon is not None
        else total_steps
    )
    if scheduler_total_steps < total_steps:
        raise ValueError("train.scheduler_total_steps cannot be smaller than train.steps")
    objective_version = str(cfg.get("objective_version", "treewm_v1"))
    if objective_version not in {"treewm_v1", "treewm_v2_rms_rank_v1"}:
        raise ValueError(f"unsupported objective_version: {objective_version!r}")
    gradient_checkpointing = bool(cfg.train.gradient_checkpointing)
    if total_steps == 1_000_000 and not gradient_checkpointing:
        raise ValueError("formal 1M TreeWM training requires train.gradient_checkpointing=true")
    explicit_run_name = cfg.run_name or os.environ.get("TREEWM_RUN_NAME")
    if total_steps == 1_000_000 and not explicit_run_name:
        raise ValueError("formal 1M runs require a stable run_name or TREEWM_RUN_NAME")
    run_dir = build_run_dir(cfg.run_root, cfg.env.short_name, cfg.arm, int(cfg.seed))
    if explicit_run_name:
        run_dir = Path(cfg.run_root) / cfg.env.short_name / cfg.arm / str(explicit_run_name)
    run_dir = run_dir.expanduser().resolve()
    ckpt = CheckpointManager(run_dir / "checkpoints", enabled=dist_info.is_main)
    completion_path = run_dir / "COMPLETED.json"

    if dist_info.is_main:
        print(f"[treewm] arm={cfg.arm} env={cfg.env.name} seed={cfg.seed}")
        print(f"[treewm] run_dir={run_dir}")

    # ---------------------------------------------------------------- data
    future_cfg = cfg_utils.future_set_config(cfg)
    if cfg.env.get("relative_endpoints") is not None:
        future_cfg = replace(
            future_cfg, relative_endpoints=bool(cfg.env.get("relative_endpoints"))
        )
    env, train_ds, val_ds, normalizer = build_datasets(
        cfg.env.name,
        future_cfg,
        dataset_dir=cfg.env.dataset_dir,
        xy_dims=tuple(cfg.env.xy_dims),
        max_train_anchors=int(cfg.train.max_train_anchors),
        max_val_anchors=int(cfg.train.max_val_anchors),
        seed=int(cfg.seed),
        cache_future_sets=bool(cfg.future_sets.get("cache", False)),
        shared_cache=bool(cfg.future_sets.get("shared_cache", False)),
        dataset_kind=str(cfg.env.get("dataset_kind", "standard")),
        source_name=str(cfg.env.get("source_name", cfg.env.name)),
        expected_shards=int(cfg.env.get("expected_shards", 1)),
        cache_root=os.environ.get("TREEWM_CACHE"),
        data_manifest_sha256=os.environ.get("TREEWM_DATA_SHA256"),
        task_metric_dims=tuple(cfg.env.get("task_metric_dims") or cfg.env.xy_dims),
    )
    # Prove the shared cache is actually backing the loader rather than merely present.
    cache_metrics = getattr(train_ds, "cache_metrics", {"cache/consumed": 0.0})
    print(f"[treewm] dataset backend={getattr(train_ds, 'cache_backend', '?')} "
          f"cache={cache_metrics}", flush=True)
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
    model.set_gradient_checkpointing(gradient_checkpointing)
    base_tree_cfg = cfg_utils.tree_config(cfg)
    tree_cfg = tree_config_for(cfg.arm, base_tree_cfg, model)
    # Small, uniform tree for gain-head supervision; evaluation still uses tree_cfg.
    gain_tree_cfg = tree_config_for(
        cfg.arm, replace(base_tree_cfg, node_budget=int(cfg.train.gain_tree_budget)), model
    )
    match_cfg = cfg_utils.matching_config(cfg)
    loss_cfg = cfg_utils.loss_config(cfg)
    planner_cfg = cfg_utils.planner_config(cfg)
    # The v2 scorer creates its set-attention modules lazily. This must precede both
    # optimiser construction and checkpoint restore so parameters/state are identical.
    model.gain_head.set_set_aware(bool(loss_cfg.gain_set_context))
    if (
        objective_version == "treewm_v2_rms_rank_v1"
        and str(loss_cfg.control_objective) != "bootstrap"
    ):
        # TreeSignature is exclusively the bootstrap target. Formal v2 uses the
        # data-derived future-set objective, so retaining trainable parameters here
        # would create an always-unreachable optimiser entry.
        model.tree_signature.requires_grad_(False)
    if loss_cfg.gain_set_context and float(loss_cfg.gain_branch_prior_weight) == 0.0:
        # V2 never advertises or optimises an untrained context-free prior. Keep the
        # module only for v1 checkpoint shape compatibility, but make it explicitly
        # unreachable by the optimiser when disabled.
        model.heads.gain_head.requires_grad_(False)
    if objective_version == "treewm_v2_rms_rank_v1" and not loss_cfg.on("mass"):
        # The formal frontier scorer never consumes predicted mode frequency. Leaving
        # this auxiliary head trainable would perturb the shared branch trunk without
        # changing inference, so v2 removes that causal confound explicitly.
        model.heads.mass_head.requires_grad_(False)

    # Four isolated streams so diagnostics cannot perturb training or planning.
    rng = RngStreams(seed=int(cfg.seed), device=device)
    model._horizon_gen = make_generator(int(cfg.seed), "train", device)

    include_branch_prior = float(loss_cfg.gain_branch_prior_weight) > 0
    world_parameters, gain_parameters = gradient_parameter_groups(
        model, include_branch_prior=include_branch_prior
    )
    separate_gain_clip = bool(cfg.train.get("separate_gain_grad_clip", False))
    if total_steps == 1_000_000 and objective_version == "treewm_v2_rms_rank_v1":
        v2_contract = formal_v2_objective_contract(
            model,
            loss_cfg,
            match_cfg,
            future_cfg,
            tree_cfg,
            separate_gain_clip=separate_gain_clip,
        )
        violations = [name for name, passed in v2_contract.items() if not passed]
        if bool(cfg.retrieval.enabled) or int(cfg.retrieval.num_keys) != 0:
            violations.append("unused_latent_retrieval_disabled")
        if violations:
            raise ValueError(
                "formal v2 objective contract failed: " + ", ".join(violations)
            )
    optimizer_parameters = (
        [{"params": world_parameters}, {"params": gain_parameters}]
        if separate_gain_clip
        else model.parameters()
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler = build_scheduler(optimizer, cfg)
    use_bf16 = bool(cfg.train.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # ------------------------------------------------- retrieval / gain target
    retrieval_target_active = (
        loss_cfg.on("expand") and str(loss_cfg.gain_target) == "retrieval"
    )
    if retrieval_target_active:
        quantizer = StateQuantizer(
            resolution=float(cfg.retrieval.grid_resolution), dims=tuple(cfg.env.xy_dims)
        )
        retrieval_cfg = cfg_utils.retrieval_config(cfg)
        if not retrieval_cfg.enabled:
            raise ValueError("retrieval gain target requires retrieval.enabled=true")
        key_idx = sample_key_indices(len(train_ds.obs_norm), retrieval_cfg, int(cfg.seed))
        endpoint_cells, endpoint_valid = compute_endpoint_cells(
            train_ds.obs_norm,
            train_ds.index,
            quantizer,
            int(cfg.retrieval.endpoint_horizon),
            key_idx=key_idx,
        )
        latent_index = LatentIndex(
            train_ds.obs_norm, endpoint_cells, endpoint_valid,
            retrieval_cfg, device, seed=int(cfg.seed), key_idx=key_idx,
        )
    else:
        quantizer = None
        latent_index = None
    training_step_model = TrainingStepModule(
        model=model,
        loss_cfg=loss_cfg,
        match_cfg=match_cfg,
        gain_tree_cfg=gain_tree_cfg,
        latent_index=latent_index,
        quantizer=quantizer,
        train_cfg=cfg.train,
        model_cfg=cfg.model,
        losses_cfg=cfg.losses,
    )
    ddp_model = training_step_model
    if is_distributed():
        # Some heads are deliberately stride-gated or ablated. All differentiable
        # losses nevertheless live inside TrainingStepModule.forward so DDP can safely
        # determine which parameters participated in this update.
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            training_step_model,
            device_ids=[dist_info.local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    from treewm.evaluation.domains import get_domain
    from treewm.evaluation.tasks import has_maze

    domain = get_domain(cfg.env.name)
    maze_spec = MazeSpec.from_env(env) if has_maze(env) else None
    anchors = tv.build_anchors(maze_spec, num=int(cfg.train.viz_anchors)) if maze_spec else None
    tasks = build_tasks(
        env, str(cfg.eval.task_split), int(cfg.eval.num_hard_tasks),
        float(cfg.eval.hard_percentile), int(cfg.eval.seed),
    )
    task_ids = [int(task.get("task_id", i + 1)) for i, task in enumerate(tasks)]
    final_episodes_per_task = int(cfg.eval.final_episodes_per_task)
    if final_episodes_per_task <= 0:
        raise ValueError("eval.final_episodes_per_task must be positive")
    if total_steps == 1_000_000 and (
        str(cfg.arm) != "treewm"
        or model.__class__.__name__ != "TreeWM"
        or str(tree_cfg.scorer) != "learned"
        or int(cfg.tree.node_budget) != 64
        or int(model.cfg.branch_factor) != 4
        or not gradient_checkpointing
        or bool(cfg.future_sets.cache)
        or not bool(cfg.future_sets.shared_cache)
        or task_ids != [1, 2, 3, 4, 5]
        or final_episodes_per_task != 50
        or scheduler_total_steps != 1_000_000
    ):
        raise ValueError(
            "formal 1M TreeWM requires arm=treewm, model_class=TreeWM, scorer=learned, "
            "node_budget=64, branch_factor=4, gradient checkpointing, "
            "future_sets.cache=false/shared_cache=true, built-in task IDs 1..5, "
            "50 final episodes per task, and a 1M scheduler horizon"
        )

    # ---------------------------------------------------------- identity/resume
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    identity_config = copy.deepcopy(resolved_config)
    # How this invocation found the checkpoint is mutable lifecycle state, not a
    # scientific hyperparameter. Everything else is locked across requeues.
    identity_config["resume"] = None
    repo_root = Path(__file__).resolve().parents[1]
    code_fingerprint = trainer_code_fingerprint(repo_root)
    runtime = runtime_fingerprint()
    protocol_sha256 = os.environ.get("TREEWM_PROTOCOL_SHA256") or str(
        cfg.get("protocol_sha256", "")
    )
    calibration_sha256 = os.environ.get("TREEWM_CALIBRATION_SHA256", "")
    future_recipe_sha256 = os.environ.get("TREEWM_FUTURE_RECIPE_SHA256", "")
    injected_code_sha256 = os.environ.get("TREEWM_CODE_SHA256")
    injected_runtime_sha256 = os.environ.get("TREEWM_RUNTIME_SHA256")
    if injected_code_sha256 and injected_code_sha256 != code_fingerprint["manifest_sha256"]:
        raise ValueError("TREEWM_CODE_SHA256 does not match the live TreeWM source manifest")
    if injected_runtime_sha256 and injected_runtime_sha256 != runtime["sha256"]:
        raise ValueError("TREEWM_RUNTIME_SHA256 does not match the live runtime")
    if total_steps == 1_000_000:
        required_hashes = required_formal_provenance_hashes(
            objective_version,
            protocol_sha256=protocol_sha256,
            code_sha256=injected_code_sha256,
            runtime_sha256=injected_runtime_sha256,
            calibration_sha256=calibration_sha256,
            future_recipe_sha256=future_recipe_sha256,
        )
        malformed = malformed_sha256_names(required_hashes)
        if malformed:
            raise ValueError(
                "formal 1M runs require valid injected provenance hashes: "
                + ", ".join(malformed)
            )
    dataset_reported_sha256 = str(
        getattr(train_ds, "manifest_sha256", getattr(train_ds, "source_manifest_sha256", ""))
    )
    injected_data_sha256 = os.environ.get("TREEWM_DATA_SHA256")
    if (
        injected_data_sha256
        and dataset_reported_sha256
        and injected_data_sha256 != dataset_reported_sha256
    ):
        raise ValueError("TREEWM_DATA_SHA256 does not match the loaded dataset manifest")
    data_manifest_sha256 = injected_data_sha256 or dataset_reported_sha256
    if total_steps == 1_000_000 and (
        len(data_manifest_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in data_manifest_sha256)
    ):
        raise ValueError("formal 1M runs require a validated TREEWM_DATA_SHA256")
    dataset_calibration_sha256 = str(getattr(train_ds, "calibration_sha256", ""))
    dataset_future_recipe_sha256 = str(getattr(train_ds, "future_recipe_sha256", ""))
    if objective_version == "treewm_v2_rms_rank_v1":
        if dataset_calibration_sha256 != calibration_sha256:
            raise ValueError(
                "TREEWM_CALIBRATION_SHA256 does not match the loaded future recipe"
            )
        if dataset_future_recipe_sha256 != future_recipe_sha256:
            raise ValueError(
                "TREEWM_FUTURE_RECIPE_SHA256 does not match the loaded future recipe"
            )
    wandb_project = os.environ.get("WANDB_PROJECT", "treewm")
    wandb_entity = os.environ.get("WANDB_ENTITY", "")
    wandb_group = os.environ.get("WANDB_RUN_GROUP", "")
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    if total_steps == 1_000_000 and wandb_mode in {"offline", "disabled"}:
        raise ValueError("formal 1M runs require online W&B mode")
    run_identity = {
        "schema_version": 1,
        "objective_version": objective_version,
        "run_dir": str(run_dir),
        "run_name": str(explicit_run_name or run_dir.name),
        "arm": str(cfg.arm),
        "env_name": str(cfg.env.name),
        "setting": str(cfg.env.short_name),
        "dataset_kind": str(cfg.env.get("dataset_kind", "standard")),
        "source_name": str(cfg.env.get("source_name", cfg.env.name)),
        "seed": int(cfg.seed),
        "total_steps": total_steps,
        "scheduler_total_steps": scheduler_total_steps,
        "world_size": dist_info.world_size,
        "model_class": model.__class__.__name__,
        "scorer": str(tree_cfg.scorer),
        "node_budget": int(cfg.tree.node_budget),
        "branch_factor": int(model.cfg.branch_factor),
        "gradient_checkpointing": gradient_checkpointing,
        "future_set_cache": bool(cfg.future_sets.cache),
        "shared_cache": bool(cfg.future_sets.shared_cache),
        "retrieval_enabled": bool(cfg.retrieval.enabled),
        "retrieval_num_keys": int(cfg.retrieval.num_keys),
        "task_ids": task_ids,
        "final_episodes_per_task": final_episodes_per_task,
        "config_sha256": _stable_hash(identity_config),
        "protocol_sha256": protocol_sha256,
        "code_sha256": code_fingerprint["manifest_sha256"],
        "runtime_sha256": runtime["sha256"],
        "data_manifest_sha256": data_manifest_sha256,
        "calibration_sha256": calibration_sha256,
        "future_recipe_sha256": future_recipe_sha256,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_group": wandb_group,
        "wandb_mode": wandb_mode,
    }
    wandb_id = os.environ.get("WANDB_RUN_ID") or _stable_hash(run_identity)[:32]
    run_identity["wandb_id"] = wandb_id
    identity_sha256 = _stable_hash(run_identity)

    if completion_path.exists():
        with completion_path.open("r", encoding="utf-8") as handle:
            completion = json.load(handle)
        completion_metrics = completion.get("final_evaluation") or {}
        expected_final_episodes = len(task_ids) * final_episodes_per_task
        if (
            completion.get("schema_version") != 1
            or completion.get("objective_version", "treewm_v1") != objective_version
            or completion.get("calibration_sha256", "") != calibration_sha256
            or completion.get("future_recipe_sha256", "") != future_recipe_sha256
            or completion.get("status") != "complete"
            or completion.get("run_identity") != run_identity
            or completion.get("identity_sha256") != identity_sha256
            or int(completion.get("completed_updates", -1)) != total_steps
            or int(completion.get("scheduler_total_steps", -1)) != scheduler_total_steps
            or int(completion.get("final_eval_step", -1)) != total_steps
            or completion.get("arm") != str(cfg.arm)
            or completion.get("model_class") != model.__class__.__name__
            or completion.get("scorer") != str(tree_cfg.scorer)
            or int(completion.get("branch_factor", -1)) != int(model.cfg.branch_factor)
            or completion.get("task_ids") != task_ids
            or int(completion.get("episodes_per_task", -1)) != final_episodes_per_task
            or int(completion.get("node_budget", -1)) != int(cfg.tree.node_budget)
            or completion.get("gradient_checkpointing") != gradient_checkpointing
            or completion.get("future_set_cache") != bool(cfg.future_sets.cache)
            or completion.get("shared_cache") != bool(cfg.future_sets.shared_cache)
            or completion.get("retrieval_enabled") != bool(cfg.retrieval.enabled)
            or int(completion.get("retrieval_num_keys", -1)) != int(cfg.retrieval.num_keys)
            or int(completion_metrics.get("eval/num_episodes", -1)) != expected_final_episodes
            or any(
                int(completion_metrics.get(f"eval/task{task_id}/num_episodes", -1))
                != final_episodes_per_task
                or f"eval/task{task_id}/success_rate" not in completion_metrics
                or f"eval/task{task_id}/successes" not in completion_metrics
                for task_id in task_ids
            )
        ):
            raise ValueError(f"completion sentinel does not match requested run: {completion_path}")
        final_progress_path = run_dir / str(completion.get("final_eval_progress", ""))
        if not final_progress_path.is_file():
            raise ValueError(f"completion sentinel has no final-eval artifact: {final_progress_path}")
        with final_progress_path.open("r", encoding="utf-8") as handle:
            final_progress = json.load(handle)
        if (
            final_progress.get("status") != "complete"
            or final_progress.get("identity_sha256") != identity_sha256
            or len(final_progress.get("completed_results", [])) != expected_final_episodes
            or final_progress.get("metrics") != completion_metrics
        ):
            raise ValueError(f"final-eval artifact is incomplete or inconsistent: {final_progress_path}")
        if dist_info.is_main:
            print(f"[treewm] already complete: {completion_path}", flush=True)
        barrier()
        cleanup_distributed()
        return

    completed_updates = 0
    pending_eval_step = None
    final_eval = None
    phase = "train"
    resume_path = None
    resume_setting = cfg.resume or ("auto" if total_steps == 1_000_000 else None)
    if resume_setting == "auto":
        candidate = run_dir / "checkpoints" / "latest.pt"
        resume_path = candidate if candidate.exists() else None
    elif resume_setting:
        resume_path = Path(str(resume_setting)).expanduser().resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"requested resume checkpoint does not exist: {resume_path}")

    resume_payload = None
    rank_resume_state = None
    if resume_path is not None:
        resume_payload = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
            map_location="cpu",
            restore_rng=False,
            rank=dist_info.rank,
            expected_identity=run_identity,
        )
        completed_updates = int(resume_payload.get("completed_updates", -1))
        if not 0 <= completed_updates <= total_steps:
            raise ValueError(f"invalid completed_updates in checkpoint: {completed_updates}")
        if int(resume_payload.get("step", -1)) != completed_updates:
            raise ValueError("checkpoint step/completed_updates mismatch")
        rank_states = resume_payload.get("rank_states") or []
        rank_resume_state = next(
            (state for state in rank_states if int(state.get("rank", -1)) == dist_info.rank),
            None,
        )
        if rank_resume_state is None:
            raise ValueError(f"checkpoint has no exact state for rank {dist_info.rank}")
        train_iter.load_state_dict(rank_resume_state.get("loader", {}))
        rng.load_state_dict(rank_resume_state.get("rng_streams", {}))
        horizon_state = rank_resume_state.get("horizon_generator")
        if horizon_state is not None:
            model._horizon_gen.set_state(horizon_state.detach().cpu())
        saved_latent_index = resume_payload.get("latent_index")
        if latent_index is None:
            if saved_latent_index not in (None, {}):
                raise ValueError("checkpoint unexpectedly contains a disabled latent index")
        else:
            latent_index.load_state_dict(saved_latent_index or {})
        ckpt.load_state_dict(resume_payload.get("checkpoint_manager", {}))
        pending_eval_step = resume_payload.get("pending_eval_step")
        final_eval = resume_payload.get("final_eval")
        phase = str(resume_payload.get("phase", "train"))
        if phase not in {"train", "final_eval"}:
            raise ValueError(f"invalid checkpoint phase: {phase}")

    logger = TreeWMLogger(
        run_dir,
        is_main=dist_info.is_main,
        wandb_project=wandb_project,
        wandb_id=wandb_id,
        wandb_name=str(explicit_run_name or run_dir.name),
        wandb_group=wandb_group or None,
        wandb_config=resolved_config,
    )
    # Initialisation of datasets/models/loggers may consume global RNG. Restore the
    # per-rank checkpoint stream last so the next training batch/update is exact.
    if rank_resume_state is not None:
        set_rng_state(
            rank_resume_state["rng_state"], strict_cuda=(total_steps == 1_000_000)
        )
        if dist_info.is_main:
            print(
                f"[treewm] resumed {completed_updates} completed updates from {resume_path}; "
                f"next zero-based update is {completed_updates}",
                flush=True,
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
            "meta/wandb_id": wandb_id,
            "meta/identity_sha256": identity_sha256,
            "meta/calibration_sha256": calibration_sha256 or "not-applicable",
            "meta/future_recipe_sha256": future_recipe_sha256 or "not-applicable",
            "meta/gradient_checkpointing": "enabled",
            **{f"meta/{k}": v for k, v in info.items()},
        }.items():
            logger.text(key, str(value))
        logger.scalar("meta/num_parameters", count_parameters(model), 0)
        logger.scalar("meta/world_size", dist_info.world_size, 0)
        logger.scalar("meta/seed", int(cfg.seed), 0)
        print(f"[treewm] parameters: {count_parameters(model)/1e6:.2f}M | scorer={tree_cfg.scorer}")

    # -------------------------------------------------------------- train loop
    tracker = MetricTracker(device)
    accum = max(1, int(cfg.train.grad_accum))
    world_grad_clip = float(cfg.train.get("world_grad_clip", cfg.train.grad_clip))
    gain_grad_clip = float(cfg.train.get("gain_grad_clip", cfg.train.grad_clip))
    if world_grad_clip <= 0 or gain_grad_clip <= 0:
        raise ValueError("world_grad_clip and gain_grad_clip must be positive")

    def local_rank_state() -> dict:
        return {
            "rank": dist_info.rank,
            "rng_state": get_rng_state(),
            "loader": train_iter.state_dict(),
            "rng_streams": rng.state_dict(),
            "horizon_generator": model._horizon_gen.get_state().detach().cpu(),
        }

    def save_training_checkpoint(reason: str) -> None:
        """Collect rank-local cursors/RNG and atomically commit on rank zero."""
        rank_states = gather_rank_objects(local_rank_state(), destination=0)
        save_failed = False
        if dist_info.is_main:
            try:
                ckpt.save_latest(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=completed_updates,
                    epoch=train_iter.epoch,
                    config=resolved_config,
                    extra={
                        "completed_updates": completed_updates,
                        "next_step": completed_updates,
                        "reason": reason,
                        "run_identity": run_identity,
                        "identity_sha256": identity_sha256,
                        "rank_states": rank_states,
                        "checkpoint_manager": ckpt.state_dict(),
                        "normalizer": normalizer.state_dict(),
                        "latent_index": (
                            latent_index.state_dict() if latent_index is not None else None
                        ),
                        "pending_eval_step": pending_eval_step,
                        "final_eval": final_eval,
                        "phase": phase,
                        "gradient_checkpointing": gradient_checkpointing,
                    },
                )
                print(
                    f"[treewm] checkpointed {completed_updates} completed updates ({reason})",
                    flush=True,
                )
            except BaseException as exc:
                save_failed = True
                print(f"[treewm] checkpoint save failed: {exc!r}", file=sys.stderr, flush=True)
        barrier()
        if any_rank_true(save_failed, device):
            raise RuntimeError(f"failed to save collective checkpoint ({reason})")

    def raise_if_stopping() -> None:
        if any_rank_true(stop.requested, device):
            reason = stop.reason or "signal-on-peer"
            save_training_checkpoint(f"graceful-stop:{reason}")
            if dist_info.is_main:
                logger.flush()
                try:
                    logger.mark_preempting()
                except Exception as exc:
                    print(f"[treewm] W&B preemption mark failed: {exc}", flush=True)
            logger.close(exit_code=GRACEFUL_EXIT_CODE)
            barrier()
            cleanup_distributed()
            raise SystemExit(GRACEFUL_EXIT_CODE)

    def run_synchronized_evaluation(eval_step: int) -> None:
        """Keep peer ranks parked while rank zero evaluates, with resumable intent."""
        nonlocal pending_eval_step
        pending_eval_step = int(eval_step)
        save_training_checkpoint("evaluation-pending")
        eval_failed = False
        if dist_info.is_main:
            try:
                model.eval()
                planner = GoalPlanner(
                    model,
                    normalizer,
                    tree_cfg,
                    planner_cfg,
                    device,
                    generator=rng.reset("eval"),
                    domain=domain,
                )
                emetrics = evaluate(
                    env,
                    planner,
                    tasks,
                    int(cfg.eval.episodes_per_task),
                    int(cfg.planner.max_env_steps),
                    int(cfg.eval.seed),
                    domain=domain,
                    stop_callback=lambda: stop.requested,
                )
                emetrics.update(cache_metrics)
                emetrics.update(resource_metrics())
                logger.scalars(emetrics, eval_step)
                ckpt.maybe_save_success(
                    emetrics["eval/success_rate"],
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=completed_updates,
                    epoch=train_iter.epoch,
                    config=resolved_config,
                    extra={
                        "completed_updates": completed_updates,
                        "run_identity": run_identity,
                        "normalizer": normalizer.state_dict(),
                        "gradient_checkpointing": gradient_checkpointing,
                    },
                )
                print(
                    f"[treewm] step {eval_step} success={emetrics['eval/success_rate']:.3f}"
                )
            except EvaluationInterrupted:
                eval_failed = True
            except BaseException as exc:
                eval_failed = True
                print(f"[treewm] evaluation failed: {exc!r}", file=sys.stderr, flush=True)
        barrier()
        if any_rank_true(eval_failed, device):
            if any_rank_true(stop.requested, device):
                raise_if_stopping()
            raise RuntimeError(f"evaluation failed at step {eval_step}")
        raise_if_stopping()
        pending_eval_step = None
        save_training_checkpoint("evaluation-complete")

    if phase == "train" and pending_eval_step is not None:
        run_synchronized_evaluation(int(pending_eval_step))

    progress = tqdm(
        range(completed_updates, total_steps), initial=completed_updates, total=total_steps,
        disable=not dist_info.is_main, desc=cfg.arm,
    )
    last_log = time.perf_counter()
    examples_since_log = 0
    # Time blocked waiting on the dataloader. High values mean retrieval, not the GPU,
    # is the bottleneck -- which is exactly what the retrieval_pool calibration fixed.
    data_wait = 0.0

    for step in progress:
        raise_if_stopping()
        ddp_model.train()
        if latent_index is not None:
            latent_index.refresh(model.encoder, step)
        optimizer.zero_grad(set_to_none=True)

        for micro in range(accum):
            _t_wait = time.perf_counter()
            batch = to_device(next(train_iter), device)
            data_wait += time.perf_counter() - _t_wait
            sync_context = (
                ddp_model.no_sync()
                if is_distributed() and micro < accum - 1
                else nullcontext()
            )
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
                    loss, metrics, artifacts = ddp_model(batch, step, rng.planner)
                if any_rank_true(not objective_finite(loss), device):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"non-finite backward objective before update {step}"
                    )
                (loss / accum).backward()
            tracker.add_many(metrics, count=batch["obs"].shape[0])
            examples_since_log += batch["obs"].shape[0]

            if micro == accum - 1 and (step + 1) % int(cfg.train.hist_every) == 0:
                tracker.add_hist("tree/keep_scores", artifacts["keep"])
                if loss_cfg.on("mass"):
                    tracker.add_hist("tree/mass_scores", artifacts["mass"])
                tracker.add_hist("tree/uncertainty", artifacts["uncertainty"])
                if include_branch_prior:
                    tracker.add_hist("tree/expansion_gain", artifacts["gain_prior"])
                tracker.add_hist("tree/predicted_horizons", artifacts["horizon_pred"].float())
                tracker.add_hist(
                    "tree/effective_branching_factor_hist", (artifacts["keep"] > 0.5).float().sum(-1)
                )

        if any_rank_true(not gradients_finite(model.parameters()), device):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"non-finite gradient before update {step}")

        log_gradient_modules = (step + 1) % int(cfg.train.log_every) == 0
        if log_gradient_modules:
            for name, value in module_gradient_norms(model, include_branch_prior).items():
                tracker.add(name, value)
        if separate_gain_clip:
            world_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    world_parameters, world_grad_clip, error_if_nonfinite=True
                )
            )
            gain_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    gain_parameters, gain_grad_clip, error_if_nonfinite=True
                )
            )
            tracker.add("train/grad_norm_world", world_norm)
            tracker.add("train/grad_norm_gain", gain_norm)
            grad_norm = math.sqrt(world_norm**2 + gain_norm**2)
            tracker.add(
                "train/grad_clip_coefficient_world",
                min(1.0, world_grad_clip / max(world_norm, 1e-12)),
            )
            tracker.add(
                "train/grad_clip_coefficient_gain",
                min(1.0, gain_grad_clip / max(gain_norm, 1e-12)),
            )
        else:
            if log_gradient_modules:
                tracker.add("train/grad_norm_world", gradient_l2_norm(world_parameters))
                tracker.add("train/grad_norm_gain", gradient_l2_norm(gain_parameters))
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(cfg.train.grad_clip),
                    error_if_nonfinite=True,
                )
            )
        optimizer.step()
        scheduler.step()
        completed_updates = step + 1
        log_step = completed_updates
        tracker.add("train/grad_norm", float(grad_norm))
        tracker.add("train/learning_rate", scheduler.get_last_lr()[0])
        tracker.add("train/weight_decay", float(cfg.train.weight_decay))
        # Signals delivered during the update are handled only after the optimizer and
        # scheduler have committed the same absolute update on every rank.
        raise_if_stopping()

        # ------------------------------------------------------------- logging
        if completed_updates % int(cfg.train.log_every) == 0:
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
            # Emitted every log_every steps, not only at eval: scripts/fleet.py bin-packs
            # from these, and a short resource probe never reaches an eval.
            scalars.update(resource_metrics())
            scalars["train/data_wait_frac"] = float(data_wait / max(elapsed, 1e-6))
            print(f"data_wait_frac={scalars['train/data_wait_frac']:.3f}", flush=True)
            data_wait = 0.0
            logger.scalars(scalars, log_step)
            logger.histograms(tracker.histograms(), log_step)
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
        if completed_updates % int(cfg.train.diag_every) == 0:
            model.eval()
            with torch.no_grad():
                dbatch = to_device(next(iter(val_loader)), device)
                dmetrics = {}
                dmetrics.update(diag.q_vs_z_retrieval(model, dbatch))
                dmetrics.update(diag.branching_diversity_correlation(model, dbatch))
                # geometry_sanity validates decoded positions against maze cells; there
                # are no cells in cube/scene/puzzle. Same maze assumption that had to be
                # guarded in the visualisation block -- it leaks in more than one place.
                if maze_spec is not None:
                    dmetrics.update(diag.geometry_sanity(model, dbatch, maze_spec, normalizer))
                # Retrieval-independent cross-check for the non-maze families; returns {}
                # where no actionable object position exists in the observation.
                dmetrics.update(diag.interaction_sanity(model, dbatch, domain, normalizer))
            logger.scalars({k: all_reduce_mean(v, device) for k, v in dmetrics.items()}, log_step)

        # --------------------------------------------------------- validation
        if completed_updates % int(cfg.train.ckpt_every) == 0:
            val_tracker = MetricTracker(device)
            model.eval()
            # `compute_branch_losses` samples control/recursive subsets even under
            # no_grad. A different validation/checkpoint cadence (the 5k pilot uses a
            # tighter one) must not perturb subsequent training RNG or parameters.
            with preserve_global_rng_state(strict_cuda=(total_steps == 1_000_000)):
                with torch.no_grad():
                    for i, vbatch in enumerate(val_loader):
                        if i >= int(cfg.train.val_batches):
                            break
                        vbatch = to_device(vbatch, device)
                        _, vmetrics, _ = compute_branch_losses(
                            model,
                            vbatch,
                            loss_cfg,
                            match_cfg,
                            step=completed_updates,
                        )
                        vmetrics["train/objective_matches_train_branch"] = 1.0
                        vmetrics["train/objective_includes_gain"] = 0.0
                        val_tracker.add_many(
                            {
                                k.replace("train/", "val/"): v
                                for k, v in vmetrics.items()
                                if "loss" in k or "objective_" in k
                            },
                            count=vbatch["obs"].shape[0],
                        )
            vscalars = val_tracker.compute(reduce=True)
            logger.scalars(vscalars, log_step)
            if dist_info.is_main:
                payload = dict(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    step=completed_updates,
                    epoch=train_iter.epoch, config=OmegaConf.to_container(cfg, resolve=True),
                    extra={
                        "completed_updates": completed_updates,
                        "run_identity": run_identity,
                        "checkpoint_manager": ckpt.state_dict(),
                        "normalizer": normalizer.state_dict(),
                        "gradient_checkpointing": gradient_checkpointing,
                    },
                )
                ckpt.maybe_save_val_loss(vscalars.get("val/loss_total", float("inf")), **payload)
            save_training_checkpoint("periodic-validation")

        # ---------------------------------------------------- goal evaluation
        if completed_updates % int(cfg.train.eval_every) == 0:
            run_synchronized_evaluation(log_step)

        # ------------------------------------------------------ visualisations
        # The xy tree renders are maze-specific. Non-spatial domains (cube, scene,
        # puzzle) get their own diagnostics in treewm/evaluation/domain_viz.py instead of
        # a meaningless projection onto the first two observation dims.
        if should_visualise(log_step, cfg) and dist_info.is_main and maze_spec is not None:
            model.eval()
            try:
                logger.figure(
                    "viz/branching_factor_heatmap",
                    diag.branching_factor_heatmap(model, maze_spec, normalizer, device), log_step,
                )
                if include_branch_prior:
                    logger.figure(
                        "viz/expansion_gain_heatmap",
                        diag.expansion_gain_heatmap(model, maze_spec, normalizer, device), log_step,
                    )
                with torch.no_grad():
                    vbatch = to_device(next(iter(val_loader)), device)
                    fig = diag.q_pca_plot(model, vbatch, normalizer, maze_spec)
                    if fig is not None:
                        logger.figure("viz/q_pca", fig, log_step)
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
                        logger.figure(f"viz/tree_xy_depth/{nm}", tv.view_depth(r, maze_spec, nm), log_step)
                        logger.figure(f"viz/tree_xy_expansion_order/{nm}",
                                      tv.view_expansion_order(r, maze_spec, nm), log_step)
                        logger.figure(f"viz/tree_xy_goal_distance/{nm}",
                                      tv.view_goal_distance(r, maze_spec, nm), log_step)
                        logger.figure(f"viz/tree_xy_root_subtree/{nm}",
                                      tv.view_root_subtree(r, maze_spec, nm), log_step)
                        logger.figure(f"viz/tree_horizon/{nm}", tv.view_horizon(r, maze_spec, nm), log_step)
                        logger.figure(f"viz/tree_selected_path/{nm}",
                                      tv.view_selected_path(r, maze_spec, nm), log_step)
                        logger.scalars(tstats.structural_summary(tree, model, normalizer), log_step)
                        logger.histogram(f"tree/horizon_hist/{nm}",
                                         tree.action_mask.sum(-1)[tree.valid].float(), log_step)
            except Exception as exc:  # visualisation must never kill a run
                print(f"[treewm] visualisation skipped at step {log_step}: {exc}")
        if should_visualise(log_step, cfg) and maze_spec is not None:
            barrier()
            raise_if_stopping()

        # ---------------------------------------- non-spatial domain diagnostics
        # Cube/scene/puzzle have no maze floor to draw on, and projecting their
        # observations onto obs[:2] would render the robot's joint angles -- visually
        # plausible, completely uninformative. Instead render the quantity the task
        # actually constrains, plus a scalar test of whether the K branches are
        # genuinely different futures or have collapsed onto one continuation.
        if should_visualise(log_step, cfg) and dist_info.is_main and maze_spec is None:
            model.eval()
            try:
                from treewm.evaluation import domain_viz as dvz

                for ti, task in enumerate(tasks[:int(cfg.train.viz_anchors)]):
                    ob0, info0 = env.reset(options={"task_id": int(task["task_id"])},
                                           seed=int(cfg.eval.seed) + ti)
                    goal0 = np.asarray(info0["goal"], dtype=np.float32)
                    obs_a = torch.from_numpy(
                        normalizer.norm_obs(np.asarray(ob0, dtype=np.float32)[None])).to(device)
                    goal_a = torch.from_numpy(normalizer.norm_obs(goal0[None])).to(device)
                    tree, _ = model.generate(model.encode(obs_a), tree_cfg, generator=rng.viz)
                    node_obs = model.decoder(tree.latent)
                    gd = torch.linalg.vector_norm(
                        node_obs[..., domain.goal_dims] - goal_a[..., domain.goal_dims].unsqueeze(1),
                        dim=-1)
                    gd = gd.masked_fill(~tree.valid, float("inf")); gd[:, 0] = float("inf")
                    sel = int(gd.argmin(dim=1).item())
                    nm = task.get("task_name", f"task{ti}")

                    if domain.goal_metric == "onehot":
                        logger.figure(f"viz/board_by_depth/{nm}",
                                      dvz.view_board_by_depth(model, tree, normalizer, domain,
                                                              goal0, title=nm), log_step)
                    else:
                        logger.figure(f"viz/object_tree/{nm}",
                                      dvz.view_object_tree(model, tree, normalizer, domain,
                                                           goal0, title=nm, selected=sel), log_step)
                    if ti == 0:
                        logger.scalars(dvz.branch_divergence(model, tree, normalizer, domain), log_step)
                        logger.scalars(tstats.structural_summary(tree, model, normalizer), log_step)
                        logger.histogram("tree/horizon_hist",
                                         tree.action_mask.sum(-1)[tree.valid].float(), log_step)
            except Exception as exc:
                print(f"[treewm] domain visualisation skipped at step {log_step}: {exc}")
        if should_visualise(log_step, cfg) and maze_spec is None:
            barrier()
            raise_if_stopping()

    # ------------------------------------------------------------ final eval
    if completed_updates != total_steps:
        raise RuntimeError(
            f"training stopped at {completed_updates}, expected exactly {total_steps} updates"
        )
    phase = "final_eval"
    pending_eval_step = total_steps
    save_training_checkpoint("final-evaluation-pending")

    final_progress_path = run_dir / "final_eval_progress.json"
    final_failed = False

    def run_final_evaluation() -> None:
        """Run/resume rank-zero terminal evaluation behind one failure boundary."""
        nonlocal final_eval
        completed_results: list[dict] = []
        saved_generator_state = None
        if final_progress_path.exists():
            with final_progress_path.open("r", encoding="utf-8") as handle:
                progress_state = json.load(handle)
            if (
                progress_state.get("identity_sha256") != identity_sha256
                or progress_state.get("task_ids")
                != [int(task.get("task_id", i + 1)) for i, task in enumerate(tasks)]
                or int(progress_state.get("episodes_per_task", -1))
                != final_episodes_per_task
            ):
                raise ValueError(
                    f"final evaluation progress does not match requested run: {final_progress_path}"
                )
            completed_results = list(progress_state.get("completed_results", []))
            saved_generator_state = progress_state.get("generator_state")

        model.eval()
        planner = GoalPlanner(
            model,
            normalizer,
            tree_cfg,
            planner_cfg,
            device,
            generator=rng.reset("eval"),
            domain=domain,
        )
        if saved_generator_state is not None:
            planner.generator.set_state(torch.tensor(saved_generator_state, dtype=torch.uint8))

        def persist_final_episode(result: dict) -> None:
            completed_results.append(result)
            atomic_json_dump(
                {
                    "schema_version": 1,
                    "objective_version": objective_version,
                    "status": "in_progress",
                    "identity_sha256": identity_sha256,
                    "task_ids": [
                        int(task.get("task_id", i + 1)) for i, task in enumerate(tasks)
                    ],
                    "episodes_per_task": final_episodes_per_task,
                    "completed_results": completed_results,
                    "generator_state": planner.generator.get_state().cpu().tolist(),
                },
                final_progress_path,
            )

        final_eval = evaluate(
            env,
            planner,
            tasks,
            final_episodes_per_task,
            int(cfg.planner.max_env_steps),
            int(cfg.eval.seed),
            domain=domain,
            stop_callback=lambda: stop.requested,
            completed_results=completed_results,
            episode_callback=persist_final_episode,
        )
        atomic_json_dump(
            {
                "schema_version": 1,
                "objective_version": objective_version,
                "status": "complete",
                "identity_sha256": identity_sha256,
                "task_ids": [
                    int(task.get("task_id", i + 1)) for i, task in enumerate(tasks)
                ],
                "episodes_per_task": final_episodes_per_task,
                "completed_results": completed_results,
                "generator_state": planner.generator.get_state().cpu().tolist(),
                "metrics": _finite_metrics(final_eval),
            },
            final_progress_path,
        )

    if dist_info.is_main:
        try:
            run_final_evaluation()
        except EvaluationInterrupted:
            final_failed = True
        except BaseException as exc:
            final_failed = True
            print(f"[treewm] final evaluation failed: {exc!r}", file=sys.stderr, flush=True)

    barrier()
    if any_rank_true(final_failed, device):
        if any_rank_true(stop.requested, device):
            raise_if_stopping()
        raise RuntimeError("final evaluation failed")
    raise_if_stopping()
    pending_eval_step = None
    save_training_checkpoint("final-evaluation-complete")

    if dist_info.is_main:
        logger.scalars(final_eval, total_steps)
        logger.hparams(
            {
                "arm": str(cfg.arm), "env": str(cfg.env.name), "seed": int(cfg.seed),
                "node_budget": int(cfg.tree.node_budget),
                "branch_factor": int(model.cfg.branch_factor),
                "z_dim": int(cfg.model.z_dim), "q_dim": int(cfg.model.q_dim),
                "coverage_space": str(cfg.losses.coverage_space),
                "context_pooling": str(cfg.tree.context_pooling),
                "scorer": str(tree_cfg.scorer),
                "gradient_checkpointing": gradient_checkpointing,
            },
            {k: v for k, v in final_eval.items() if np.isfinite(v)},
        )
        logger.flush()
    logger.close(exit_code=0)
    barrier()

    completion_failed = False
    if dist_info.is_main:
        try:
            atomic_json_dump(
                {
                    "schema_version": 1,
                    "objective_version": objective_version,
                    "status": "complete",
                    "run_identity": run_identity,
                    "identity_sha256": identity_sha256,
                    "protocol_sha256": protocol_sha256,
                    "code_sha256": code_fingerprint["manifest_sha256"],
                    "runtime_sha256": runtime["sha256"],
                    "runtime": runtime["software"],
                    "data_manifest_sha256": data_manifest_sha256,
                    "calibration_sha256": calibration_sha256,
                    "future_recipe_sha256": future_recipe_sha256,
                    "arm": str(cfg.arm),
                    "model_class": model.__class__.__name__,
                    "scorer": str(tree_cfg.scorer),
                    "setting": str(cfg.env.short_name),
                    "env_name": str(cfg.env.name),
                    "dataset_kind": str(cfg.env.get("dataset_kind", "standard")),
                    "source_name": str(cfg.env.get("source_name", cfg.env.name)),
                    "dataset_dir": str(cfg.env.dataset_dir),
                    "seed": int(cfg.seed),
                    "wandb_id": wandb_id,
                    "wandb_group": wandb_group,
                    "completed_updates": completed_updates,
                    "scheduler_total_steps": scheduler_total_steps,
                    "final_eval_step": total_steps,
                    "task_ids": task_ids,
                    "episodes_per_task": final_episodes_per_task,
                    "node_budget": int(cfg.tree.node_budget),
                    "branch_factor": int(model.cfg.branch_factor),
                    "gradient_checkpointing": gradient_checkpointing,
                    "future_set_cache": bool(cfg.future_sets.cache),
                    "shared_cache": bool(cfg.future_sets.shared_cache),
                    "retrieval_enabled": bool(cfg.retrieval.enabled),
                    "retrieval_num_keys": int(cfg.retrieval.num_keys),
                    "final_evaluation": _finite_metrics(final_eval),
                    "checkpoint": "checkpoints/latest.pt",
                    "final_eval_progress": final_progress_path.name,
                },
                completion_path,
            )
            print(f"[treewm] done. final success={final_eval['eval/success_rate']:.3f}")
        except BaseException as exc:
            completion_failed = True
            print(f"[treewm] completion write failed: {exc!r}", file=sys.stderr, flush=True)
    barrier()
    if any_rank_true(completion_failed, device):
        cleanup_distributed()
        raise RuntimeError("failed to durably write completion sentinel")
    cleanup_distributed()


if __name__ == "__main__":
    main()
