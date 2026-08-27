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
import importlib.util
import json
import math
import os
import random
from dataclasses import replace
import sys
import time
from pathlib import Path
from types import ModuleType

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import default_collate
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treewm.data.ogbench_dataset import build_datasets
from treewm.data.retrieval_index import LatentIndex, compute_endpoint_cells, sample_key_indices
from treewm.data.samplers import (
    InfiniteLoader,
    build_dataloader,
    build_fixed_validation_dataloader,
    to_device,
)
from treewm.evaluation import diagnostics as diag
from treewm.evaluation import tree_stats as tstats
from treewm.evaluation import tree_viz as tv
from treewm.tree.frontier import GOAL_AWARE_SCORERS
from treewm.evaluation.rollout import (
    EvaluationInterrupted,
    build_evaluation_seed_tables,
    evaluate,
)
from treewm.evaluation.tasks import build_tasks, describe_tasks
from treewm.evaluation.coverage import StateQuantizer
from treewm.data.maze_utils import MazeSpec
from treewm.logging.metrics import MetricTracker
from treewm.logging.tensorboard import TreeWMLogger
from treewm.losses.expansion_losses import novelty_gain_loss
from treewm.losses.latent_gauge import LatentGauge
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
    PostUpdateCadenceState,
    StopController,
    atomic_json_dump,
    build_checkpoint,
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


TREEWM_V2_OBJECTIVES = frozenset(
    {
        "treewm_v2_rms_rank_v1",
        "treewm_v2_grounded_pilot_v1",
        "treewm_v2_grounded_formal_v1",
        "treewm_v2_grounded_repair_pilot_v1",
        "treewm_v2_grounded_repair_formal_v1",
        "treewm_v2_grounded_gauge_pilot_v1",
        "treewm_v2_grounded_gauge_pilot_v2",
        "treewm_v2_grounded_gauge_formal_v1",
        "treewm_v2_grounded_executable_prefix_pilot_v1",
    }
)
BOUNDED_PILOT_OBJECTIVES = {
    "treewm_v2_grounded_pilot_v1": 20_000,
    "treewm_v2_grounded_repair_pilot_v1": 25_000,
    "treewm_v2_grounded_gauge_pilot_v1": 25_000,
    "treewm_v2_grounded_gauge_pilot_v2": 25_000,
    "treewm_v2_grounded_executable_prefix_pilot_v1": 25_000,
}
LATENT_GAUGE_OBJECTIVES = frozenset(
    {
        "treewm_v2_grounded_gauge_pilot_v1",
        "treewm_v2_grounded_gauge_pilot_v2",
        "treewm_v2_grounded_gauge_formal_v1",
        "treewm_v2_grounded_executable_prefix_pilot_v1",
    }
)
EXECUTABLE_PREFIX_OBJECTIVES = frozenset(
    {"treewm_v2_grounded_executable_prefix_pilot_v1"}
)
FORMAL_STAGE_UPDATES = frozenset({2_000, 25_000, 100_000, 1_000_000})
GAUGE_PILOT_STAGE_UPDATES = frozenset({5_000, 25_000})
GROUNDED_FORMAL_OBJECTIVES = frozenset(
    {
        "treewm_v2_grounded_formal_v1",
        "treewm_v2_grounded_repair_formal_v1",
        "treewm_v2_grounded_gauge_formal_v1",
    }
)
STRICT_GROUNDED_EXECUTION_FORMAL_OBJECTIVES = frozenset(
    {
        "treewm_v2_grounded_repair_formal_v1",
        "treewm_v2_grounded_gauge_formal_v1",
    }
)

# These two objectives are intentionally package-authorized.  Their scientific
# recipe is selected by an upstream experiment, so composing the Hydra config is
# not sufficient authority to start a formal run.  Keep the labels distinct: a
# gauge-formal receipt must never be mistaken for the older repair-formal gate.
FORMAL_RECIPE_AUTHORIZATION_LABELS = {
    "treewm_v2_grounded_repair_formal_v1": "repaired formal",
    "treewm_v2_grounded_gauge_formal_v1": "Exp22 gauge formal",
}
EXP22_CAMPAIGN_DIRECTORY = "experiments/22-treewm-grounded-gauge-formal-v1"


def validate_objective_version(objective_version: str, total_steps: int) -> None:
    """Reject unknown objectives and prevent diagnostic pilots becoming formal runs."""
    if objective_version not in {"treewm_v1", *TREEWM_V2_OBJECTIVES}:
        raise ValueError(f"unsupported objective_version: {objective_version!r}")
    pilot_cap = BOUNDED_PILOT_OBJECTIVES.get(objective_version)
    if pilot_cap is not None and total_steps > pilot_cap:
        raise ValueError(
            f"{objective_version} is a bounded diagnostic objective: "
            f"train.steps={total_steps} exceeds the {pilot_cap}-update cap"
        )
    if objective_version in GROUNDED_FORMAL_OBJECTIVES and int(total_steps) != 1_000_000:
        raise ValueError(
            f"{objective_version} requires exactly 1,000,000 scientific updates"
        )


def resolve_stage_stop_after(
    objective_version: str,
    total_steps: int,
    value: str | None,
) -> tuple[int, bool]:
    """Resolve registered lifecycle staging without changing scientific identity."""
    if value is None:
        return int(total_steps), False
    if objective_version in GROUNDED_FORMAL_OBJECTIVES:
        allowed_stages = FORMAL_STAGE_UPDATES
        stage_label = "grounded formal"
    elif objective_version in LATENT_GAUGE_OBJECTIVES:
        allowed_stages = GAUGE_PILOT_STAGE_UPDATES
        stage_label = "latent-gauge pilot"
    else:
        raise ValueError(
            "TREEWM_STOP_AFTER_UPDATE is reserved for registered staged objectives"
        )
    stage = int(value)
    if stage not in allowed_stages:
        raise ValueError(
            f"{stage_label} stage limit must be one of {sorted(allowed_stages)}"
        )
    if stage > int(total_steps):
        raise ValueError("stage limit cannot exceed scientific train.steps")
    return stage, True


def validate_formal_recipe_authorization(
    objective_version: str,
    cfg: Mapping[str, object],
    environ: Mapping[str, str] | None = None,
    argv: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Require the package-issued recipe receipt for selected-recipe formals.

    Exp22 is independently rederived from its fixed package, sealed prerequisite
    receipt, canonical manifest run, exact argv, and complete launch environment.
    The older repair formal retains its historical config/environment consistency
    check. Both checks run before loading data or constructing a model.
    """
    label = FORMAL_RECIPE_AUTHORIZATION_LABELS.get(objective_version)
    if label is None:
        return
    environment = os.environ if environ is None else environ
    authorization = {
        "campaign_prerequisite_binding_sha256": str(
            cfg.get("campaign_prerequisite_binding_sha256", "")
        ),
        "campaign_selected_recipe_sha256": str(
            cfg.get("campaign_selected_recipe_sha256", "")
        ),
        "TREEWM_PREREQUISITE_BINDING_SHA256": environment.get(
            "TREEWM_PREREQUISITE_BINDING_SHA256"
        ),
        "TREEWM_SELECTED_RECIPE_SHA256": environment.get(
            "TREEWM_SELECTED_RECIPE_SHA256"
        ),
    }
    malformed = malformed_sha256_names(authorization)
    if malformed:
        raise ValueError(
            f"{label} objective requires sealed prerequisite/recipe hashes: "
            + ", ".join(malformed)
        )
    if (
        authorization["campaign_prerequisite_binding_sha256"]
        != authorization["TREEWM_PREREQUISITE_BINDING_SHA256"]
        or authorization["campaign_selected_recipe_sha256"]
        != authorization["TREEWM_SELECTED_RECIPE_SHA256"]
    ):
        raise ValueError(
            f"{label} prerequisite hashes differ between config and environment"
        )

    if objective_version != "treewm_v2_grounded_gauge_formal_v1":
        return

    repository = Path(__file__).resolve().parents[1]
    package = repository / EXP22_CAMPAIGN_DIRECTORY
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        campaign = _load_exp22_campaign_authority(package)
        manifest = campaign.load_manifest(package / "manifest.json")
        protocol_sha256 = campaign.verify_protocol_lock(package)
        binding = campaign.load_prerequisite_bindings(
            manifest,
            package / "prerequisite_bindings.json",
            verify_external_files=False,
        )
        run_name = environment.get("TREEWM_RUN_NAME", "")
        matches = [
            run for run in campaign.expand_runs(manifest) if run.run_name == run_name
        ]
        if len(matches) != 1:
            raise ValueError(
                "TREEWM_RUN_NAME does not identify one canonical Exp22 manifest run"
            )
        expected = campaign.trainer_command(
            manifest,
            matches[0],
            repo_root=repository,
        )
        expected_argv = list(expected["argv"][2:])
        if actual_argv != expected_argv:
            raise ValueError("trainer arguments differ from canonical Exp22 launch")
        expected_environment = expected.get("environment")
        if not isinstance(expected_environment, Mapping):
            raise ValueError("canonical Exp22 launch environment is malformed")
        for key, value in expected_environment.items():
            if environment.get(str(key)) != str(value):
                raise ValueError(
                    f"trainer environment differs from canonical Exp22 launch: {key}"
                )
        expected_hashes = expected.get("hashes")
        if not isinstance(expected_hashes, Mapping):
            raise ValueError("canonical Exp22 launch hashes are malformed")
        if protocol_sha256 != expected_hashes.get("package_protocol_sha256"):
            raise ValueError("canonical Exp22 protocol hash differs")
        if (
            binding.get("binding_sha256")
            != expected_hashes.get("prerequisite_binding_sha256")
            or binding.get("selected_recipe_sha256")
            != expected_hashes.get("selected_recipe_sha256")
            or authorization["campaign_prerequisite_binding_sha256"]
            != binding.get("binding_sha256")
            or authorization["campaign_selected_recipe_sha256"]
            != binding.get("selected_recipe_sha256")
        ):
            raise ValueError(
                "trainer hashes differ from canonical Exp22 prerequisite receipt"
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"canonical Exp22 launch authorization failed: {exc}") from exc


def _load_exp22_campaign_authority(package: Path) -> ModuleType:
    """Load the fixed package verifier without accepting a caller-selected module."""
    campaign_path = package / "campaign.py"
    if not campaign_path.is_file() or campaign_path.is_symlink():
        raise ValueError("canonical Exp22 campaign verifier is missing or symlinked")
    module_name = "_treewm_exp22_campaign_authority"
    spec = importlib.util.spec_from_file_location(module_name, campaign_path)
    if spec is None or spec.loader is None:
        raise ValueError("canonical Exp22 campaign verifier cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    return module


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


def replay_pending_post_update_cadence(
    cadence: PostUpdateCadenceState,
    pending_eval_step: int | None,
    *,
    run_evaluation,
    run_visualization,
    save_completion,
) -> None:
    """Replay a checkpointed post-update suffix before entering the training range."""
    if pending_eval_step is not None:
        if pending_eval_step != cadence.committed_update:
            raise ValueError("pending evaluation differs from post-update boundary")
        run_evaluation(pending_eval_step)
    elif cadence.replay_action == "evaluation":
        raise ValueError("post-update evaluation replay lacks pending evaluation intent")

    if cadence.replay_action == "visualization":
        update = cadence.committed_update
        run_visualization(update)
        cadence.finish(update)
        save_completion("post-update-cadence-complete")
    if not cadence.complete:
        raise RuntimeError("post-update cadence replay did not reach a complete boundary")


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


@contextmanager
def fixed_validation_rng(seed: int, rank: int = 0, *, strict_cuda: bool = False):
    """Run validation from the same private RNG state at every checkpoint.

    The validation objective subsamples control and recursive targets. Merely restoring
    the training RNG afterward makes that sampling observational, but does not make two
    checkpoints comparable: each would start from a different training RNG state. This
    context both fixes the measurement stream and restores all training streams on exit.
    """
    with preserve_global_rng_state(strict_cuda=strict_cuda):
        effective_seed = int(seed) * 1_000_003 + 30_013 + int(rank)
        random.seed(effective_seed)
        np.random.seed(effective_seed % (2**32))
        torch.manual_seed(effective_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(effective_seed)
        yield


def resolve_validation_sample_seed(train_cfg, model_seed: int) -> int:
    """Resolve the held-out anchor/sampler/RNG seed without changing old configs."""
    configured = train_cfg.get("validation_sample_seed")
    return int(model_seed) if configured is None else int(configured)


def multistep_transition_kwargs(loss_cfg) -> dict[str, object]:
    """One auditable argument bundle shared by training and both held-out rails."""
    return {
        "transition_mode": str(loss_cfg.multistep_transition_mode),
        "grounded_select_action_weight": float(
            loss_cfg.grounded_select_action_weight
        ),
        "grounded_select_endpoint_weight": float(
            loss_cfg.grounded_select_endpoint_weight
        ),
        "grounded_select_horizon_weight": float(
            loss_cfg.grounded_select_horizon_weight
        ),
        "grounded_loss_latent_weight": float(loss_cfg.grounded_loss_latent_weight),
        "grounded_loss_action_weight": float(loss_cfg.grounded_loss_action_weight),
        "grounded_loss_horizon_weight": float(loss_cfg.grounded_loss_horizon_weight),
        "grounded_loss_endpoint_weight": float(
            loss_cfg.grounded_loss_endpoint_weight
        ),
        "grounded_detach_self_fed_parent": bool(
            loss_cfg.grounded_detach_self_fed_parent
        ),
    }


def validate_multistep_transition_configuration(
    objective_version: str,
    loss_cfg,
    model=None,
) -> None:
    """Fail closed before training if the opt-in recursive transition is incoherent."""
    mode = str(loss_cfg.multistep_transition_mode)
    if mode not in {"teacher_action", "grounded_execution_v2"}:
        raise ValueError(f"unsupported multistep_transition_mode: {mode!r}")
    if mode == "teacher_action":
        return
    if objective_version not in TREEWM_V2_OBJECTIVES:
        raise ValueError("grounded_execution_v2 is restricted to TreeWM-v2 objectives")
    if not loss_cfg.on("multistep"):
        raise ValueError("grounded_execution_v2 requires the multistep objective")
    kwargs = multistep_transition_kwargs(loss_cfg)
    weight_names = tuple(name for name in kwargs if name.endswith("_weight"))
    invalid = [
        name
        for name in weight_names
        if not math.isfinite(float(kwargs[name])) or float(kwargs[name]) < 0.0
    ]
    if invalid:
        raise ValueError(
            "grounded recursive weights must be finite and nonnegative: "
            + ", ".join(invalid)
        )
    if sum(float(kwargs[name]) for name in weight_names if "_select_" in name) <= 0:
        raise ValueError("grounded recursive branch selection requires a positive weight")
    if sum(float(kwargs[name]) for name in weight_names if "_loss_" in name) <= 0:
        raise ValueError("grounded recursive objective requires a positive loss weight")
    decoded_active = (
        float(loss_cfg.grounded_select_endpoint_weight) > 0.0
        or float(loss_cfg.grounded_loss_endpoint_weight) > 0.0
    )
    if decoded_active and model is not None and getattr(model, "decoder", None) is None:
        raise ValueError("grounded decoded recursive terms require model.decoder")


def validate_latent_gauge_configuration(
    objective_version: str,
    loss_cfg,
) -> None:
    """Bind the new gauge graph to its fresh bounded v2 identity only."""
    active = loss_cfg.on("latent_gauge")
    registered = objective_version in LATENT_GAUGE_OBJECTIVES
    if active and not registered:
        raise ValueError("latent_gauge is restricted to a registered TreeWM-v2 objective")
    if not registered:
        return
    configured_weight = float(loss_cfg.weights.latent_gauge)
    configured_enabled = bool(loss_cfg.enabled.get("latent_gauge", False))
    if (configured_enabled, configured_weight) not in {(False, 0.0), (True, 1.0)}:
        raise ValueError(
            "a registered gauge objective requires either monitor-only false/0.0 "
            "or active true/1.0"
        )
    if active and float(loss_cfg.scale("latent_gauge", 0)) != 1.0:
        raise ValueError("latent-gauge regularization must be active from update zero")
    if active and (
        int(loss_cfg.warmup.get("latent_gauge", 0)) != 0
        or int(loss_cfg.decay.get("latent_gauge", 0)) != 0
    ):
        raise ValueError("latent-gauge regularization cannot warm up or decay")
    if (
        not math.isfinite(float(loss_cfg.latent_gauge_epsilon))
        or float(loss_cfg.latent_gauge_epsilon) <= 0.0
    ):
        raise ValueError("latent_gauge_epsilon must be finite and positive")
    if (
        not math.isfinite(float(loss_cfg.latent_gauge_min_reference_scale))
        or float(loss_cfg.latent_gauge_min_reference_scale)
        <= float(loss_cfg.latent_gauge_epsilon)
    ):
        raise ValueError(
            "latent_gauge_min_reference_scale must exceed latent_gauge_epsilon"
        )


def validate_executable_prefix_configuration(
    objective_version: str,
    loss_cfg,
    future_cfg,
    planner_cfg,
    *,
    tree_cfg=None,
    action_space=None,
    model=None,
) -> None:
    """Fail closed around the one registered prospective prefix objective."""

    names = (
        "executable_prefix_action",
        "executable_prefix_latent",
        "executable_prefix_endpoint",
    )
    enabled = tuple(bool(loss_cfg.enabled.get(name, False)) for name in names)
    weights = tuple(float(getattr(loss_cfg.weights, name)) for name in names)
    loss_bounds = (
        loss_cfg.executable_action_lower_bound,
        loss_cfg.executable_action_upper_bound,
    )
    planner_bounds = (
        planner_cfg.action_lower_bound,
        planner_cfg.action_upper_bound,
    )
    registered = objective_version in EXECUTABLE_PREFIX_OBJECTIVES
    if not registered:
        if (
            any(enabled)
            or any(weight != 0.0 for weight in weights)
            or int(future_cfg.executable_prefix_steps) != 0
            or any(value is not None for value in (*loss_bounds, *planner_bounds))
            or any(
                int(loss_cfg.warmup.get(name, 0)) != 0
                or int(loss_cfg.decay.get(name, 0)) != 0
                for name in names
            )
        ):
            raise ValueError(
                "executable-prefix data/loss/planner fields are restricted to a "
                "registered bounded objective"
            )
        return

    if not all(enabled):
        raise ValueError(
            "registered executable-prefix objective requires all three components enabled"
        )
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("executable-prefix weights must be finite and nonnegative")
    # The paired causal pilot uses the exact same graph in both cells. Its package will
    # hash-bind the outcome-blind gradient-audited positive tuple; core accepts that
    # future tuple without blessing a provisional numerical choice here.
    if not (
        all(weight == 0.0 for weight in weights)
        or all(weight > 0.0 for weight in weights)
    ):
        raise ValueError(
            "registered executable-prefix weights must be all-zero monitor-only "
            "or an all-positive treatment tuple"
        )
    if any(
        int(loss_cfg.warmup.get(name, 0)) != 0
        or int(loss_cfg.decay.get(name, 0)) != 0
        for name in names
    ):
        raise ValueError("executable-prefix component weights cannot warm up or decay")
    if int(future_cfg.executable_prefix_steps) != 4:
        raise ValueError(
            "registered executable-prefix objective requires a four-step target"
        )
    sealed_horizons = (4, 8, 16, 32, 64)
    if (
        tuple(int(value) for value in future_cfg.horizons) != sealed_horizons
        or int(future_cfg.h_max) != 64
    ):
        raise ValueError(
            "registered executable-prefix objective requires sealed formal horizons/h_max"
        )
    if model is None or getattr(model, "decoder", None) is None:
        raise ValueError("registered executable-prefix objective requires model.decoder")
    model_cfg = getattr(model, "cfg", None)
    model_horizons = getattr(model, "horizons", None)
    if (
        model_cfg is None
        or tuple(int(value) for value in getattr(model_cfg, "horizons", ()))
        != sealed_horizons
        or int(getattr(model_cfg, "h_max", -1)) != 64
        or model_horizons is None
        or tuple(
            int(value)
            for value in torch.as_tensor(model_horizons).detach().cpu().tolist()
        )
        != sealed_horizons
    ):
        raise ValueError(
            "registered executable-prefix objective requires aligned sealed model horizons"
        )
    if (
        tree_cfg is None
        or int(getattr(model_cfg, "max_depth", -1)) != 3
        or int(getattr(tree_cfg, "max_depth", -1)) != 3
    ):
        raise ValueError(
            "registered executable-prefix objective requires aligned model/tree depth 3"
        )

    prefix_steps = int(future_cfg.executable_prefix_steps)
    minimum_improvement = float(planner_cfg.min_first_edge_improvement)
    if (
        str(planner_cfg.score_space) != "decoded"
        or str(planner_cfg.decoded_metric) != "domain_raw"
        or str(planner_cfg.execute_mode) != "clipped"
        or int(planner_cfg.execute_steps) != prefix_steps
        or not bool(planner_cfg.require_first_edge_improvement)
        or not math.isfinite(minimum_improvement)
        or minimum_improvement < 0.0
    ):
        raise ValueError(
            "registered executable-prefix objective requires decoded domain_raw planning, "
            "clipped execution at the supervised prefix, and a finite nonnegative "
            "first-edge improvement guard"
        )

    if any(value is None for value in (*loss_bounds, *planner_bounds)):
        raise ValueError(
            "registered executable-prefix objective requires sealed train/planner bounds"
        )
    numeric_loss_bounds = tuple(float(value) for value in loss_bounds)
    numeric_planner_bounds = tuple(float(value) for value in planner_bounds)
    if (
        not all(
            math.isfinite(value)
            for value in (*numeric_loss_bounds, *numeric_planner_bounds)
        )
        or numeric_loss_bounds[0] >= numeric_loss_bounds[1]
        or numeric_planner_bounds != numeric_loss_bounds
    ):
        raise ValueError("executable-prefix train/planner action bounds are invalid or differ")

    if (
        action_space is None
        or not hasattr(action_space, "low")
        or not hasattr(action_space, "high")
    ):
        raise ValueError(
            "executable-prefix objective requires sealed environment action bounds"
        )
    environment_low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    environment_high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    model_action_dim = int(getattr(model_cfg, "action_dim", -1))
    if (
        environment_low.size == 0
        or environment_low.size != model_action_dim
        or environment_low.shape != environment_high.shape
        or not np.isfinite(environment_low).all()
        or not np.isfinite(environment_high).all()
        or not np.all(environment_low < environment_high)
        or not np.array_equal(
            environment_low,
            np.full_like(environment_low, numeric_loss_bounds[0]),
        )
        or not np.array_equal(
            environment_high,
            np.full_like(environment_high, numeric_loss_bounds[1]),
        )
    ):
        raise ValueError(
            "sealed executable-prefix bounds do not equal the environment action space"
        )


def heldout_multistep_validation(
    model,
    batch: dict[str, torch.Tensor],
    depth_weights: tuple[float, ...] | None,
    transition_kwargs: Mapping[str, object] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Measure both teacher-forced and fully self-fed recursive generalisation.

    The teacher-forced loss remains the validation counterpart of the train objective
    and is therefore the tensor returned for objective assembly.  The separate p=1
    diagnostic exposes compounding recursive error without changing that objective.
    Validation runs under :func:`fixed_validation_rng`, so both measurements use an
    identical private stream at every checkpoint and cannot perturb training RNG.
    """
    shared_transition_kwargs = dict(transition_kwargs or {})
    teacher_loss, teacher_metrics = multi_step_recursive_loss(
        model,
        batch,
        scheduled_sampling_p=0.0,
        depth_weights=depth_weights,
        **shared_transition_kwargs,
    )
    self_fed_loss, self_fed_metrics = multi_step_recursive_loss(
        model,
        batch,
        scheduled_sampling_p=1.0,
        depth_weights=depth_weights,
        **shared_transition_kwargs,
    )
    metrics: dict[str, torch.Tensor | float] = {
        "train/loss_multistep_teacher_forced": teacher_loss,
        "train/loss_multistep_self_fed": self_fed_loss,
        "train/objective_multistep_teacher_forced": 1.0,
        "train/objective_multistep_self_fed_diagnostic": 1.0,
    }
    for key, value in teacher_metrics.items():
        if key.startswith("recursive/loss_depth"):
            depth = key.removeprefix("recursive/loss_depth")
            # Preserve the pre-existing per-depth validation names.
            metrics[f"train/loss_multistep_depth{depth}"] = value
        elif key.startswith("recursive/grounded/"):
            suffix = key.removeprefix("recursive/grounded/")
            metrics[f"train/grounded/multistep_teacher_forced/{suffix}"] = value
    for key, value in self_fed_metrics.items():
        if key.startswith("recursive/loss_depth"):
            depth = key.removeprefix("recursive/loss_depth")
            metrics[f"train/loss_multistep_self_fed_depth{depth}"] = value
        elif key.startswith("recursive/grounded/"):
            suffix = key.removeprefix("recursive/grounded/")
            metrics[f"train/grounded/multistep_self_fed/{suffix}"] = value
    return teacher_loss, metrics


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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
    if objective_version in TREEWM_V2_OBJECTIVES:
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


def add_validation_label_metrics(
    tracker: MetricTracker,
    batch: dict[str, torch.Tensor],
    horizons: tuple[int, ...],
    max_modes: int,
) -> None:
    """Accumulate exact label distributions for the fixed validation sample."""
    valid = batch["fut_valid"] > 0
    valid_count = float(valid.sum().item())
    if valid_count:
        labels = batch["fut_horizon_idx"].long()
        for index, horizon in enumerate(horizons):
            count = float(((labels == index) & valid).sum().item())
            tracker.add(
                f"data/validation_horizon_label_fraction_h{int(horizon)}",
                count / valid_count,
                count=valid_count,
            )
    if "num_modes" in batch:
        modes = batch["num_modes"].long().clamp(0, int(max_modes))
        anchor_count = float(modes.numel())
        for value in range(int(max_modes) + 1):
            count = float((modes == value).sum().item())
            tracker.add(
                f"data/validation_num_modes_fraction_{value}",
                count / max(anchor_count, 1.0),
                count=anchor_count,
            )


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


def split_branch_transformer_parameters(
    model,
    world_parameters,
) -> tuple[list, list]:
    """Split the recurrent transformer from the existing non-gain world group.

    The input order is retained in both outputs so optimizer parameter ordering is
    deterministic and exact-resume checks can reject any configuration drift.
    """
    branch_ids = {
        id(parameter)
        for parameter in model.branch_transformer.parameters()
        if parameter.requires_grad
    }
    branch_parameters = [
        parameter for parameter in world_parameters if id(parameter) in branch_ids
    ]
    world_rest = [
        parameter for parameter in world_parameters if id(parameter) not in branch_ids
    ]
    if not branch_parameters:
        raise ValueError("separate branch-transformer clipping found no trainable parameters")
    if len(world_rest) + len(branch_parameters) != len(world_parameters):
        raise RuntimeError("branch-transformer parameter split is not exhaustive")
    return world_rest, branch_parameters


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
    expected_depth: int = 16,
    require_grounded_multistep: bool = False,
    required_scheduled_sampling_granularity: str | None = None,
    required_multistep_transition_mode: str | None = None,
    train_cfg=None,
) -> dict[str, bool]:
    """Pure, testable formal-v2 contract evaluated before optimiser construction."""
    if required_scheduled_sampling_granularity not in {None, "step", "sequence"}:
        raise ValueError(
            "required_scheduled_sampling_granularity must be None, 'step', or 'sequence'"
        )
    if required_multistep_transition_mode not in {
        None,
        "teacher_action",
        "grounded_execution_v2",
    }:
        raise ValueError(
            "required_multistep_transition_mode must be None, 'teacher_action', "
            "or 'grounded_execution_v2'"
        )
    contract = {
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
        == int(expected_depth),
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
        "gain_rank_only": float(loss_cfg.gain_calibration_weight) == 0,
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
        **(
            {
                "grounded_multistep_active": loss_cfg.on("multistep"),
                "grounded_multistep_depth_three": int(future_cfg.multi_step_depth) == 3,
                "grounded_multistep_weights": tuple(loss_cfg.multistep_depth_weights)
                == (1.0, 1.0, 1.0),
                "scheduled_sampling_active": float(loss_cfg.scheduled_sampling_p) > 0.0,
            }
            if require_grounded_multistep
            else {}
        ),
    }
    if train_cfg is not None:
        configured_gain_lr = getattr(train_cfg, "gain_lr", None)
        configured_gain_weight_decay = getattr(train_cfg, "gain_weight_decay", None)
        configured_gain_scorers = getattr(train_cfg, "gain_training_scorers", None)
        contract.update(
            {
                "gain_update_every_step": int(train_cfg.gain_loss_every) == 1,
                "dedicated_gain_learning_rate": configured_gain_lr is not None
                and float(configured_gain_lr) == 3.0e-4,
                "zero_gain_weight_decay": configured_gain_weight_decay is not None
                and float(configured_gain_weight_decay) == 0.0,
                "mixed_gain_training_behavior": tuple(configured_gain_scorers or ())
                == ("learned", "novelty_q"),
            }
        )
    if required_scheduled_sampling_granularity is not None:
        # Existing formal identities pass None and retain their historical contract.
        # A fresh objective can bind the revised sampling semantics explicitly.
        contract["scheduled_sampling_granularity"] = (
            str(loss_cfg.scheduled_sampling_granularity)
            == required_scheduled_sampling_granularity
        )
    if required_multistep_transition_mode is not None:
        # Existing formal identities pass None. A fresh v2 identity can bind the
        # grounded execution semantics without retroactively changing sealed runs.
        contract["multistep_transition_mode"] = (
            str(loss_cfg.multistep_transition_mode)
            == required_multistep_transition_mode
        )
    return contract


def repaired_formal_recipe_contract(loss_cfg, train_cfg, factorial_arm: str) -> dict[str, bool]:
    """Require the exp16 arm label and its exact registered repaired loss scale."""
    loss_weights = (
        float(loss_cfg.grounded_loss_latent_weight),
        float(loss_cfg.grounded_loss_action_weight),
        float(loss_cfg.grounded_loss_horizon_weight),
        float(loss_cfg.grounded_loss_endpoint_weight),
    )
    weights_by_arm = {
        "exp16-F": (0.25, 0.5, 0.25, 0.5),
        "exp16-H": (0.125, 0.25, 0.125, 0.25),
    }
    return {
        "balanced_keep_supervision": bool(loss_cfg.keep_balance),
        "grounded_selector_weights": (
            float(loss_cfg.grounded_select_action_weight),
            float(loss_cfg.grounded_select_endpoint_weight),
            float(loss_cfg.grounded_select_horizon_weight),
        )
        == (1.0, 1.0, 0.25),
        "registered_full_or_half_grounded_loss_weights": loss_weights
        in set(weights_by_arm.values()),
        "selected_arm_matches_grounded_loss_weights": weights_by_arm.get(factorial_arm)
        == loss_weights,
        "detached_self_fed_parent": bool(
            loss_cfg.grounded_detach_self_fed_parent
        ),
        "registered_world_learning_rate": float(train_cfg.lr) == 3.0e-5,
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
                scheduled_sampling_granularity=str(
                    self.loss_cfg.scheduled_sampling_granularity
                ),
                depth_weights=self.loss_cfg.multistep_depth_weights or None,
                **multistep_transition_kwargs(self.loss_cfg),
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
                configured_gain_training_scorers = getattr(
                    self.train_cfg, "gain_training_scorers", None
                )
                gain_loss, gain_metrics = novelty_gain_loss(
                    self.model,
                    artifacts["z"][:n_gain],
                    self.gain_tree_cfg,
                    space=str(self.model_cfg.novelty_space),
                    generator=planner_generator,
                    rank_weight=float(self.loss_cfg.gain_rank_weight),
                    calibration_weight=float(self.loss_cfg.gain_calibration_weight),
                    branch_prior_weight=float(self.loss_cfg.gain_branch_prior_weight),
                    training_scorers=(
                        tuple(configured_gain_training_scorers)
                        if configured_gain_training_scorers is not None
                        else None
                    ),
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
    validate_objective_version(objective_version, total_steps)
    # Exp22 is authorized only by the exact launch rederived from its sealed package.
    # Reject a bare/directly fabricated composition before any dataset/model work.
    # The repair-formal check retains its historical position and error ordering.
    if objective_version == "treewm_v2_grounded_gauge_formal_v1":
        validate_formal_recipe_authorization(objective_version, cfg)
    stage_limit_value = os.environ.get("TREEWM_STOP_AFTER_UPDATE")
    stage_stop_after, stage_lifecycle_active = resolve_stage_stop_after(
        objective_version, total_steps, stage_limit_value
    )
    checkpoint_every = int(cfg.train.ckpt_every)
    validation_every = int(
        cfg.train.get("val_every")
        if cfg.train.get("val_every") is not None
        else checkpoint_every
    )
    if validation_every <= 0 or checkpoint_every <= 0:
        raise ValueError("train.val_every and train.ckpt_every must be positive")
    validation_sample_seed = resolve_validation_sample_seed(cfg.train, int(cfg.seed))
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
    executable_domain = None
    executable_dataset_kwargs: dict[str, object] = {}
    if objective_version in EXECUTABLE_PREFIX_OBJECTIVES:
        from treewm.evaluation.domains import get_domain

        executable_domain = get_domain(cfg.env.name)
        executable_dataset_kwargs = {
            "task_goal_metric": str(executable_domain.goal_metric),
            "task_subgoals": tuple(executable_domain.subgoals),
        }
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
        recipe_anchor_policy=str(
            cfg.future_sets.get("recipe_anchor_policy", "selected_seed")
        ),
        validation_sample_seed=validation_sample_seed,
        **executable_dataset_kwargs,
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

    # Separate loader generators: validation must not advance the stream the training
    # loader samples from. The fixed sampler also avoids the historical low-rank bias:
    # val_ds.anchors is sorted, while validation intentionally evaluates only a bounded
    # number of batches.
    train_loader, train_sampler = build_dataloader(
        train_ds, int(cfg.train.batch_size), shuffle=True,
        num_workers=int(cfg.train.num_workers), seed=int(cfg.seed),
        generator=make_generator(int(cfg.seed), "train"),
    )
    val_loader, val_sampler = build_fixed_validation_dataloader(
        val_ds, int(cfg.train.batch_size), int(cfg.train.val_batches),
        num_workers=max(2, int(cfg.train.num_workers) // 4),
        seed=validation_sample_seed,
        generator=make_generator(validation_sample_seed, "viz"),
    )
    # Materialise the first representative batch directly from the sampler once. Every
    # diagnostic checkpoint reuses these exact anchors instead of constructing another
    # iterator (and implicitly selecting whatever happens to be at its prefix).
    diagnostic_positions = val_sampler.local_indices[: int(cfg.train.batch_size)]
    fixed_diagnostic_batch = default_collate(
        [val_ds[int(position)] for position in diagnostic_positions.tolist()]
    )
    fixed_validation_summary = val_sampler.summary()
    fixed_validation_scalars = {
        "data/validation_fixed_sample_count": float(
            fixed_validation_summary["global_sample_size"]
        ),
        "data/validation_fixed_batches_per_rank": float(
            fixed_validation_summary["num_batches"]
        ),
        **{
            f"data/validation_anchor_rank_fraction_{name}": float(value)
            for name, value in fixed_validation_summary[
                "anchor_rank_fraction_quantiles"
            ].items()
        },
    }
    if dist_info.is_main:
        print(f"[treewm] fixed validation sample {fixed_validation_summary}", flush=True)
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
    validate_multistep_transition_configuration(objective_version, loss_cfg, model)
    validate_latent_gauge_configuration(objective_version, loss_cfg)
    if objective_version in LATENT_GAUGE_OBJECTIVES:
        # Attach only for the fresh objective. Existing model/checkpoint state dicts are
        # therefore byte-for-byte unchanged, while gauge references become ordinary
        # persistent model buffers before DDP and checkpoint restore are constructed.
        model.add_module(
            "latent_gauge",
            LatentGauge(
                epsilon=float(loss_cfg.latent_gauge_epsilon),
                min_reference_scale=float(
                    loss_cfg.latent_gauge_min_reference_scale
                ),
            ).to(device),
        )
    planner_cfg = cfg_utils.planner_config(cfg)
    validate_executable_prefix_configuration(
        objective_version,
        loss_cfg,
        future_cfg,
        planner_cfg,
        tree_cfg=tree_cfg,
        action_space=getattr(env, "action_space", None),
        model=model,
    )
    # The v2 scorer creates its set-attention modules lazily. This must precede both
    # optimiser construction and checkpoint restore so parameters/state are identical.
    model.gain_head.set_set_aware(bool(loss_cfg.gain_set_context))
    if (
        objective_version in TREEWM_V2_OBJECTIVES
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
    if objective_version in TREEWM_V2_OBJECTIVES and not loss_cfg.on("mass"):
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
    separate_branch_clip = bool(
        cfg.train.get("separate_branch_transformer_grad_clip", False)
    )
    if separate_branch_clip:
        world_rest_parameters, branch_transformer_parameters = (
            split_branch_transformer_parameters(model, world_parameters)
        )
    else:
        # Aliases only: the disabled branch follows the exact historical optimizer and
        # clipping construction below.
        world_rest_parameters = world_parameters
        branch_transformer_parameters = []
    if total_steps == 1_000_000 and objective_version in TREEWM_V2_OBJECTIVES:
        v2_contract = formal_v2_objective_contract(
            model,
            loss_cfg,
            match_cfg,
            future_cfg,
            tree_cfg,
            separate_gain_clip=separate_gain_clip,
            expected_depth=(
                3 if objective_version in GROUNDED_FORMAL_OBJECTIVES else 16
            ),
            require_grounded_multistep=(
                objective_version in GROUNDED_FORMAL_OBJECTIVES
            ),
            required_scheduled_sampling_granularity=(
                "sequence"
                if objective_version
                in STRICT_GROUNDED_EXECUTION_FORMAL_OBJECTIVES
                else None
            ),
            required_multistep_transition_mode=(
                "grounded_execution_v2"
                if objective_version
                in STRICT_GROUNDED_EXECUTION_FORMAL_OBJECTIVES
                else None
            ),
            train_cfg=cfg.train,
        )
        violations = [name for name, passed in v2_contract.items() if not passed]
        if objective_version == "treewm_v2_grounded_repair_formal_v1":
            repaired_contract = repaired_formal_recipe_contract(
                loss_cfg,
                cfg.train,
                str(cfg.get("campaign_factorial_arm", "")),
            )
            violations.extend(
                name for name, passed in repaired_contract.items() if not passed
            )
        if bool(cfg.retrieval.enabled) or int(cfg.retrieval.num_keys) != 0:
            violations.append("unused_latent_retrieval_disabled")
        if violations:
            raise ValueError(
                "formal v2 objective contract failed: " + ", ".join(violations)
            )
    configured_gain_lr = cfg.train.get("gain_lr")
    configured_gain_weight_decay = cfg.train.get("gain_weight_decay")
    gain_lr = (
        float(configured_gain_lr)
        if configured_gain_lr is not None
        else float(cfg.train.lr)
    )
    gain_weight_decay = (
        float(configured_gain_weight_decay)
        if configured_gain_weight_decay is not None
        else float(cfg.train.weight_decay)
    )
    separate_gain_optimizer = bool(
        separate_gain_clip
        or configured_gain_lr is not None
        or configured_gain_weight_decay is not None
    )
    if separate_branch_clip:
        # Stable group order is part of exact checkpoint resume. The gain group remains
        # explicit even when it happens to share world hyperparameters.
        optimizer_parameters = [
            {
                "params": world_rest_parameters,
                "lr": float(cfg.train.lr),
                "weight_decay": float(cfg.train.weight_decay),
                "name": "world_rest",
            },
            {
                "params": branch_transformer_parameters,
                "lr": float(cfg.train.lr),
                "weight_decay": float(cfg.train.weight_decay),
                "name": "branch_transformer",
            },
            {
                "params": gain_parameters,
                "lr": gain_lr,
                "weight_decay": gain_weight_decay,
                "name": "gain",
            },
        ]
    else:
        optimizer_parameters = (
            [
                {
                    "params": world_parameters,
                    "lr": float(cfg.train.lr),
                    "weight_decay": float(cfg.train.weight_decay),
                    "name": "world",
                },
                {
                    "params": gain_parameters,
                    "lr": gain_lr,
                    "weight_decay": gain_weight_decay,
                    "name": "gain",
                },
            ]
            if separate_gain_optimizer
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

    domain = (
        executable_domain
        if executable_domain is not None
        else get_domain(cfg.env.name)
    )
    maze_spec = MazeSpec.from_env(env) if has_maze(env) else None
    anchors = (
        tv.expand_xy_anchors(
            tv.build_anchors(maze_spec, num=int(cfg.train.viz_anchors)), normalizer
        )
        if maze_spec
        else None
    )
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
    if objective_version in GROUNDED_FORMAL_OBJECTIVES and (
        str(planner_cfg.score_space) != "decoded"
        or str(planner_cfg.decoded_metric) != "domain_raw"
        or str(planner_cfg.execute_mode) != "clipped"
        or int(planner_cfg.execute_steps) != 4
        or not bool(planner_cfg.require_first_edge_improvement)
        or float(planner_cfg.min_first_edge_improvement) < 0.0
        or int(model.cfg.max_depth) != 3
        or int(tree_cfg.max_depth) != 3
    ):
        raise ValueError(
            "grounded formal objectives require decoded domain_raw planning, clipped e4, "
            "the first-edge improvement guard, and aligned model/tree depth 3"
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
    recipe_code_sha256 = os.environ.get(
        "TREEWM_RECIPE_CODE_SHA256", os.environ.get("TREEWM_CODE_SHA256", "")
    )
    recipe_runtime_sha256 = os.environ.get(
        "TREEWM_RECIPE_RUNTIME_SHA256", os.environ.get("TREEWM_RUNTIME_SHA256", "")
    )
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
    if objective_version == "treewm_v2_grounded_repair_formal_v1":
        validate_formal_recipe_authorization(objective_version, cfg)
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
    if objective_version in TREEWM_V2_OBJECTIVES:
        if dataset_calibration_sha256 != calibration_sha256:
            raise ValueError(
                "TREEWM_CALIBRATION_SHA256 does not match the loaded future recipe"
            )
        if dataset_future_recipe_sha256 != future_recipe_sha256:
            raise ValueError(
                "TREEWM_FUTURE_RECIPE_SHA256 does not match the loaded future recipe"
            )
    if objective_version in GROUNDED_FORMAL_OBJECTIVES and (
        str(getattr(train_ds, "recipe_anchor_policy", "")) != "published_union"
        or str(getattr(val_ds, "recipe_anchor_policy", "")) != "published_union"
    ):
        raise ValueError(
            "grounded formal objectives require sealed published-union train/validation anchors"
        )
    wandb_project = os.environ.get("WANDB_PROJECT", "treewm")
    wandb_entity = os.environ.get("WANDB_ENTITY", "")
    wandb_group = os.environ.get("WANDB_RUN_GROUP", "")
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    if total_steps == 1_000_000 and wandb_mode in {"offline", "disabled"}:
        raise ValueError("formal 1M runs require online W&B mode")
    injected_evaluation_seed_protocol = os.environ.get(
        "TREEWM_EVALUATION_SEED_PROTOCOL_SHA256"
    )
    if injected_evaluation_seed_protocol and malformed_sha256_names(
        {"TREEWM_EVALUATION_SEED_PROTOCOL_SHA256": injected_evaluation_seed_protocol}
    ):
        raise ValueError("TREEWM_EVALUATION_SEED_PROTOCOL_SHA256 must be lowercase SHA256")
    evaluation_seed_protocol_sha256 = (
        injected_evaluation_seed_protocol
        or (
            protocol_sha256
            if not malformed_sha256_names({"protocol": protocol_sha256})
            else _stable_hash(
            {
                "namespace": "unsealed-treewm-evaluation-seeds-v1",
                "objective_version": objective_version,
                "env_name": str(cfg.env.name),
                "task_ids": task_ids,
            }
        )
        )
    )
    evaluation_seed_tables = build_evaluation_seed_tables(
        evaluation_seed_protocol_sha256,
        int(cfg.seed),
        task_ids,
        int(cfg.eval.episodes_per_task),
        final_episodes_per_task,
    )
    use_protocol_bound_evaluation_seeds = (
        objective_version in GROUNDED_FORMAL_OBJECTIVES
    )
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
        "evaluation_seed_protocol_sha256": evaluation_seed_protocol_sha256,
        "evaluation_seed_tables_sha256": evaluation_seed_tables["sha256"],
        "monitor_seed_table_sha256": evaluation_seed_tables["monitor"]["sha256"],
        "final_seed_table_sha256": evaluation_seed_tables["final"]["sha256"],
        "config_sha256": _stable_hash(identity_config),
        "protocol_sha256": protocol_sha256,
        "code_sha256": code_fingerprint["manifest_sha256"],
        "runtime_sha256": runtime["sha256"],
        "data_manifest_sha256": data_manifest_sha256,
        "calibration_sha256": calibration_sha256,
        "future_recipe_sha256": future_recipe_sha256,
        "recipe_anchor_policy": str(
            cfg.future_sets.get("recipe_anchor_policy", "selected_seed")
        ),
        "train_anchor_count": int(len(train_ds)),
        "validation_anchor_count": int(len(val_ds)),
        "recipe_code_sha256": recipe_code_sha256,
        "recipe_runtime_sha256": recipe_runtime_sha256,
        # Optional bounded-campaign seals. Empty for historical/general runs; when a
        # campaign injects them they become part of checkpoint/completion identity and
        # cannot drift across requeues.
        "campaign_source_sha256": str(cfg.get("campaign_source_sha256", "")),
        "campaign_protocol_sha256": str(cfg.get("campaign_protocol_sha256", "")),
        "campaign_config_sha256": str(cfg.get("campaign_config_sha256", "")),
        "campaign_input_contract_sha256": str(
            cfg.get("campaign_input_contract_sha256", "")
        ),
        "campaign_factorial_arm": str(cfg.get("campaign_factorial_arm", "")),
        "campaign_prerequisite_binding_sha256": str(
            cfg.get("campaign_prerequisite_binding_sha256", "")
        ),
        "campaign_selected_recipe_sha256": str(
            cfg.get("campaign_selected_recipe_sha256", "")
        ),
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_group": wandb_group,
        "wandb_mode": wandb_mode,
    }
    wandb_id = os.environ.get("WANDB_RUN_ID") or _stable_hash(run_identity)[:32]
    run_identity["wandb_id"] = wandb_id
    identity_sha256 = _stable_hash(run_identity)

    seed_tables_path = run_dir / "evaluation_seed_tables.json"
    seed_table_failed = False
    if dist_info.is_main:
        try:
            if seed_tables_path.exists():
                with seed_tables_path.open("r", encoding="utf-8") as handle:
                    existing_seed_tables = json.load(handle)
                if existing_seed_tables != evaluation_seed_tables:
                    raise ValueError(
                        f"evaluation seed tables differ from existing run: {seed_tables_path}"
                    )
            else:
                atomic_json_dump(evaluation_seed_tables, seed_tables_path)
        except BaseException as exc:
            seed_table_failed = True
            print(
                f"[treewm] evaluation seed-table publication failed: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
    if any_rank_true(seed_table_failed, device):
        raise RuntimeError("failed to publish exact evaluation seed tables")
    barrier()

    final_episode_seeds = (
        evaluation_seed_tables["final"]["seeds"]
        if use_protocol_bound_evaluation_seeds
        else [
            [
                int(cfg.eval.seed) + 1_000 * task_index + episode_index
                for episode_index in range(final_episodes_per_task)
            ]
            for task_index in range(len(task_ids))
        ]
    )
    expected_final_episode_rows = [
        (task_index, task_id, episode_index, int(seed_value))
        for task_index, (task_id, task_seeds) in enumerate(
            zip(task_ids, final_episode_seeds, strict=True)
        )
        for episode_index, seed_value in enumerate(task_seeds)
    ]

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
            or completion.get("evaluation_seed_tables_sha256")
            != evaluation_seed_tables["sha256"]
            or completion.get("final_seed_table_sha256")
            != evaluation_seed_tables["final"]["sha256"]
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
            or final_progress.get("seed_table_sha256")
            != evaluation_seed_tables["final"]["sha256"]
            or len(final_progress.get("completed_results", [])) != expected_final_episodes
            or [
                (
                    row.get("task_index"),
                    row.get("task_id"),
                    row.get("episode_index"),
                    row.get("episode_seed"),
                )
                for row in final_progress.get("completed_results", [])
            ]
            != expected_final_episode_rows
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
    # This is checkpointed for the gauge pilot so a signal between 50-update logging
    # boundaries cannot turn the first post-resume scalar into a partial-window mean.
    tracker = MetricTracker(device)
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
            require_exact_resume=True,
            expected_world_size=dist_info.world_size,
            require_cuda_rng=(
                device.type == "cuda"
                and (
                    total_steps == 1_000_000
                    or objective_version in LATENT_GAUGE_OBJECTIVES
                )
            ),
        )
        completed_updates = int(resume_payload.get("completed_updates", -1))
        if not 0 <= completed_updates <= total_steps:
            raise ValueError(f"invalid completed_updates in checkpoint: {completed_updates}")
        if (
            objective_version in LATENT_GAUGE_OBJECTIVES
            and completed_updates > 0
            and not model.latent_gauge.is_sealed
        ):
            raise ValueError(
                "post-update latent-gauge checkpoint has no sealed initialization reference"
            )
        if completed_updates > stage_stop_after:
            raise ValueError(
                "checkpoint is beyond TREEWM_STOP_AFTER_UPDATE for this lifecycle stage"
            )
        if int(resume_payload.get("step", -1)) != completed_updates:
            raise ValueError("checkpoint step/completed_updates mismatch")
        rank_states = resume_payload.get("rank_states") or []
        rank_resume_state = next(
            (state for state in rank_states if int(state.get("rank", -1)) == dist_info.rank),
            None,
        )
        if rank_resume_state is None:
            raise ValueError(f"checkpoint has no exact state for rank {dist_info.rank}")
        metric_tracker_state = rank_resume_state.get("metric_tracker")
        if metric_tracker_state is None:
            if objective_version in LATENT_GAUGE_OBJECTIVES and completed_updates > 0:
                raise ValueError(
                    "post-update latent-gauge checkpoint has no metric-tracker state"
                )
        else:
            tracker.load_state_dict(metric_tracker_state)
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

    cadence_payload = (
        resume_payload.get("post_update_cadence") if resume_payload is not None else None
    )
    if cadence_payload is None:
        # Historical objectives retain their pre-field resume behavior. Fresh Exp22
        # checkpoints are rejected centrally before state mutation if this is absent.
        post_update_cadence = PostUpdateCadenceState(completed_updates)
    else:
        post_update_cadence = PostUpdateCadenceState.from_state_dict(
            cadence_payload,
            require_durable=True,
        )
        if post_update_cadence.committed_update != completed_updates:
            raise ValueError("checkpoint post-update cadence/update boundary differs")

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
            rank_resume_state["rng_state"],
            strict_cuda=(
                total_steps == 1_000_000
                or objective_version in LATENT_GAUGE_OBJECTIVES
            ),
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
            "meta/recipe_code_sha256": recipe_code_sha256 or "not-applicable",
            "meta/recipe_runtime_sha256": recipe_runtime_sha256 or "not-applicable",
            "meta/gradient_checkpointing": "enabled",
            **{f"meta/{k}": v for k, v in info.items()},
        }.items():
            logger.text(key, str(value))
        logger.scalar("meta/num_parameters", count_parameters(model), 0)
        logger.scalar("meta/world_size", dist_info.world_size, 0)
        logger.scalar("meta/seed", int(cfg.seed), 0)
        logger.scalars(fixed_validation_scalars, 0)
        logger.text(
            "meta/fixed_validation_sample",
            json.dumps(fixed_validation_summary, sort_keys=True, indent=2),
        )
        print(f"[treewm] parameters: {count_parameters(model)/1e6:.2f}M | scorer={tree_cfg.scorer}")

    # -------------------------------------------------------------- train loop
    accum = max(1, int(cfg.train.grad_accum))
    world_grad_clip = float(cfg.train.get("world_grad_clip", cfg.train.grad_clip))
    gain_grad_clip = float(cfg.train.get("gain_grad_clip", cfg.train.grad_clip))
    branch_transformer_grad_clip = float(
        cfg.train.get("branch_transformer_grad_clip", cfg.train.grad_clip)
    )
    if (
        world_grad_clip <= 0
        or gain_grad_clip <= 0
        or branch_transformer_grad_clip <= 0
    ):
        raise ValueError("all configured gradient-clip thresholds must be positive")

    def local_rank_state() -> dict:
        return {
            "rank": dist_info.rank,
            "rng_state": get_rng_state(),
            "loader": train_iter.state_dict(),
            "rng_streams": rng.state_dict(),
            "horizon_generator": model._horizon_gen.get_state().detach().cpu(),
            "metric_tracker": tracker.state_dict(),
        }

    def save_training_checkpoint(
        reason: str,
        *,
        best_success: float | None = None,
        best_val_loss: float | None = None,
    ) -> None:
        """Commit one complete collective boundary to latest and selected best slots."""
        rank_states = gather_rank_objects(local_rank_state(), destination=0)
        save_failed = False
        if dist_info.is_main:
            try:
                improved_success = (
                    ckpt.record_success(best_success)
                    if best_success is not None
                    else False
                )
                improved_val_loss = (
                    ckpt.record_val_loss(best_val_loss)
                    if best_val_loss is not None
                    else False
                )
                payload = build_checkpoint(
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
                        "post_update_cadence": post_update_cadence.state_dict(),
                        "final_eval": final_eval,
                        "phase": phase,
                        "gradient_checkpointing": gradient_checkpointing,
                        "evaluation_seed_tables_sha256": evaluation_seed_tables["sha256"],
                    },
                )
                if improved_success:
                    ckpt.save_best_success_payload(payload)
                if improved_val_loss:
                    ckpt.save_best_val_loss_payload(payload)
                # Publish latest last: it is the authoritative transaction boundary.
                # If a best-slot write fails, the previous latest still causes the
                # metric selection to be retried after resume.
                ckpt.save_latest_payload(payload)
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
            # A checkpoint at N resumes with an empty N-boundary loop. Defer the
            # graceful exit until its deterministic cadence is complete. Replayable
            # evaluation/visualization suffixes are the exceptions: their
            # explicit cadence action makes every omitted outcome resume-visible.
            if not post_update_cadence.stop_checkpoint_is_durable(pending_eval_step):
                return
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

    def run_synchronized_visualization(viz_step: int) -> None:
        """Render every visualization due at one committed update on rank zero."""
        if not should_visualise(viz_step, cfg):
            return
        # The xy tree renders are maze-specific. Non-spatial domains (cube, scene,
        # puzzle) get domain-native diagnostics instead of an obs[:2] projection.
        if dist_info.is_main and maze_spec is not None:
            model.eval()
            try:
                logger.figure(
                    "viz/branching_factor_heatmap",
                    diag.branching_factor_heatmap(model, maze_spec, normalizer, device),
                    viz_step,
                )
                if include_branch_prior:
                    logger.figure(
                        "viz/expansion_gain_heatmap",
                        diag.expansion_gain_heatmap(model, maze_spec, normalizer, device),
                        viz_step,
                    )
                with torch.no_grad():
                    vbatch = to_device(fixed_diagnostic_batch, device)
                    fig = diag.q_pca_plot(model, vbatch, normalizer, maze_spec)
                    if fig is not None:
                        logger.figure("viz/q_pca", fig, viz_step)
                    for a in range(min(int(cfg.train.viz_anchors), len(anchors))):
                        start, goal = anchors.starts[a], anchors.goals[a]
                        obs_a = torch.from_numpy(normalizer.norm_obs(start[None])).to(device)
                        goal_a = torch.from_numpy(normalizer.norm_obs(goal[None])).to(device)
                        tree, _ = model.generate(
                            model.encode(obs_a),
                            tree_cfg,
                            generator=rng.viz,
                            goal_obs=(
                                goal_a
                                if tree_cfg.scorer in GOAL_AWARE_SCORERS
                                else None
                            ),
                        )
                        node_obs = model.decoder(tree.latent)
                        gd = torch.linalg.vector_norm(
                            node_obs[..., domain.goal_dims]
                            - goal_a[..., domain.goal_dims].unsqueeze(1),
                            dim=-1,
                        )
                        gd = gd.masked_fill(~tree.valid, float("inf"))
                        gd[:, 0] = float("inf")
                        rendered = tv.TreeRender.from_tree(
                            model,
                            tree,
                            normalizer,
                            goal,
                            start,
                            0,
                            int(gd.argmin(dim=1).item()),
                        )
                        name = anchors.names[a]
                        logger.figure(
                            f"viz/tree_xy_depth/{name}",
                            tv.view_depth(rendered, maze_spec, name),
                            viz_step,
                        )
                        logger.figure(
                            f"viz/tree_xy_expansion_order/{name}",
                            tv.view_expansion_order(rendered, maze_spec, name),
                            viz_step,
                        )
                        logger.figure(
                            f"viz/tree_xy_goal_distance/{name}",
                            tv.view_goal_distance(rendered, maze_spec, name),
                            viz_step,
                        )
                        logger.figure(
                            f"viz/tree_xy_root_subtree/{name}",
                            tv.view_root_subtree(rendered, maze_spec, name),
                            viz_step,
                        )
                        logger.figure(
                            f"viz/tree_horizon/{name}",
                            tv.view_horizon(rendered, maze_spec, name),
                            viz_step,
                        )
                        logger.figure(
                            f"viz/tree_selected_path/{name}",
                            tv.view_selected_path(rendered, maze_spec, name),
                            viz_step,
                        )
                        logger.scalars(
                            tstats.structural_summary(tree, model, normalizer),
                            viz_step,
                        )
                        logger.histogram(
                            f"tree/horizon_hist/{name}",
                            tree.action_mask.sum(-1)[tree.valid].float(),
                            viz_step,
                        )
            except Exception as exc:  # visualisation must never kill a run
                print(f"[treewm] visualisation skipped at step {viz_step}: {exc}")
        elif dist_info.is_main:
            model.eval()
            try:
                from treewm.evaluation import domain_viz as dvz

                for task_index, task in enumerate(
                    tasks[: int(cfg.train.viz_anchors)]
                ):
                    observation, info = env.reset(
                        options={"task_id": int(task["task_id"])},
                        seed=int(cfg.eval.seed) + task_index,
                    )
                    goal = np.asarray(info["goal"], dtype=np.float32)
                    obs_a = torch.from_numpy(
                        normalizer.norm_obs(
                            np.asarray(observation, dtype=np.float32)[None]
                        )
                    ).to(device)
                    goal_a = torch.from_numpy(normalizer.norm_obs(goal[None])).to(device)
                    tree, _ = model.generate(
                        model.encode(obs_a), tree_cfg, generator=rng.viz
                    )
                    node_obs = model.decoder(tree.latent)
                    gd = torch.linalg.vector_norm(
                        node_obs[..., domain.goal_dims]
                        - goal_a[..., domain.goal_dims].unsqueeze(1),
                        dim=-1,
                    )
                    gd = gd.masked_fill(~tree.valid, float("inf"))
                    gd[:, 0] = float("inf")
                    selected = int(gd.argmin(dim=1).item())
                    name = task.get("task_name", f"task{task_index}")
                    if domain.goal_metric == "onehot":
                        logger.figure(
                            f"viz/board_by_depth/{name}",
                            dvz.view_board_by_depth(
                                model,
                                tree,
                                normalizer,
                                domain,
                                goal,
                                title=name,
                            ),
                            viz_step,
                        )
                    else:
                        logger.figure(
                            f"viz/object_tree/{name}",
                            dvz.view_object_tree(
                                model,
                                tree,
                                normalizer,
                                domain,
                                goal,
                                title=name,
                                selected=selected,
                            ),
                            viz_step,
                        )
                    if task_index == 0:
                        logger.scalars(
                            dvz.branch_divergence(model, tree, normalizer, domain),
                            viz_step,
                        )
                        logger.scalars(
                            tstats.structural_summary(tree, model, normalizer),
                            viz_step,
                        )
                        logger.histogram(
                            "tree/horizon_hist",
                            tree.action_mask.sum(-1)[tree.valid].float(),
                            viz_step,
                        )
            except Exception as exc:
                print(
                    f"[treewm] domain visualisation skipped at step {viz_step}: {exc}"
                )
        barrier()

    def run_synchronized_evaluation(eval_step: int) -> None:
        """Keep peer ranks parked while rank zero evaluates, with resumable intent."""
        nonlocal pending_eval_step
        post_update_cadence.mark_replay("evaluation", eval_step)
        pending_eval_step = eval_step
        save_training_checkpoint("evaluation-pending")
        eval_failed = False
        eval_success = None
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
                    episode_seed_table=(
                        evaluation_seed_tables["monitor"]
                        if use_protocol_bound_evaluation_seeds
                        else None
                    ),
                    expected_episode_seed_split=(
                        "monitor" if use_protocol_bound_evaluation_seeds else None
                    ),
                )
                emetrics.update(cache_metrics)
                emetrics.update(resource_metrics())
                logger.scalars(emetrics, eval_step)
                eval_success = float(emetrics["eval/success_rate"])
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
        pending_eval_step = None
        if should_visualise(eval_step, cfg):
            post_update_cadence.mark_replay("visualization", eval_step)
        else:
            post_update_cadence.finish(eval_step)
        save_training_checkpoint(
            "evaluation-complete",
            best_success=eval_success if dist_info.is_main else None,
        )

    if phase == "train" and (
        pending_eval_step is not None or not post_update_cadence.complete
    ):
        replay_pending_post_update_cadence(
            post_update_cadence,
            pending_eval_step,
            run_evaluation=run_synchronized_evaluation,
            run_visualization=run_synchronized_visualization,
            save_completion=save_training_checkpoint,
        )

    progress = tqdm(
        range(completed_updates, stage_stop_after),
        initial=completed_updates,
        total=stage_stop_after,
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
        if separate_branch_clip:
            if separate_gain_clip:
                world_rest_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        world_rest_parameters,
                        world_grad_clip,
                        error_if_nonfinite=True,
                    )
                )
                gain_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        gain_parameters, gain_grad_clip, error_if_nonfinite=True
                    )
                )
                world_rest_coefficient = min(
                    1.0, world_grad_clip / max(world_rest_norm, 1e-12)
                )
                gain_coefficient = min(
                    1.0, gain_grad_clip / max(gain_norm, 1e-12)
                )
            else:
                # Isolate only the transformer. The remaining historical parameter
                # population retains one common global clipping coefficient.
                world_rest_norm = gradient_l2_norm(world_rest_parameters)
                gain_norm = gradient_l2_norm(gain_parameters)
                non_branch_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        [*world_rest_parameters, *gain_parameters],
                        float(cfg.train.grad_clip),
                        error_if_nonfinite=True,
                    )
                )
                non_branch_coefficient = min(
                    1.0, float(cfg.train.grad_clip) / max(non_branch_norm, 1e-12)
                )
                world_rest_coefficient = non_branch_coefficient
                gain_coefficient = non_branch_coefficient
            branch_transformer_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    branch_transformer_parameters,
                    branch_transformer_grad_clip,
                    error_if_nonfinite=True,
                )
            )
            branch_transformer_coefficient = min(
                1.0,
                branch_transformer_grad_clip
                / max(branch_transformer_norm, 1e-12),
            )
            world_norm = math.sqrt(
                world_rest_norm**2 + branch_transformer_norm**2
            )
            world_coefficient = min(
                world_rest_coefficient, branch_transformer_coefficient
            )
            tracker.add("train/grad_norm_world_rest", world_rest_norm)
            tracker.add(
                "train/grad_norm_branch_transformer", branch_transformer_norm
            )
            tracker.add("train/grad_norm_world", world_norm)
            tracker.add("train/grad_norm_gain", gain_norm)
            tracker.add(
                "train/grad_clip_coefficient_world_rest", world_rest_coefficient
            )
            tracker.add(
                "train/grad_clip_coefficient_branch_transformer",
                branch_transformer_coefficient,
            )
            tracker.add("train/grad_clip_coefficient_world", world_coefficient)
            tracker.add("train/grad_clip_coefficient_gain", gain_coefficient)
            grad_norm = math.sqrt(world_norm**2 + gain_norm**2)
        elif separate_gain_clip:
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
        post_update_cadence.begin(completed_updates)
        log_step = completed_updates
        tracker.add("train/grad_norm", float(grad_norm))
        learning_rates = scheduler.get_last_lr()
        if separate_branch_clip:
            tracker.add("train/learning_rate", learning_rates[0])
            tracker.add("train/learning_rate_branch_transformer", learning_rates[1])
            tracker.add("train/learning_rate_gain", learning_rates[2])
        else:
            tracker.add("train/learning_rate", learning_rates[0])
            tracker.add(
                "train/learning_rate_gain",
                learning_rates[1] if separate_gain_optimizer else learning_rates[0],
            )
        tracker.add("train/weight_decay", float(cfg.train.weight_decay))
        tracker.add("train/weight_decay_gain", gain_weight_decay)
        # A signal delivered during the optimizer step is now latched while every
        # deterministic action due at this absolute update is completed below.

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
                dbatch = to_device(fixed_diagnostic_batch, device)
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
        evaluation_due = completed_updates % int(cfg.train.eval_every) == 0
        visualization_due = should_visualise(log_step, cfg)
        validation_due = completed_updates % validation_every == 0
        checkpoint_due = completed_updates % checkpoint_every == 0
        if validation_due:
            val_tracker = MetricTracker(device)
            val_label_tracker = MetricTracker(device)
            model.eval()
            # `compute_branch_losses` samples control/recursive subsets even under
            # no_grad. A different validation/checkpoint cadence (the 5k pilot uses a
            # tighter one) must not perturb subsequent training RNG or parameters.
            with fixed_validation_rng(
                validation_sample_seed,
                dist_info.rank,
                strict_cuda=(total_steps == 1_000_000),
            ):
                with torch.no_grad():
                    for i, vbatch in enumerate(val_loader):
                        if i >= int(cfg.train.val_batches):
                            break
                        vbatch = to_device(vbatch, device)
                        _, vmetrics, _, validation_branch_terms = compute_branch_losses(
                            model,
                            vbatch,
                            loss_cfg,
                            match_cfg,
                            step=completed_updates,
                            return_loss_terms=True,
                        )
                        validation_raw_terms = dict(validation_branch_terms.raw)
                        if loss_cfg.on("multistep"):
                            # Keep the stable teacher-forced objective and separately
                            # expose fully self-fed compounding error on the same sealed
                            # multi-step recipe targets.
                            validation_multistep, validation_multistep_metrics = (
                                heldout_multistep_validation(
                                    model,
                                    vbatch,
                                    tuple(loss_cfg.multistep_depth_weights) or None,
                                    multistep_transition_kwargs(loss_cfg),
                                )
                            )
                            validation_raw_terms["multistep"] = validation_multistep
                            vmetrics.update(validation_multistep_metrics)
                        validation_terms = assemble_loss_terms(
                            validation_raw_terms,
                            loss_cfg,
                            completed_updates,
                        )
                        # Replace the branch-only total/components with the exact
                        # held-out world objective, including grounded multistep.
                        vmetrics.update(loss_term_metrics(validation_terms))
                        vmetrics["train/objective_matches_train_branch"] = 1.0
                        vmetrics["train/objective_includes_gain"] = 0.0
                        val_tracker.add_many(
                            {
                                k.replace("train/", "val/"): v
                                for k, v in vmetrics.items()
                                if "loss" in k
                                or "objective_" in k
                                or "grounded/" in k
                                or "executable_prefix/" in k
                            },
                            count=vbatch["obs"].shape[0],
                        )
                        add_validation_label_metrics(
                            val_label_tracker,
                            vbatch,
                            tuple(int(value) for value in future_cfg.horizons),
                            int(future_cfg.max_modes),
                        )
            vscalars = val_tracker.compute(reduce=True)
            vscalars.update(val_label_tracker.compute(reduce=True))
            horizon_probabilities = np.asarray(
                [
                    vscalars.get(
                        f"data/validation_horizon_label_fraction_h{int(horizon)}",
                        0.0,
                    )
                    for horizon in future_cfg.horizons
                ],
                dtype=np.float64,
            )
            nonzero = horizon_probabilities[horizon_probabilities > 0]
            entropy = float(-(nonzero * np.log(nonzero)).sum()) if len(nonzero) else 0.0
            entropy_normalizer = (
                math.log(len(future_cfg.horizons))
                if len(future_cfg.horizons) > 1
                else 1.0
            )
            vscalars["data/validation_horizon_label_normalized_entropy"] = (
                entropy / entropy_normalizer
            )
            vscalars.update(fixed_validation_scalars)
            logger.scalars(vscalars, log_step)
            if evaluation_due:
                # Make a validation checkpoint immediately before evaluation replay
                # that outcome if the process is killed before evaluation can start.
                pending_eval_step = completed_updates
                post_update_cadence.mark_replay(
                    "evaluation", completed_updates
                )
            elif visualization_due:
                post_update_cadence.mark_replay(
                    "visualization", completed_updates
                )
            else:
                post_update_cadence.finish(completed_updates)
            save_training_checkpoint(
                "periodic-validation",
                best_val_loss=(
                    vscalars.get("val/loss_total", float("inf"))
                    if dist_info.is_main
                    else None
                ),
            )
        elif checkpoint_due:
            if evaluation_due:
                pending_eval_step = completed_updates
                post_update_cadence.mark_replay(
                    "evaluation", completed_updates
                )
            elif visualization_due:
                post_update_cadence.mark_replay(
                    "visualization", completed_updates
                )
            else:
                post_update_cadence.finish(completed_updates)
            save_training_checkpoint("periodic")

        if not validation_due and not checkpoint_due:
            if evaluation_due:
                pending_eval_step = completed_updates
                post_update_cadence.mark_replay(
                    "evaluation", completed_updates
                )
            elif visualization_due:
                post_update_cadence.mark_replay(
                    "visualization", completed_updates
                )
            else:
                post_update_cadence.finish(completed_updates)

        # ---------------------------------------------------- goal evaluation
        if evaluation_due:
            run_synchronized_evaluation(log_step)

        # ------------------------------------------------------ visualisations
        run_synchronized_visualization(log_step)
        if not post_update_cadence.complete:
            post_update_cadence.finish(completed_updates)
        # This is the sole ordinary post-update graceful-stop boundary. It follows
        # logging, diagnostics, validation, periodic checkpoint/evaluation, and any
        # visualization due for the committed update.
        raise_if_stopping()

    # ------------------------------------------------------- external stage gate
    if not post_update_cadence.complete:
        raise RuntimeError("training loop exited with incomplete post-update cadence")
    # Close the terminal-stage race: a signal received after the final loop-body
    # check is still checkpointed before an empty-loop resume can publish a gate.
    raise_if_stopping()
    if stage_lifecycle_active:
        if completed_updates != stage_stop_after:
            raise RuntimeError(
                f"stage stopped at {completed_updates}, expected {stage_stop_after} updates"
            )
        phase = "train"
        pending_eval_step = None
        save_training_checkpoint("awaiting-external-stage-gate")
        stage_marker_failed = False
        if dist_info.is_main:
            try:
                latest_path = ckpt.directory / "latest.pt"
                atomic_json_dump(
                    {
                        "schema_version": 1,
                        "status": "awaiting_external_stage_gate",
                        "objective_version": objective_version,
                        "completed_updates": completed_updates,
                        "step": completed_updates,
                        "total_steps": total_steps,
                        "scheduler_total_steps": scheduler_total_steps,
                        "identity_sha256": identity_sha256,
                        "checkpoint": "checkpoints/latest.pt",
                        "checkpoint_sha256": _file_sha256(latest_path),
                        "evaluation_seed_tables_sha256": evaluation_seed_tables["sha256"],
                    },
                    run_dir
                    / "stage-gates"
                    / f"AWAITING_GATE_{stage_stop_after}.json",
                )
                logger.flush()
            except BaseException as exc:
                stage_marker_failed = True
                print(
                    f"[treewm] stage-gate marker write failed: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
        barrier()
        if any_rank_true(stage_marker_failed, device):
            cleanup_distributed()
            raise RuntimeError("failed to durably publish stage-gate marker")
        logger.close(exit_code=0)
        barrier()
        cleanup_distributed()
        return

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
                or progress_state.get("seed_table_sha256")
                != evaluation_seed_tables["final"]["sha256"]
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
                    "seed_table_sha256": evaluation_seed_tables["final"]["sha256"],
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
            episode_seed_table=(
                evaluation_seed_tables["final"]
                if use_protocol_bound_evaluation_seeds
                else None
            ),
            expected_episode_seed_split=(
                "final" if use_protocol_bound_evaluation_seeds else None
            ),
        )
        atomic_json_dump(
            {
                "schema_version": 1,
                "objective_version": objective_version,
                "status": "complete",
                "identity_sha256": identity_sha256,
                "seed_table_sha256": evaluation_seed_tables["final"]["sha256"],
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
                    "evaluation_seed_tables": seed_tables_path.name,
                    "evaluation_seed_tables_sha256": evaluation_seed_tables["sha256"],
                    "final_seed_table_sha256": evaluation_seed_tables["final"]["sha256"],
                    "protocol_sha256": protocol_sha256,
                    "code_sha256": code_fingerprint["manifest_sha256"],
                    "runtime_sha256": runtime["sha256"],
                    "runtime": runtime["software"],
                    "data_manifest_sha256": data_manifest_sha256,
                    "calibration_sha256": calibration_sha256,
                    "future_recipe_sha256": future_recipe_sha256,
                    "recipe_code_sha256": recipe_code_sha256,
                    "recipe_runtime_sha256": recipe_runtime_sha256,
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
