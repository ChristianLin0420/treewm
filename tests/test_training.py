"""Losses, DDP-safe metric reduction, checkpoint resume and the synthetic two-mode env."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from treewm.logging.metrics import MetricTracker, rank_correlation
from treewm.evaluation import rollout
from treewm.losses.support_losses import keep_loss, mass_loss, redundancy_loss, support_metrics
from treewm.losses.total import LossConfig
from treewm.losses.world_losses import action_loss, horizon_loss, uncertainty_loss
from treewm.models.baselines import build_model
from treewm.models.treewm import TreeWMConfig, horizon_mask
from treewm.tree.expansion import TreeConfig
from treewm.utils.checkpoint import load_checkpoint, save_checkpoint
from treewm.utils.provenance import trainer_code_fingerprint
from treewm.utils.rng import make_generator

SMALL = TreeWMConfig(obs_dim=1, action_dim=1, z_dim=32, q_dim=16, hidden_dim=64, num_layers=2, branch_factor=2)


def test_formal_requirements_are_explicit_in_base_config():
    cfg = OmegaConf.load(Path(__file__).parents[1] / "configs" / "base.yaml")
    assert cfg.train.gradient_checkpointing is True
    assert cfg.eval.final_episodes_per_task == 50
    assert cfg.losses.scheduled_sampling_granularity == "step"
    assert cfg.train.validation_sample_seed is None
    assert cfg.losses.multistep_transition_mode == "teacher_action"
    assert cfg.losses.grounded_select_action_weight == 0.0
    assert cfg.losses.grounded_select_endpoint_weight == 0.0
    assert cfg.losses.grounded_select_horizon_weight == 0.0
    assert cfg.losses.grounded_loss_latent_weight == 0.0
    assert cfg.losses.grounded_loss_action_weight == 0.0
    assert cfg.losses.grounded_loss_horizon_weight == 0.0
    assert cfg.losses.grounded_loss_endpoint_weight == 0.0
    assert cfg.losses.grounded_detach_self_fed_parent is True


def test_horizon_mask_matches_horizons():
    horizons = torch.tensor([4, 8, 16])
    idx = torch.tensor([[0, 2]])
    mask = horizon_mask(idx, horizons, h_max=16)
    assert mask.shape == (1, 2, 16)
    assert int(mask[0, 0].sum()) == 4
    assert int(mask[0, 1].sum()) == 16


def test_action_loss_ignores_padding():
    pred = torch.zeros(1, 1, 8, 2)
    tgt = torch.zeros(1, 1, 8, 2)
    tgt[0, 0, 4:] = 100.0  # garbage beyond the horizon
    mask = torch.zeros(1, 1, 8)
    mask[0, 0, :4] = 1.0
    matched = torch.ones(1, 1)
    assert float(action_loss(pred, tgt, mask, matched)) == 0.0


def test_losses_only_supervise_matched_branches():
    logits = torch.zeros(1, 3, 5, requires_grad=True)
    target = torch.zeros(1, 3, dtype=torch.long)
    matched = torch.tensor([[1.0, 0.0, 0.0]])
    horizon_loss(logits, target, matched).backward()
    grad = logits.grad[0]
    assert grad[0].abs().sum() > 0, "matched branch must receive gradient"
    assert grad[1].abs().sum() == 0, "unmatched branch must not be supervised"


def test_keep_loss_is_class_balanced():
    """With 3 of 4 branches matched, an unbalanced loss would favour 'keep all'.

    Evaluated at a confident "keep everything" prediction, where the single unmatched
    branch is the minority class: balancing must make that one error dominate, otherwise
    the head drifts to predicting keep=1 everywhere and the effective branching factor
    is pinned at K.
    """
    logit = torch.full((1, 4), 3.0)  # confidently "keep" all four
    matched = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    balanced = keep_loss(logit, matched, balance=True)
    unbalanced = keep_loss(logit, matched, balance=False)
    assert float(balanced) > float(unbalanced), "balancing must up-weight the minority class"

    # At logit=0 every element has the same BCE, so the two agree exactly -- a useful
    # invariant, and the reason this test does not probe at zero.
    assert torch.isclose(
        keep_loss(torch.zeros(1, 4), matched, True), keep_loss(torch.zeros(1, 4), matched, False)
    )


def test_redundancy_penalises_only_kept_duplicates():
    q = torch.zeros(1, 2, 1, 4)  # two identical branches -> maximally redundant
    cdist = lambda a, b: torch.cdist(a.flatten(2), b.flatten(2))
    high = redundancy_loss(q, torch.ones(1, 2), cdist, 0.25)
    low = redundancy_loss(q, torch.zeros(1, 2), cdist, 0.25)
    assert float(high) > float(low), "keeping duplicates must cost more than dropping them"

    distinct = torch.zeros(1, 2, 1, 4)
    distinct[0, 1] = 10.0
    assert float(redundancy_loss(distinct, torch.ones(1, 2), cdist, 0.25)) < float(high)


def test_uncertainty_head_receives_signal():
    sigma = torch.zeros(1, 2, requires_grad=True)
    pred = torch.zeros(1, 2, 4)
    tgt = torch.ones(1, 2, 4)
    matched = torch.ones(1, 2)
    loss = uncertainty_loss(sigma, pred, tgt, matched)
    loss.backward()
    assert sigma.grad.abs().sum() > 0, "sigma must be trained, else UncertaintyTreeWM is random"


def test_support_frequency_separation_metric():
    """A rare-but-valid mode that is matched must count as recalled."""
    keep = torch.tensor([[0.9, 0.9, 0.9, 0.1]])
    matched = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    target_mass = torch.tensor([[0.80, 0.15, 0.05, 0.0]])
    branch_mass = torch.tensor([[0.80, 0.15, 0.05, 0.0]])
    mode_valid = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    mode_to_branch = torch.tensor([[0, 1, 2, -1]])
    q = torch.randn(1, 4, 1, 8)
    z = torch.randn(1, 4, 8)
    cdist = lambda a, b: torch.cdist(a.flatten(2), b.flatten(2))

    m = support_metrics(keep, torch.softmax(branch_mass, -1), matched, target_mass, branch_mass,
                        mode_valid, mode_to_branch, q, z, cdist)
    assert m["tree/rare_mode_recall"] == 1.0, "the 5% mode must be recalled, not pruned"
    assert m["tree/support_recall"] == 1.0
    assert m["tree/effective_branching_factor"] == 3.0


def test_mass_loss_targets_frequency_not_support():
    logit = torch.zeros(1, 3, requires_grad=True)
    target = torch.tensor([[0.8, 0.15, 0.05]])
    matched = torch.ones(1, 3)
    mass_loss(logit, target, matched).backward()
    grad = logit.grad[0]
    # The dominant mode should pull hardest -- mass IS frequency-aware.
    assert grad[0] < grad[1] < grad[2]


def test_metric_tracker_weighted_mean_and_nonfinite():
    t = MetricTracker()
    t.add("a", 1.0, count=1)
    t.add("a", 3.0, count=3)  # weighted mean = (1 + 9) / 4 = 2.5
    t.add("b", float("nan"))
    out = t.compute(reduce=False)
    assert abs(out["a"] - 2.5) < 1e-6
    assert "b" not in out, "non-finite values must not poison the log"
    assert "b__nonfinite" in out


def test_rank_correlation_handles_degenerate_input():
    assert rank_correlation(np.zeros(5), np.arange(5)) == 0.0
    assert rank_correlation(np.arange(5), np.arange(5)) == pytest.approx(1.0)
    assert rank_correlation(np.arange(5), np.arange(5)[::-1]) == pytest.approx(-1.0)
    assert rank_correlation(np.arange(3), np.arange(5)) == 0.0  # mismatched lengths


def test_checkpoint_save_and_exact_resume(tmp_path):
    model = build_model("treewm", SMALL)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 1)
    model.encode(x).sum().backward()
    opt.step()

    identity = {"run": "tiny", "total_steps": 1_000_000}
    torch.manual_seed(1234)
    path = save_checkpoint(
        tmp_path / "latest.pt",
        model=model,
        optimizer=opt,
        step=7,
        epoch=2,
        config={"a": 1},
        extra={"run_identity": identity, "completed_updates": 7, "next_step": 7},
    )
    assert path.exists()
    expected_draw = torch.randn(3)

    restored = build_model("treewm", SMALL)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    payload = load_checkpoint(path, restored, restored_opt, expected_identity=identity)

    assert payload["step"] == 7 and payload["epoch"] == 2 and payload["config"] == {"a": 1}
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), restored.named_parameters()):
        assert n1 == n2 and torch.allclose(p1, p2), f"parameter {n1} did not round-trip"
    assert payload["completed_updates"] == 7 and payload["next_step"] == 7
    assert torch.equal(torch.randn(3), expected_draw)
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(path, expected_identity={**identity, "total_steps": 999_999})


def test_checkpoint_resume_matches_next_stochastic_optimizer_update(tmp_path):
    """The checkpoint boundary means next update, including global/explicit RNG."""
    cfg = replace(SMALL, dropout=0.1, horizon_mode="random")
    torch.manual_seed(23)
    uninterrupted = build_model("treewm", cfg).train()
    uninterrupted.set_gradient_checkpointing(True)
    uninterrupted._horizon_gen = make_generator(23, "train")
    optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=1e-3)

    def update(model, opt, observations):
        opt.zero_grad(set_to_none=True)
        child = model.predict_children(model.encode(observations))
        loss = (
            child["latent"].square().mean()
            + child["q"].square().mean()
            + child["branch"].keep_logit.square().mean()
        )
        loss.backward()
        opt.step()
        return loss.detach(), child["horizon_idx"].detach()

    update(uninterrupted, optimizer, torch.randn(4, cfg.obs_dim))
    identity = {"run": "stochastic", "total_steps": 1_000_000}
    path = save_checkpoint(
        tmp_path / "latest.pt",
        model=uninterrupted,
        optimizer=optimizer,
        step=1,
        extra={
            "run_identity": identity,
            "horizon_generator": uninterrupted._horizon_gen.get_state().clone(),
        },
    )

    expected_batch = torch.randn(4, cfg.obs_dim)
    expected_loss, expected_horizon = update(uninterrupted, optimizer, expected_batch)

    resumed = build_model("treewm", cfg).train()
    resumed.set_gradient_checkpointing(True)
    resumed._horizon_gen = make_generator(23, "train")
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    payload = load_checkpoint(
        path, resumed, resumed_optimizer, expected_identity=identity
    )
    resumed._horizon_gen.set_state(payload["horizon_generator"])
    resumed_batch = torch.randn(4, cfg.obs_dim)
    resumed_loss, resumed_horizon = update(resumed, resumed_optimizer, resumed_batch)

    torch.testing.assert_close(resumed_batch, expected_batch, rtol=0, atol=0)
    torch.testing.assert_close(resumed_horizon, expected_horizon, rtol=0, atol=0)
    torch.testing.assert_close(resumed_loss, expected_loss)
    for expected, actual in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_trainer_fingerprint_includes_hydra_configs(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "treewm").mkdir()
    (tmp_path / "configs" / "model").mkdir(parents=True)
    (tmp_path / "scripts" / "__init__.py").write_bytes(b"")
    (tmp_path / "scripts" / "train.py").write_text("# trainer\n")
    (tmp_path / "treewm" / "model.py").write_text("# model\n")
    config_package = tmp_path / "configs" / "__init__.py"
    config_package.write_bytes(b"")
    config = tmp_path / "configs" / "model" / "treewm.yaml"
    config.write_text("hidden_dim: 256\n")

    before = trainer_code_fingerprint(tmp_path)
    assert "scripts/__init__.py" in before["files"]
    assert "configs/__init__.py" in before["files"]
    assert "configs/model/treewm.yaml" in before["files"]
    config.write_text("hidden_dim: 512\n")
    after = trainer_code_fingerprint(tmp_path)
    assert after["manifest_sha256"] != before["manifest_sha256"]

    config.write_text("hidden_dim: 256\n")
    config_package.write_text("# replacement marker\n")
    replaced = trainer_code_fingerprint(tmp_path)
    assert replaced["manifest_sha256"] != before["manifest_sha256"]


def test_gradient_checkpointing_preserves_forward_and_gradients():
    torch.manual_seed(7)
    ordinary = build_model("treewm", SMALL).train()
    rematerialized = copy.deepcopy(ordinary).train()
    rematerialized.set_gradient_checkpointing(True)
    obs = torch.randn(5, SMALL.obs_dim)

    def loss_of(model):
        branch = model.branch(model.encode(obs))
        return branch.action.square().mean() + branch.keep_logit.square().mean()

    loss_a = loss_of(ordinary)
    loss_b = loss_of(rematerialized)
    loss_a.backward()
    loss_b.backward()
    torch.testing.assert_close(loss_a, loss_b)
    for (name_a, param_a), (name_b, param_b) in zip(
        ordinary.named_parameters(), rematerialized.named_parameters()
    ):
        assert name_a == name_b
        if param_a.grad is None or param_b.grad is None:
            assert param_a.grad is None and param_b.grad is None
        else:
            torch.testing.assert_close(param_a.grad, param_b.grad)


def test_evaluation_emits_per_task_metrics_and_resumes_prefix(monkeypatch):
    calls = []

    def fake_episode(_env, _planner, task, seed, **_kwargs):
        calls.append((task["task_id"], seed))
        return rollout.EpisodeResult(
            success=bool(seed % 2),
            steps=1,
            replans=1,
            nodes=4,
            final_goal_distance=0.5,
            best_goal_distance=0.25,
            progress={"fraction": np.float32(0.5)},
        )

    monkeypatch.setattr(rollout, "run_episode", fake_episode)
    tasks = [{"task_id": 1}, {"task_id": 2}]
    persisted = []
    metrics = rollout.evaluate(
        object(), object(), tasks, episodes_per_task=3, episode_callback=persisted.append
    )
    assert metrics["eval/num_episodes"] == 6
    assert metrics["eval/task1/num_episodes"] == 3
    assert metrics["eval/task2/num_episodes"] == 3
    assert "eval/task1/success_rate" in metrics and "eval/task2/success_rate" in metrics
    json.dumps(persisted)  # progress artifacts must contain only JSON-native values

    calls.clear()
    resumed = rollout.evaluate(
        object(), object(), tasks, episodes_per_task=3, completed_results=persisted[:4]
    )
    assert len(calls) == 2
    assert resumed["eval/num_episodes"] == 6


def test_evaluation_aggregates_no_action_and_guard_diagnostics(monkeypatch):
    results = iter(
        [
            rollout.EpisodeResult(
                success=False,
                steps=0,
                replans=1,
                nodes=8,
                final_goal_distance=2.0,
                best_goal_distance=2.0,
                no_action_plans=1,
                guard_plans=1,
                guard_rejections=1,
                guard_candidate_count=4,
                guard_accepted_count=0,
                guard_best_predicted_improvements=[-0.25],
            ),
            rollout.EpisodeResult(
                success=False,
                steps=4,
                replans=2,
                nodes=16,
                final_goal_distance=1.0,
                best_goal_distance=0.9,
                guard_plans=2,
                guard_candidate_count=8,
                guard_accepted_count=3,
                guard_best_predicted_improvements=[0.5, 0.25],
                guard_selected_predicted_improvements=[0.4, 0.2],
            ),
        ]
    )
    monkeypatch.setattr(rollout, "run_episode", lambda *_args, **_kwargs: next(results))

    metrics = rollout.evaluate(
        object(), object(), [{"task_id": 1}], episodes_per_task=2
    )

    assert metrics["eval/no_action_plan_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["eval/no_action_episode_fraction"] == pytest.approx(0.5)
    assert metrics["eval/guard/plan_fraction"] == pytest.approx(1.0)
    assert metrics["eval/guard/rejection_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["eval/guard/candidate_acceptance_rate"] == pytest.approx(0.25)
    assert metrics["eval/guard/best_predicted_executable_improvement"] == pytest.approx(
        1.0 / 6.0
    )
    assert metrics[
        "eval/guard/selected_predicted_executable_improvement"
    ] == pytest.approx(0.3)


def test_budget_sweep_propagates_domain_to_planner_and_evaluator(monkeypatch):
    domain = object()
    seen = {"planner": [], "evaluate": []}

    class FakePlanner:
        def __init__(self, *_args, domain=None, **_kwargs):
            seen["planner"].append(domain)

    def fake_evaluate(*_args, domain=None, **_kwargs):
        seen["evaluate"].append(domain)
        return {"eval/world_model_nodes_per_replan": 4.0}

    monkeypatch.setattr(rollout, "GoalPlanner", FakePlanner)
    monkeypatch.setattr(rollout, "evaluate", fake_evaluate)
    import treewm.models.baselines as baselines

    monkeypatch.setattr(
        baselines,
        "tree_config_for",
        lambda _arm, cfg, _model: cfg,
    )
    rollout.sweep_budgets(
        object(),
        object(),
        object(),
        [],
        [4],
        TreeConfig(node_budget=4),
        object(),
        domain=domain,
    )
    assert seen == {"planner": [domain], "evaluate": [domain]}


def test_budget_sweep_counts_first_edge_guard_predictions(monkeypatch):
    class FakePlanner:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(rollout, "GoalPlanner", FakePlanner)
    monkeypatch.setattr(
        rollout,
        "evaluate",
        lambda *_args, **_kwargs: {"eval/world_model_nodes_per_replan": 8.0},
    )
    import treewm.models.baselines as baselines

    monkeypatch.setattr(baselines, "tree_config_for", lambda _arm, cfg, _model: cfg)
    model = type("Model", (), {"cfg": type("Cfg", (), {"branch_factor": 4})()})()
    planner_cfg = type(
        "PlannerCfg", (), {"require_first_edge_improvement": True}
    )()
    rollout.sweep_budgets(
        object(),
        model,
        object(),
        [],
        [4],
        TreeConfig(node_budget=4),
        planner_cfg,
    )


def test_loss_warmup_ramps_redundancy():
    cfg = LossConfig()
    assert cfg.scale("redundancy", 0) == 0.0
    assert 0 < cfg.scale("redundancy", 2500) < 1.0
    assert cfg.scale("redundancy", 10_000) == 1.0
    assert cfg.scale("state", 0) == 1.0, "core losses are not ramped"


def test_synthetic_two_mode_environment_is_representable():
    """state 0 --(-1)--> -1 and --(+1)--> +1.

    The model must be able to represent both modes at once: after fitting, its K
    branches should predict two distinct action chunks with opposite sign and land on
    two distinct successor latents. This is the minimal check that multimodality is not
    averaged away into a single "do nothing" prediction.
    """
    torch.manual_seed(0)
    model = build_model("treewm", SMALL)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    h_max, batch = SMALL.h_max, 64
    z_in = torch.zeros(batch, 1)
    targets = torch.stack(
        [
            torch.full((h_max, 1), -1.0),
            torch.full((h_max, 1), +1.0),
        ]
    )  # [2, h_max, 1]

    for _ in range(400):
        z = model.encode(z_in)
        out = model.branch(z)  # [B, 2, h_max, 1]
        # Assign each of the two targets to its best-matching branch (Hungarian on 2x2).
        cost = ((out.action.unsqueeze(2) - targets.view(1, 1, 2, h_max, 1)) ** 2).mean((-1, -2))
        assign = cost.argmin(dim=1)  # [B, 2] branch per target
        loss = torch.zeros((), dtype=torch.float32)
        for t in range(2):
            picked = out.action[torch.arange(batch), assign[:, t]]
            loss = loss + ((picked - targets[t]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        actions = model.branch(model.encode(torch.zeros(1, 1))).action[0].mean(dim=(-1, -2))
    assert actions.min() < -0.5, f"negative mode not represented: {actions.tolist()}"
    assert actions.max() > 0.5, f"positive mode not represented: {actions.tolist()}"
