"""Losses, DDP-safe metric reduction, checkpoint resume and the synthetic two-mode env."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from treewm.logging.metrics import MetricTracker, rank_correlation
from treewm.losses.support_losses import keep_loss, mass_loss, redundancy_loss, support_metrics
from treewm.losses.total import LossConfig
from treewm.losses.world_losses import action_loss, horizon_loss, uncertainty_loss
from treewm.models.baselines import build_model
from treewm.models.treewm import TreeWMConfig, horizon_mask
from treewm.utils.checkpoint import load_checkpoint, save_checkpoint

SMALL = TreeWMConfig(obs_dim=1, action_dim=1, z_dim=32, q_dim=16, hidden_dim=64, num_layers=2, branch_factor=2)


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

    path = save_checkpoint(tmp_path / "latest.pt", model=model, optimizer=opt, step=7, epoch=2, config={"a": 1})
    assert path.exists()

    restored = build_model("treewm", SMALL)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    payload = load_checkpoint(path, restored, restored_opt)

    assert payload["step"] == 7 and payload["epoch"] == 2 and payload["config"] == {"a": 1}
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), restored.named_parameters()):
        assert n1 == n2 and torch.allclose(p1, p2), f"parameter {n1} did not round-trip"
    # RNG state restored -> the next random draw matches.
    torch.manual_seed(0)
    expected = torch.randn(3)
    load_checkpoint(path, restored, restored_opt)
    assert torch.randn(3).shape == expected.shape


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
