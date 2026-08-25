import math

import pytest
import torch

from treewm.losses.world_losses import prediction_metrics


def test_horizon_target_histogram_exposes_collapsed_and_balanced_targets():
    pred_z = torch.zeros(1, 5, 2)
    target_z = torch.zeros_like(pred_z)
    pred_action = torch.zeros(1, 5, 4, 1)
    target_action = torch.zeros_like(pred_action)
    mask = torch.ones(1, 5, 4)
    matched = torch.ones(1, 5)
    horizons = torch.tensor([4, 8, 16, 32, 64])
    logits = torch.eye(5).unsqueeze(0) * 10

    balanced = prediction_metrics(
        pred_z,
        target_z,
        pred_action,
        target_action,
        mask,
        logits,
        torch.arange(5).unsqueeze(0),
        matched,
        horizons,
    )
    assert balanced["data/horizon_target_normalized_entropy"] == pytest.approx(1.0)
    assert balanced["data/horizon_target_occupied_classes"] == 5.0
    assert sum(
        balanced[f"data/horizon_target_fraction_h{int(h)}"] for h in horizons
    ) == pytest.approx(1.0)

    collapsed = prediction_metrics(
        pred_z,
        target_z,
        pred_action,
        target_action,
        mask,
        logits,
        torch.full((1, 5), 4),
        matched,
        horizons,
    )
    assert math.isfinite(collapsed["data/horizon_target_entropy"])
    assert collapsed["data/horizon_target_normalized_entropy"] == 0.0
    assert collapsed["data/horizon_target_occupied_classes"] == 1.0
