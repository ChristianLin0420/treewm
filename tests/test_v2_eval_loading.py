"""Standalone checkpoint loading must reconstruct the v2 scorer architecture."""

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.eval import load_run
from treewm.models.baselines import build_model
from treewm.utils import config as cfg_utils


def _config():
    return OmegaConf.create(
        {
            "arm": "treewm",
            "env": {"obs_dim": 2, "action_dim": 1},
            "future_sets": {"h_max": 4, "horizons": [1, 2, 4]},
            "model": {
                "z_dim": 16,
                "q_dim": 8,
                "hidden_dim": 32,
                "encoder_hidden": 32,
                "num_layers": 1,
                "num_heads": 4,
                "branch_factor": 2,
                "h_max": 4,
                "horizons": [1, 2, 4],
                "scales": [["mixed", 2, 1.0]],
                "max_depth": 4,
                "dropout": 0.0,
                "reconstruction": True,
                "residual_dynamics": True,
                "normalize_q": True,
                "use_tree_context": True,
                "use_depth_embedding": False,
                "novelty_space": "q",
                "horizon_mode": "learned",
                "fixed_horizon_index": 0,
                "flatk_max": 8,
            },
            "losses": {"gain_set_context": True},
        }
    )


def test_load_run_constructs_set_aware_gain_before_state_restore(tmp_path):
    cfg = _config()
    model = build_model("treewm", cfg_utils.model_config(cfg), k_max=8)
    model.gain_head.set_set_aware(True)
    checkpoint = tmp_path / "v2.pt"
    torch.save(
        {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "model": model.state_dict(),
            "normalizer": {
                "obs_mean": np.zeros(2, dtype=np.float32),
                "obs_std": np.ones(2, dtype=np.float32),
                "act_mean": np.zeros(1, dtype=np.float32),
                "act_std": np.ones(1, dtype=np.float32),
            },
        },
        checkpoint,
    )

    restored, _, _, _ = load_run(str(checkpoint), torch.device("cpu"))
    assert restored.gain_head.set_aware_enabled is True
    assert restored.gain_head.cross_attention is not None
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
