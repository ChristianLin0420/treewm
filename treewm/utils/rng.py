"""Isolated random streams.

Logging must be *observational*: rendering a diagnostic tree must not change what the
model trains on or what the planner does. Previously `random_score` fell back to the
global torch stream whenever no generator was supplied, and both the planner and the
visualisation path supplied none -- so changing the visualisation cadence perturbed
training and planning. The same config measured 0.600 and 0.507 depending only on how
often it was visualised.

Four streams, derived deterministically from one run seed:

    train    data sampling, dropout, loss-side subsampling   (the global torch stream)
    planner  frontier randomness during planning
    eval     evaluation-time frontier randomness
    viz      diagnostic/visualisation sampling

`viz` and `eval` are separate so that adding a diagnostic cannot move an evaluation
number, and `planner` and `eval` are separate so that a planner change cannot silently
reshuffle which episodes are run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# Offsets keep the four streams far apart in seed space.
_OFFSETS = {"train": 0, "planner": 10_007, "eval": 20_011, "viz": 30_013}


def make_generator(seed: int, stream: str, device: torch.device | str = "cpu") -> torch.Generator:
    """A dedicated generator for ``stream``, reproducible from ``seed`` alone."""
    if stream not in _OFFSETS:
        raise ValueError(f"unknown rng stream {stream!r}; options: {sorted(_OFFSETS)}")
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) * 1_000_003 + _OFFSETS[stream])
    return gen


@dataclass
class RngStreams:
    """The four isolated streams for one run."""

    seed: int
    device: torch.device

    def __post_init__(self) -> None:
        self.planner = make_generator(self.seed, "planner", self.device)
        self.eval = make_generator(self.seed, "eval", self.device)
        self.viz = make_generator(self.seed, "viz", self.device)

    def reset(self, stream: str) -> torch.Generator:
        """Re-seed one stream to its initial state.

        Evaluation is reset before each sweep so that a run's Nth evaluation does not
        depend on how many diagnostics happened to precede it.
        """
        gen = make_generator(self.seed, stream, self.device)
        setattr(self, stream, gen)
        return gen

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return portable states for every explicit non-global stream."""
        return {
            name: getattr(self, name).get_state().detach().cpu().clone()
            for name in ("planner", "eval", "viz")
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore streams without changing their configured device."""
        for name in ("planner", "eval", "viz"):
            value = state.get(name)
            if value is not None:
                getattr(self, name).set_state(value.detach().cpu())
