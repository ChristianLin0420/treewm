"""DDP-safe metric accumulation.

Every scalar is accumulated locally as a (sum, count) pair and reduced across ranks
at flush time, so a mean over an uneven per-rank batch count is still correct:

    mean = sum_over_ranks(local_sum) / sum_over_ranks(local_count)

This is *not* the same as averaging per-rank means, which is why the reduction is
centralised here rather than done ad hoc at each call site.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from treewm.utils.distributed import all_reduce_sum


class MetricTracker:
    """Accumulate scalars and histogram samples, then reduce across ranks."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, float] = defaultdict(float)
        self._hists: dict[str, list[np.ndarray]] = defaultdict(list)

    def add(self, name: str, value: torch.Tensor | float, count: float = 1.0) -> None:
        """Add a scalar observation. ``count`` is the weight (e.g. batch size)."""
        if torch.is_tensor(value):
            if value.numel() == 0:
                return
            value = value.detach().float().mean().item()
        value = float(value)
        if not np.isfinite(value):
            # A NaN in one metric must not silently poison the whole log.
            self._sums[f"{name}__nonfinite"] += 1.0
            self._counts[f"{name}__nonfinite"] += 1.0
            return
        self._sums[name] += value * count
        self._counts[name] += count

    def add_many(self, values: dict[str, torch.Tensor | float], count: float = 1.0) -> None:
        for k, v in values.items():
            self.add(k, v, count)

    def add_hist(self, name: str, values: torch.Tensor | np.ndarray) -> None:
        """Buffer raw samples for a histogram. Histograms are rank-local by design.

        Gathering full histograms across ranks costs bandwidth for no scientific gain --
        the distribution shape on rank 0 is representative, and every scalar summary of
        it (mean/rate) is tracked separately via :meth:`add` and properly reduced.
        """
        if torch.is_tensor(values):
            values = values.detach().float().flatten().cpu().numpy()
        else:
            values = np.asarray(values, dtype=np.float32).ravel()
        values = values[np.isfinite(values)]
        if values.size:
            self._hists[name].append(values)

    def compute(self, reduce: bool = True) -> dict[str, float]:
        """Return reduced means. Keys with zero count are dropped."""
        out: dict[str, float] = {}
        for name in sorted(self._sums):
            total = self._sums[name]
            count = self._counts[name]
            if reduce:
                total = all_reduce_sum(total, self.device)
                count = all_reduce_sum(count, self.device)
            if count > 0:
                out[name] = total / count
        return out

    def histograms(self) -> dict[str, np.ndarray]:
        return {name: np.concatenate(chunks) for name, chunks in self._hists.items() if chunks}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()
        self._hists.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._sums


def rank_correlation(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    """Spearman rank correlation. Returns 0.0 for degenerate input.

    Used for ``expansion/gain_rank_correlation``: what matters for best-first expansion
    is whether the predicted gain *orders* frontier nodes correctly, not its scale.
    """
    from scipy.stats import spearmanr

    # .float() first: numpy has no bfloat16, and these tensors arrive from autocast.
    a = a.detach().float().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
    b = b.detach().float().cpu().numpy() if torch.is_tensor(b) else np.asarray(b)
    a, b = a.ravel(), b.ravel()
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return 0.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    rho, _ = spearmanr(a, b)
    return float(rho) if np.isfinite(rho) else 0.0


def entropy_from_probs(probs: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy in nats of a normalised distribution."""
    p = probs.clamp_min(eps)
    p = p / p.sum(dim=dim, keepdim=True)
    return -(p * p.log()).sum(dim=dim)
