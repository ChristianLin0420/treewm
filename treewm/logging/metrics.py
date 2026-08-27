"""DDP-safe metric accumulation.

Every scalar is accumulated locally as a (sum, count) pair and reduced across ranks
at flush time, so a mean over an uneven per-rank batch count is still correct:

    mean = sum_over_ranks(local_sum) / sum_over_ranks(local_count)

This is *not* the same as averaging per-rank means, which is why the reduction is
centralised here rather than done ad hoc at each call site.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

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

    def state_dict(self) -> dict[str, object]:
        """Return the exact local accumulation window for checkpoint/resume.

        A signal can arrive between logging boundaries.  Persisting only model and
        loader state would make the first scalar after resume a partial-window mean,
        which is unacceptable when a preregistered gate consumes a complete cadence.
        """
        return {
            "schema_version": 1,
            "sums": dict(self._sums),
            "counts": dict(self._counts),
            "hists": {
                name: [np.asarray(chunk, dtype=np.float32).copy() for chunk in chunks]
                for name, chunks in self._hists.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore a structurally validated local accumulation window."""
        if not isinstance(state, Mapping) or state.get("schema_version") != 1:
            raise ValueError("metric-tracker checkpoint state is invalid")
        sums = state.get("sums")
        counts = state.get("counts")
        hists = state.get("hists")
        if not isinstance(sums, Mapping) or not isinstance(counts, Mapping):
            raise ValueError("metric-tracker scalar state is invalid")
        if set(sums) != set(counts) or not all(
            isinstance(name, str) and name for name in sums
        ):
            raise ValueError("metric-tracker scalar keys differ")
        restored_sums: dict[str, float] = {}
        restored_counts: dict[str, float] = {}
        for name in sums:
            try:
                total = float(sums[name])
                count = float(counts[name])
            except (TypeError, ValueError) as exc:
                raise ValueError("metric-tracker scalar state is nonnumeric") from exc
            if not np.isfinite(total) or not np.isfinite(count) or count < 0.0:
                raise ValueError("metric-tracker scalar state is non-finite")
            restored_sums[name] = total
            restored_counts[name] = count
        if not isinstance(hists, Mapping):
            raise ValueError("metric-tracker histogram state is invalid")
        restored_hists: dict[str, list[np.ndarray]] = {}
        for name, chunks in hists.items():
            if not isinstance(name, str) or not name or not isinstance(chunks, (list, tuple)):
                raise ValueError("metric-tracker histogram entry is invalid")
            restored_chunks: list[np.ndarray] = []
            for chunk in chunks:
                array = np.asarray(chunk, dtype=np.float32).ravel().copy()
                if not np.isfinite(array).all():
                    raise ValueError("metric-tracker histogram state is non-finite")
                if array.size:
                    restored_chunks.append(array)
            restored_hists[name] = restored_chunks

        self.reset()
        self._sums.update(restored_sums)
        self._counts.update(restored_counts)
        self._hists.update(restored_hists)

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


def pearson_correlation(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    """Linear correlation. Returns 0.0 for degenerate input.

    Reported alongside Spearman because they answer different questions: Pearson says
    whether the head reproduces the *magnitude* of the novelty target, Spearman whether
    it reproduces the *ordering*. Best-first expansion consumes only the ordering, so a
    high Pearson with a low Spearman would still produce bad allocation.
    """
    a = a.detach().float().cpu().numpy() if torch.is_tensor(a) else np.asarray(a, dtype=float)
    b = b.detach().float().cpu().numpy() if torch.is_tensor(b) else np.asarray(b, dtype=float)
    a, b = a.ravel(), b.ravel()
    if a.size < 2 or a.size != b.size:
        return 0.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    out = float(np.corrcoef(a, b)[0, 1])
    return out if np.isfinite(out) else 0.0


def entropy_from_probs(probs: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy in nats of a normalised distribution."""
    p = probs.clamp_min(eps)
    p = p / p.sum(dim=dim, keepdim=True)
    return -(p * p.log()).sum(dim=dim)
