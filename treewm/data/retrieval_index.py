"""Latent-space retrieval over dataset states, and the data-grounded gain target.

The expansion-gain head must be supervised by something that exists on step 1. Using
"expand deeper with the current model and measure coverage in the model's own q-space"
would bootstrap a self-graded signal off noise, so instead the target is:

    G(n | T) = | { endpoint_cell[c] : c in dataset_neighbours(z_n) } \\ cells(T) |

i.e. *how many new regions of the world become reachable if this node is expanded*,
where reachability comes from the offline data and regions come from the global
quantiser. Both are model-independent; only the neighbour lookup uses the encoder,
and it is refreshed periodically rather than trained through.

The index is brute-force on GPU (chunked ``cdist`` + ``topk``). A KD-tree would be
slower here: tree search degrades to linear scan above ~20 dimensions and z_dim is 128.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from treewm.evaluation.coverage import StateQuantizer


@dataclass
class RetrievalConfig:
    num_keys: int = 100_000
    num_neighbors: int = 16
    refresh_every: int = 2000
    query_chunk: int = 4096
    key_chunk: int = 50_000
    enabled: bool = True


def compute_endpoint_cells(
    obs_norm: np.ndarray,
    index,
    quantizer: StateQuantizer,
    horizon: int,
    key_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute, for every dataset index, the quantised cell of its own continuation.

    Returns ``(cells [N], valid [N])``. Vectorised: this runs once at startup over the
    full dataset, so it must not be a python loop.
    """
    # On 100M datasets, materialising `steps_remaining`, `arange`, advanced-indexed
    # observations and endpoint arrays for every transition costs tens of GB even
    # though LatentIndex retains only cfg.num_keys. Address only the deterministic key
    # subset when supplied.
    source_idx = (
        np.arange(len(obs_norm), dtype=np.int64)
        if key_idx is None
        else np.asarray(key_idx, dtype=np.int64)
    )
    # Index *before* converting so a memmapped 100M `steps_remaining` array only
    # materialises the bounded key subset.
    valid = np.asarray(index.steps_remaining[source_idx]) >= horizon
    target_idx = np.where(valid, source_idx + horizon, source_idx)
    cells = quantizer.cell_ids(obs_norm[target_idx])
    return cells.astype(np.int64), valid.astype(np.float32)


def sample_key_indices(num_observations: int, cfg: RetrievalConfig, seed: int) -> np.ndarray:
    """Deterministic, bounded retrieval-key subset without a full-size permutation."""
    rng = np.random.default_rng(seed)
    num = min(cfg.num_keys, int(num_observations))
    if num == int(num_observations):
        return np.arange(num, dtype=np.int64)
    if int(num_observations) <= 10_000_000:
        # Preserve the original sampling stream for standard datasets.
        return np.sort(rng.choice(int(num_observations), size=num, replace=False))
    # Generator.choice(replace=False) may allocate work proportional to population on
    # some NumPy versions. Rejection sampling is O(num_keys) memory and has negligible
    # collisions for 100k keys from 100M transitions.
    selected = np.empty(0, dtype=np.int64)
    while len(selected) < num:
        needed = num - len(selected)
        draws = rng.integers(0, int(num_observations), size=max(needed * 2, 1024), dtype=np.int64)
        selected = np.unique(np.concatenate((selected, draws)))
    if len(selected) > num:
        selected = rng.choice(selected, size=num, replace=False)
    return np.sort(selected)


class LatentIndex:
    """Nearest dataset states in encoder-latent space, refreshed on a schedule."""

    def __init__(
        self,
        obs_norm: np.ndarray,
        endpoint_cells: np.ndarray,
        endpoint_valid: np.ndarray,
        cfg: RetrievalConfig,
        device: torch.device,
        seed: int = 0,
        key_idx: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device
        n = len(obs_norm)
        self.key_idx = (
            sample_key_indices(n, cfg, seed)
            if key_idx is None
            else np.asarray(key_idx, dtype=np.int64)
        )
        self.key_obs = torch.from_numpy(obs_norm[self.key_idx]).to(device)
        if len(endpoint_cells) == n:
            endpoint_cells = endpoint_cells[self.key_idx]
            endpoint_valid = endpoint_valid[self.key_idx]
        elif len(endpoint_cells) != len(self.key_idx):
            raise ValueError("endpoint arrays must cover either all observations or selected keys")
        self.key_cells = torch.from_numpy(np.asarray(endpoint_cells)).to(device)
        self.key_valid = torch.from_numpy(np.asarray(endpoint_valid)).to(device)
        self.keys: torch.Tensor | None = None
        self.last_refresh = -1

    @torch.no_grad()
    def refresh(self, encoder, step: int, force: bool = False) -> bool:
        """Re-encode the key states. Returns True if a refresh happened."""
        if not self.cfg.enabled:
            return False
        if not force and self.keys is not None and (step - self.last_refresh) < self.cfg.refresh_every:
            return False
        was_training = encoder.training
        encoder.eval()
        chunks = []
        for i in range(0, len(self.key_obs), self.cfg.key_chunk):
            chunks.append(encoder(self.key_obs[i : i + self.cfg.key_chunk]).float())
        encoder.train(was_training)
        self.keys = torch.cat(chunks, dim=0)
        self.last_refresh = step
        return True

    def state_dict(self) -> dict[str, Any]:
        """Persist the moving encoder snapshot used by the gain target.

        Recomputing the index with the resumed *current* encoder would differ from an
        uninterrupted run whenever the saved index was intentionally stale between
        refreshes.  The keys are therefore part of the exact training state.
        """
        return {
            "last_refresh": int(self.last_refresh),
            "keys": self.keys.detach().cpu() if self.keys is not None else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        keys = state.get("keys")
        if keys is not None:
            if keys.ndim != 2 or keys.shape[0] != len(self.key_idx):
                raise ValueError(
                    f"latent-index checkpoint shape {tuple(keys.shape)} is incompatible "
                    f"with {len(self.key_idx)} configured keys"
                )
            self.keys = keys.to(self.device)
        else:
            self.keys = None
        self.last_refresh = int(state.get("last_refresh", -1))

    @torch.no_grad()
    def query_cells(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """For query latents ``[B, D]`` return neighbour endpoint cells ``[B, k]``.

        Also returns a validity mask of the same shape.
        """
        assert self.keys is not None, "LatentIndex.refresh() must be called before query"
        k = min(self.cfg.num_neighbors, len(self.keys))
        out_cells, out_valid = [], []
        z = z.float()
        for i in range(0, len(z), self.cfg.query_chunk):
            q = z[i : i + self.cfg.query_chunk]
            best_d = None
            best_i = None
            for j in range(0, len(self.keys), self.cfg.key_chunk):
                d = torch.cdist(q, self.keys[j : j + self.cfg.key_chunk])
                dv, di = torch.topk(d, k=min(k, d.shape[1]), dim=1, largest=False)
                di = di + j
                if best_d is None:
                    best_d, best_i = dv, di
                else:
                    cat_d = torch.cat([best_d, dv], dim=1)
                    cat_i = torch.cat([best_i, di], dim=1)
                    sel = torch.topk(cat_d, k=k, dim=1, largest=False).indices
                    best_d = torch.gather(cat_d, 1, sel)
                    best_i = torch.gather(cat_i, 1, sel)
            out_cells.append(self.key_cells[best_i])
            out_valid.append(self.key_valid[best_i])
        return torch.cat(out_cells, 0), torch.cat(out_valid, 0)


@torch.no_grad()
def gain_targets(
    index: LatentIndex,
    node_z: torch.Tensor,
    covered_cells: torch.Tensor,
    covered_valid: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Data-grounded expansion-gain targets.

    Args:
        node_z: ``[B, N, D]`` latents of candidate (frontier) nodes.
        covered_cells: ``[B, C]`` cells already covered by the tree.
        covered_valid: ``[B, C]`` mask.

    Returns:
        ``[B, N]`` marginal new-cell counts, optionally divided by the neighbour count
        so the target lives in ``[0, 1]`` and the head's scale is stable across configs.
    """
    from treewm.evaluation.coverage import batched_new_cell_counts

    b, n, d = node_z.shape
    cells, valid = index.query_cells(node_z.reshape(b * n, d))
    cells = cells.view(b, n, -1)
    valid = valid.view(b, n, -1)
    counts = batched_new_cell_counts(cells, valid, covered_cells, covered_valid)
    if normalize:
        counts = counts / max(cells.shape[-1], 1)
    return counts
