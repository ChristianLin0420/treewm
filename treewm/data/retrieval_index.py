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
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute, for every dataset index, the quantised cell of its own continuation.

    Returns ``(cells [N], valid [N])``. Vectorised: this runs once at startup over the
    full dataset, so it must not be a python loop.
    """
    n = len(obs_norm)
    remaining = index.steps_remaining
    valid = remaining >= horizon
    target_idx = np.where(valid, np.arange(n) + horizon, np.arange(n))
    cells = quantizer.cell_ids(obs_norm[target_idx])
    return cells.astype(np.int64), valid.astype(np.float32)


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
    ) -> None:
        self.cfg = cfg
        self.device = device
        rng = np.random.default_rng(seed)
        n = len(obs_norm)
        num = min(cfg.num_keys, n)
        self.key_idx = np.sort(rng.choice(n, size=num, replace=False))
        self.key_obs = torch.from_numpy(obs_norm[self.key_idx]).to(device)
        self.key_cells = torch.from_numpy(endpoint_cells[self.key_idx]).to(device)
        self.key_valid = torch.from_numpy(endpoint_valid[self.key_idx]).to(device)
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
