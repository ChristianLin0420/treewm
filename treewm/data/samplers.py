"""DataLoader construction with DDP-aware sampling.

Training is step-based rather than epoch-based, so :class:`InfiniteLoader` wraps a
DataLoader and calls ``set_epoch`` on each pass. Without that call a DistributedSampler
reshuffles identically every epoch and each rank sees the same order forever.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from treewm.utils.distributed import is_distributed


class FixedRepresentativeSampler(Sampler[int]):
    """A fixed, distributed validation sample whose every batch spans the dataset.

    ``ChunkDataset.anchors`` is sorted.  Consequently, evaluating only the first few
    batches of a sequential loader measures the lowest anchor ranks rather than a
    representative validation slice.  This sampler selects one item from each of
    equally sized rank strata, then interleaves those strata so *every global batch*
    covers the full selected-anchor range.

    The complete order is constructed once from a private generator.  Iterating the
    sampler again never reshuffles it, and distributed ranks receive disjoint strided
    views with exactly ``batch_size * num_batches`` examples each.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        num_batches: int,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_size = int(len(dataset))
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.batch_size <= 0 or self.num_batches <= 0:
            raise ValueError("fixed validation batch_size and num_batches must be positive")
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("invalid fixed validation distributed rank")

        global_batch_size = self.batch_size * self.num_replicas
        global_sample_size = global_batch_size * self.num_batches
        if global_sample_size > self.dataset_size:
            raise ValueError(
                f"fixed validation sample needs {global_sample_size} examples but "
                f"dataset has {self.dataset_size}"
            )

        # Select one rank from every equal-width stratum without touching the global
        # torch RNG. Since sample_size <= population, every stratum is non-empty and
        # the resulting dataset positions are unique.
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        stratum = torch.arange(global_sample_size, dtype=torch.int64)
        starts = torch.div(
            stratum * self.dataset_size, global_sample_size, rounding_mode="floor"
        )
        ends = torch.div(
            (stratum + 1) * self.dataset_size,
            global_sample_size,
            rounding_mode="floor",
        )
        widths = ends - starts
        offsets = torch.floor(
            torch.rand(global_sample_size, generator=generator) * widths.float()
        ).to(torch.int64)
        selected = starts + offsets

        # A sorted stratified sample would recreate the original prefix bias. Arrange
        # consecutive narrow strata down columns so each row/global batch spans the
        # population, then seed-permute each row. The strided rank split below preserves
        # local batch boundaries in DDP.
        by_batch = selected.view(global_batch_size, self.num_batches).transpose(0, 1)
        ordered_batches = []
        for values in by_batch:
            permutation = torch.randperm(global_batch_size, generator=generator)
            ordered_batches.append(values[permutation])
        self.global_indices = torch.stack(ordered_batches).reshape(-1)
        self.local_indices = self.global_indices[self.rank :: self.num_replicas].clone()

    def __iter__(self):
        return iter(self.local_indices.tolist())

    def __len__(self) -> int:
        return int(self.local_indices.numel())

    def summary(self) -> dict[str, Any]:
        """JSON-safe provenance and anchor-rank coverage for the fixed sample."""
        positions = self.global_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        fractions = positions.astype(np.float64) / max(self.dataset_size - 1, 1)
        quantiles = np.quantile(fractions, [0.0, 0.25, 0.5, 0.75, 1.0])
        return {
            "sampler": "fixed_representative_stratified_permutation_v1",
            "seed": self.seed,
            "dataset_size": self.dataset_size,
            "global_sample_size": int(len(positions)),
            "batch_size_per_rank": self.batch_size,
            "num_batches": self.num_batches,
            "num_replicas": self.num_replicas,
            "indices_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
            "anchor_rank_fraction_quantiles": {
                name: float(value)
                for name, value in zip(
                    ("q00", "q25", "q50", "q75", "q100"), quantiles, strict=True
                )
            },
        }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 8,
    seed: int = 0,
    drop_last: bool = True,
    pin_memory: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Build a DataLoader on an explicit generator.

    Without one, PyTorch draws each iterator's worker ``base_seed`` from the *global*
    torch stream. Any code that re-creates an iterator -- e.g. ``next(iter(val_loader))``
    inside a diagnostic -- therefore advances the stream training samples from, which is
    how visualisation cadence was changing training results.
    """
    sampler: DistributedSampler | None = None
    if is_distributed():
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed, drop_last=drop_last)
    elif shuffle:
        # Use the same epoch-addressable sampler for single-GPU training.  PyTorch's
        # RandomSampler materialises its permutation from the loader generator once and
        # exposes no cursor, so a checkpoint in the middle of an epoch cannot recreate
        # the remaining order.  A one-replica DistributedSampler has identical shuffled
        # without-replacement semantics and lets InfiniteLoader replay an exact epoch.
        sampler = DistributedSampler(
            dataset,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=seed,
            drop_last=drop_last,
        )
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        generator=generator,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return loader, sampler


def build_fixed_validation_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_batches: int,
    num_workers: int = 2,
    seed: int = 0,
    pin_memory: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[DataLoader, FixedRepresentativeSampler]:
    """Build the stable, representative loader used by validation and diagnostics.

    ``num_batches`` is per rank, matching the historical validation-loop cap.  If the
    dataset is smaller than the requested distributed sample, the largest whole number
    of global batches is used.  No padding or duplication is allowed in measurement.
    """
    if batch_size <= 0 or num_batches <= 0:
        raise ValueError("validation batch_size and num_batches must be positive")
    if is_distributed():
        num_replicas = dist.get_world_size()
        rank = dist.get_rank()
    else:
        num_replicas = 1
        rank = 0
    available_batches = len(dataset) // (int(batch_size) * num_replicas)
    actual_batches = min(int(num_batches), int(available_batches))
    if actual_batches <= 0:
        raise ValueError(
            "validation dataset must contain at least one complete batch per rank"
        )
    sampler = FixedRepresentativeSampler(
        dataset,
        batch_size=int(batch_size),
        num_batches=actual_batches,
        seed=int(seed),
        num_replicas=num_replicas,
        rank=rank,
    )
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        generator=generator,
        batch_size=int(batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=int(num_workers),
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return loader, sampler


class InfiniteLoader:
    """Yield batches forever, advancing the sampler epoch on every wrap."""

    def __init__(self, loader: DataLoader, sampler: DistributedSampler | None = None, start_epoch: int = 0) -> None:
        self.loader = loader
        self.sampler = sampler
        self.epoch = start_epoch
        self._iter: Iterator | None = None
        self.batches_yielded_in_epoch = 0
        self._epoch_generator_state: torch.Tensor | None = None
        self._resume_batches = 0
        self._resume_generator_state: torch.Tensor | None = None

    def __iter__(self) -> "InfiniteLoader":
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._iter is None:
            self._new_epoch()
        try:
            batch = next(self._iter)
        except StopIteration:
            self.epoch += 1
            self.batches_yielded_in_epoch = 0
            self._new_epoch()
            batch = next(self._iter)
        self.batches_yielded_in_epoch += 1
        return batch

    def _new_epoch(self) -> None:
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch)
        generator = self.loader.generator
        if generator is not None:
            if self._resume_generator_state is not None:
                generator.set_state(self._resume_generator_state)
                self._resume_generator_state = None
            self._epoch_generator_state = generator.get_state().clone()
        self._iter = iter(self.loader)
        if self._resume_batches:
            to_skip = self._resume_batches
            self._resume_batches = 0
            for _ in range(to_skip):
                try:
                    next(self._iter)
                except StopIteration as exc:
                    raise ValueError(
                        f"loader checkpoint offset {to_skip} exceeds epoch length"
                    ) from exc
            self.batches_yielded_in_epoch = to_skip

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "batches_yielded_in_epoch": self.batches_yielded_in_epoch,
            "epoch_generator_state": (
                self._epoch_generator_state.clone()
                if self._epoch_generator_state is not None
                else None
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.batches_yielded_in_epoch = 0
        self._resume_batches = int(state.get("batches_yielded_in_epoch", 0))
        generator_state = state.get("epoch_generator_state")
        self._resume_generator_state = (
            generator_state.detach().cpu().clone() if torch.is_tensor(generator_state) else None
        )
        self._iter = None


def to_device(batch: dict[str, torch.Tensor], device: torch.device, non_blocking: bool = True) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v for k, v in batch.items()}
