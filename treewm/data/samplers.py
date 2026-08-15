"""DataLoader construction with DDP-aware sampling.

Training is step-based rather than epoch-based, so :class:`InfiniteLoader` wraps a
DataLoader and calls ``set_epoch`` on each pass. Without that call a DistributedSampler
reshuffles identically every epoch and each rank sees the same order forever.
"""

from __future__ import annotations

from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from treewm.utils.distributed import is_distributed


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


class InfiniteLoader:
    """Yield batches forever, advancing the sampler epoch on every wrap."""

    def __init__(self, loader: DataLoader, sampler: DistributedSampler | None = None, start_epoch: int = 0) -> None:
        self.loader = loader
        self.sampler = sampler
        self.epoch = start_epoch
        self._iter: Iterator | None = None

    def __iter__(self) -> "InfiniteLoader":
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._iter is None:
            self._new_epoch()
        try:
            return next(self._iter)
        except StopIteration:
            self.epoch += 1
            self._new_epoch()
            return next(self._iter)

    def _new_epoch(self) -> None:
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch)
        self._iter = iter(self.loader)

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self._iter = None


def to_device(batch: dict[str, torch.Tensor], device: torch.device, non_blocking: bool = True) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v for k, v in batch.items()}
