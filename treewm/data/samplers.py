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
