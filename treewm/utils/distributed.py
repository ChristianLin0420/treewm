"""DDP helpers.

The training script must work both under ``torchrun --nproc_per_node=2`` and as a
plain single-process run, so every helper degrades gracefully when
``torch.distributed`` is uninitialised.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(backend: str = "nccl") -> DistInfo:
    """Initialise the process group if launched under torchrun."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DistInfo(rank=0, local_rank=0, world_size=1, distributed=False)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size == 1:
        return DistInfo(rank=0, local_rank=local_rank, world_size=1, distributed=False)

    if not dist.is_initialized():
        if backend == "nccl" and not torch.cuda.is_available():
            backend = "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return DistInfo(rank=rank, local_rank=local_rank, world_size=world_size, distributed=True)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def any_rank_true(value: bool, device: torch.device | None = None) -> bool:
    """Collectively report whether any rank requested an action."""
    if not is_distributed():
        return bool(value)
    tensor = torch.tensor(
        int(bool(value)),
        dtype=torch.int32,
        device=device if device is not None else _default_device(),
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(tensor.item())


def gather_rank_objects(value: Any, destination: int = 0) -> list[Any] | None:
    """Gather a small Python state object on ``destination`` only."""
    if not is_distributed():
        return [value]
    gathered = [None] * dist.get_world_size() if dist.get_rank() == destination else None
    dist.gather_object(value, object_gather_list=gathered, dst=destination)
    return gathered


def all_reduce_mean(value: torch.Tensor | float, device: torch.device | None = None) -> float:
    """Average a scalar across ranks. Returns a python float.

    Metrics must be reduced before logging (spec section 23), and this is the single
    entry point used by :class:`treewm.logging.metrics.MetricTracker`.
    """
    if not is_distributed():
        return float(value.item() if torch.is_tensor(value) else value)
    tensor = value.detach().clone().float() if torch.is_tensor(value) else torch.tensor(float(value))
    tensor = tensor.to(device if device is not None else _default_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / dist.get_world_size())


def all_reduce_sum(value: torch.Tensor | float, device: torch.device | None = None) -> float:
    if not is_distributed():
        return float(value.item() if torch.is_tensor(value) else value)
    tensor = value.detach().clone().float() if torch.is_tensor(value) else torch.tensor(float(value))
    tensor = tensor.to(device if device is not None else _default_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Gather equal-shaped tensors from all ranks and concatenate along dim 0."""
    if not is_distributed():
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module whether or not it is DDP-wrapped."""
    return model.module if hasattr(model, "module") else model
