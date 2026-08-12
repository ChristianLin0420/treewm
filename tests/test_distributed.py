"""DDP-safe metric reduction, exercised with a real 2-process gloo group."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from treewm.logging.metrics import MetricTracker
from treewm.utils.distributed import all_reduce_mean, all_reduce_sum


def _worker(rank: int, world_size: int, queue) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29517",
        RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Uneven per-rank counts: rank 0 sees 1 sample of value 10, rank 1 sees 3 of value 2.
    tracker = MetricTracker()
    if rank == 0:
        tracker.add("loss", 10.0, count=1)
    else:
        tracker.add("loss", 2.0, count=3)
    reduced = tracker.compute(reduce=True)

    results = {
        "loss": reduced["loss"],
        "mean": all_reduce_mean(float(rank)),
        "sum": all_reduce_sum(float(rank + 1)),
    }
    if rank == 0:
        queue.put(results)
    dist.destroy_process_group()


def test_metric_reduction_is_sample_weighted_across_ranks():
    """The reduced mean must weight by sample count, not average per-rank means.

    Per-rank means would give (10 + 2) / 2 = 6.0. The correct sample-weighted answer is
    (10*1 + 2*3) / 4 = 4.0. Getting this wrong silently biases every logged metric
    whenever ranks see different batch counts.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, queue)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=90)

    assert all(p.exitcode == 0 for p in procs), f"worker failed: {[p.exitcode for p in procs]}"
    results = queue.get(timeout=10)
    assert abs(results["loss"] - 4.0) < 1e-5, f"expected sample-weighted 4.0, got {results['loss']}"
    assert abs(results["mean"] - 0.5) < 1e-5  # mean of ranks {0, 1}
    assert abs(results["sum"] - 3.0) < 1e-5  # sum of {1, 2}


def test_reduction_is_identity_without_a_process_group():
    assert all_reduce_mean(3.5) == 3.5
    assert all_reduce_sum(torch.tensor(2.0)) == 2.0
    tracker = MetricTracker()
    tracker.add("x", 1.0, count=2)
    tracker.add("x", 3.0, count=2)
    assert abs(tracker.compute(reduce=True)["x"] - 2.0) < 1e-6
