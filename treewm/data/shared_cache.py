"""Build each dataset's preprocessed arrays exactly once, share them across all jobs.

Sixteen concurrent single-GPU jobs would otherwise each hold a private copy of the
observation/action arrays and recompute identical normalisation statistics. For
humanoidmaze and antsoccer that is several GB per process; the arrays are also identical
by construction, so the copies buy nothing.

Here they are written once to ``.npy`` and opened with ``mmap_mode='r'``. Every process
then reads the *same* physical pages through the OS page cache, so N jobs cost roughly
one copy of RAM rather than N.

Three properties this has to get right:

* **exactly once** -- construction is guarded by an exclusive lockfile, so a second job
  starting concurrently waits rather than racing to write the same files.
* **deterministic keys** -- the key hashes dataset name, array shapes/dtypes, the
  normalisation recipe and the cache format version. A config change produces a new key
  instead of silently reusing arrays built under different settings.
* **provably consumed** -- a previous cache bug in this project was read but not actually
  used by the dataloader, so the run looked cached while recomputing everything. The
  loader therefore asserts the arrays it ends up holding are genuinely ``np.memmap``
  instances backed by the cache path, and exports that as a metric rather than a promise.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CACHE_VERSION = 3
DEFAULT_ROOT = Path(os.environ.get("TREEWM_CACHE", Path.home() / ".cache" / "treewm"))
LOCK_STALE_S = 3600.0

# module-level counters, surfaced by the trainer as cache/{hit,miss} metrics
STATS = {"hit": 0, "miss": 0, "wait_s": 0.0}


def cache_key(dataset_name: str, recipe: dict[str, Any]) -> str:
    payload = json.dumps({"v": CACHE_VERSION, "dataset": dataset_name, **recipe},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class _Lock:
    """Exclusive lockfile with stale-holder recovery.

    ``O_CREAT | O_EXCL`` is atomic on POSIX, so exactly one process wins. A crashed
    holder would otherwise block every job forever, so a lock older than
    ``LOCK_STALE_S`` is broken.
    """

    def __init__(self, path: Path, timeout: float = 7200.0) -> None:
        self.path = path
        self.timeout = timeout
        self.fd: int | None = None

    def __enter__(self) -> "_Lock":
        start = time.time()
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, f"{os.getpid()} {time.time()}\n".encode())
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > LOCK_STALE_S:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.time() - start > self.timeout:
                    raise TimeoutError(f"waited {self.timeout}s for cache lock {self.path}")
                STATS["wait_s"] += 2.0
                time.sleep(2.0)

    def __exit__(self, *exc) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


@dataclass
class SharedSplit:
    """Memory-mapped arrays for one split."""

    obs: np.ndarray
    act: np.ndarray
    terminals: np.ndarray
    obs_norm: np.ndarray
    act_norm: np.ndarray
    path: Path

    def assert_memmapped(self) -> None:
        for name in ("obs", "act", "obs_norm", "act_norm", "terminals"):
            arr = getattr(self, name)
            assert isinstance(arr, np.memmap), (
                f"{name} is {type(arr).__name__}, not np.memmap -- the shared cache was "
                "loaded but is not actually backing the dataloader. This exact failure "
                "(cache read but unused) has happened before; refusing to continue."
            )
            assert Path(arr.filename).parent == self.path, (
                f"{name} is memmapped from {arr.filename}, not the expected cache dir {self.path}"
            )


@dataclass
class SharedCache:
    key: str
    path: Path
    train: SharedSplit
    val: SharedSplit
    norm_stats: dict[str, np.ndarray]
    was_hit: bool

    def assert_consumed_by(self, train_ds, val_ds) -> dict[str, float]:
        """Prove the datasets really hold the mapped arrays, then report it as metrics."""
        self.train.assert_memmapped()
        self.val.assert_memmapped()
        for ds, split in ((train_ds, self.train), (val_ds, self.val)):
            for attr, ref in (("obs", split.obs), ("obs_norm", split.obs_norm),
                              ("act_norm", split.act_norm)):
                got = getattr(ds, attr)
                assert isinstance(got, np.memmap), (
                    f"dataset.{attr} is {type(got).__name__}: the ChunkDataset copied the "
                    "cache instead of mapping it, so RAM scales with job count again."
                )
                assert got.base is ref or np.shares_memory(got, ref), (
                    f"dataset.{attr} does not share memory with the cache array"
                )
        return {"cache/hit": float(self.was_hit), "cache/miss": float(not self.was_hit),
                "cache/consumed": 1.0, "cache/wait_s": float(STATS["wait_s"])}


def _write_split(dst: Path, name: str, arrays: dict[str, np.ndarray]) -> None:
    for k, v in arrays.items():
        tmp = dst / f".{name}_{k}.tmp.npy"
        np.save(tmp, v)
        tmp.replace(dst / f"{name}_{k}.npy")


def _load_split(dst: Path, name: str) -> SharedSplit:
    m = lambda k: np.load(dst / f"{name}_{k}.npy", mmap_mode="r")
    return SharedSplit(obs=m("obs"), act=m("act"), terminals=m("terminals"),
                       obs_norm=m("obs_norm"), act_norm=m("act_norm"), path=dst)


def build_or_load(
    dataset_name: str,
    dataset_dir: str | None = None,
    root: Path | None = None,
    eps: float = 1e-6,
    verbose: bool = True,
) -> SharedCache:
    """Return memory-mapped arrays for ``dataset_name``, building them if absent."""
    from treewm.data.ogbench_dataset import Normalizer, load_ogbench

    recipe = {"norm": "mean_std", "eps": eps, "dtype": "float32"}
    key = cache_key(dataset_name, recipe)
    root = Path(root or DEFAULT_ROOT)
    dst = root / f"{dataset_name}__{key}"
    manifest = dst / "manifest.json"

    if manifest.exists():
        STATS["hit"] += 1
        if verbose:
            print(f"[cache] HIT  {dataset_name} key={key} path={dst}", flush=True)
        stats = json.loads(manifest.read_text())["norm_stats"]
        return SharedCache(key, dst, _load_split(dst, "train"), _load_split(dst, "val"),
                           {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()},
                           was_hit=True)

    dst.mkdir(parents=True, exist_ok=True)
    with _Lock(root / f"{dataset_name}__{key}.lock"):
        # Another job may have finished while we queued for the lock.
        if manifest.exists():
            STATS["hit"] += 1
            if verbose:
                print(f"[cache] HIT (after wait) {dataset_name} key={key}", flush=True)
            stats = json.loads(manifest.read_text())["norm_stats"]
            return SharedCache(key, dst, _load_split(dst, "train"), _load_split(dst, "val"),
                               {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()},
                               was_hit=True)

        STATS["miss"] += 1
        if verbose:
            print(f"[cache] MISS {dataset_name} key={key} -- building at {dst}", flush=True)
        t0 = time.time()
        _, train, val = load_ogbench(dataset_name, dataset_dir=dataset_dir)
        normalizer = Normalizer.fit(train["observations"], train["actions"])

        for split_name, split in (("train", train), ("val", val)):
            obs = np.ascontiguousarray(split["observations"], dtype=np.float32)
            act = np.ascontiguousarray(split["actions"], dtype=np.float32)
            _write_split(dst, split_name, {
                "obs": obs, "act": act,
                "terminals": np.ascontiguousarray(split["terminals"]),
                "obs_norm": np.ascontiguousarray(normalizer.norm_obs(obs), dtype=np.float32),
                "act_norm": np.ascontiguousarray(normalizer.norm_act(act), dtype=np.float32),
            })

        stats = {k: np.asarray(v).tolist() for k, v in normalizer.state_dict().items()}
        tmp = dst / ".manifest.tmp"
        tmp.write_text(json.dumps({
            "version": CACHE_VERSION, "dataset": dataset_name, "key": key, "recipe": recipe,
            "built_s": round(time.time() - t0, 1), "norm_stats": stats,
            "shapes": {"train_obs": list(np.shape(train["observations"])),
                       "val_obs": list(np.shape(val["observations"]))},
        }, indent=2))
        tmp.replace(manifest)  # atomic: manifest presence is the "ready" flag
        if verbose:
            print(f"[cache] built {dataset_name} in {time.time() - t0:.0f}s", flush=True)

    return SharedCache(key, dst, _load_split(dst, "train"), _load_split(dst, "val"),
                       {k: np.asarray(v, dtype=np.float32) for k, v in
                        json.loads(manifest.read_text())["norm_stats"].items()},
                       was_hit=False)


def prebuild(datasets: list[str], dataset_dir: str | None = None,
             root: Path | None = None) -> dict[str, str]:
    """Build every cache serially before any training job starts."""
    out = {}
    for name in datasets:
        c = build_or_load(name, dataset_dir=dataset_dir, root=root)
        out[name] = str(c.path)
        print(f"  {name:34s} {'hit ' if c.was_hit else 'built'} "
              f"train_obs={c.train.obs.shape} val_obs={c.val.obs.shape}", flush=True)
    return out
