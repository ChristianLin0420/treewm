"""Small, dependency-light helpers for resumable RQL training.

This file is local infrastructure around the vendored RQL implementation.  It
deliberately has no JAX, Flax, OGBench, or W&B dependency so checkpoint and
preemption behavior can be unit tested on a CPU-only login node.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import pickle
import platform
import random
import re
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CHECKPOINT_SCHEMA_VERSION = 1
COMPLETION_SCHEMA_VERSION = 1
GRACEFUL_EXIT_CODE = 75
RUNTIME_DISTRIBUTIONS = {
    "jax": "jax",
    "jaxlib": "jaxlib",
    "flax": "flax",
    "optax": "optax",
    "distrax": "distrax",
    "einops": "einops",
    "ml_collections": "ml-collections",
    "gymnasium": "gymnasium",
    "ogbench": "ogbench",
    "wandb": "wandb",
    "numpy": "numpy",
    "mujoco": "mujoco",
}
TRAINER_FINGERPRINT_FILES = (
    'main.py',
    'agents/__init__.py',
    'agents/rql.py',
    'utils/csv_logger.py',
    'utils/datasets.py',
    'utils/evaluation.py',
    'utils/flax_utils.py',
    'utils/log_utils.py',
    'utils/networks.py',
    'utils/resume.py',
    'envs/env_utils.py',
    'envs/ogbench_utils.py',
)


class GracefulStop(RuntimeError):
    """Raised at a safe point after a signal or wall-clock deadline."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class StopController:
    """Signal-safe stop flag plus an optional monotonic wall-clock deadline."""

    def __init__(self, walltime_seconds: float = 0.0, clock=time.monotonic):
        if walltime_seconds < 0:
            raise ValueError("walltime_seconds must be non-negative")
        self._clock = clock
        self._deadline = clock() + walltime_seconds if walltime_seconds else None
        self._reason: str | None = None

    @property
    def reason(self) -> str | None:
        if self._reason is not None:
            return self._reason
        if self._deadline is not None and self._clock() >= self._deadline:
            return "walltime"
        return None

    def request(self, reason: str) -> None:
        # CPython assignment is safe in a Python signal handler.  Do not perform
        # file or JAX operations from the handler itself.
        if self._reason is None:
            self._reason = reason

    def signal_handler(self, signum: int, _frame: Any) -> None:
        try:
            reason = signal.Signals(signum).name
        except ValueError:
            reason = f"signal-{signum}"
        self.request(reason)

    def raise_if_requested(self) -> None:
        reason = self.reason
        if reason is not None:
            raise GracefulStop(reason)


def install_stop_handlers(controller: StopController) -> None:
    """Install deferred handlers for Slurm's preemption signals."""

    signal.signal(signal.SIGUSR1, controller.signal_handler)
    signal.signal(signal.SIGTERM, controller.signal_handler)


def capture_rng_state() -> dict[str, Any]:
    """Capture the global RNGs used by upstream sampling and Python helpers."""

    return {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore global NumPy and Python RNG state from a checkpoint."""

    if "numpy" not in state or "python" not in state:
        raise ValueError("checkpoint RNG state is incomplete")
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def collect_runtime_provenance() -> dict[str, Any]:
    """Collect software and non-host-specific platform provenance.

    Distribution metadata avoids importing optional GPU libraries or creating a
    JAX device.  Hostname is intentionally excluded because a requeued job may
    resume on another node.
    """

    packages = {}
    for label, distribution in RUNTIME_DISTRIBUTIONS.items():
        try:
            packages[label] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[label] = None
    libc_name, libc_version = platform.libc_ver()
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "executable": sys.executable,
        },
        "packages": packages,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "libc": {"name": libc_name or None, "version": libc_version or None},
        },
    }


def stable_json_hash(value: Mapping[str, Any]) -> str:
    """Return a reproducible SHA-256 for a JSON-compatible run identity."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def trainer_code_fingerprint(upstream_dir: os.PathLike[str] | str) -> dict[str, Any]:
    """Hash every runtime-critical local trainer source without importing JAX."""

    root = Path(upstream_dir).resolve()
    files = {}
    for relative_path in TRAINER_FINGERPRINT_FILES:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f'runtime-critical trainer source is missing: {path}')
        files[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        'files': files,
        'manifest_sha256': stable_json_hash(files),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def to_jsonable(value: Any) -> Any:
    """Recursively convert common metric values to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # JAX scalar arrays expose item(), but importing JAX here would defeat the
    # CPU-only nature of this module.
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"cannot convert {type(value).__name__} to JSON")


def atomic_pickle_dump(value: Any, path: os.PathLike[str] | str) -> None:
    """Durably replace a pickle without exposing a partial destination file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            pickle.dump(value, output, protocol=pickle.HIGHEST_PROTOCOL)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def atomic_json_dump(value: Any, path: os.PathLike[str] | str) -> None:
    """Durably replace a JSON document without exposing a partial file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            json.dump(to_jsonable(value), output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Some parallel filesystems reject directory fsync.  The same-directory
        # atomic rename still prevents readers from observing partial content.
        pass


def make_checkpoint(payload: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    """Attach versioned identity metadata to a training-state payload."""

    result = dict(payload)
    result.update(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "identity": dict(identity),
            "identity_sha256": stable_json_hash(identity),
        }
    )
    return result


def load_checkpoint(
    path: os.PathLike[str] | str,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Load a checkpoint and reject incompatible run identities."""

    checkpoint_path = Path(path)
    with checkpoint_path.open("rb") as source:
        checkpoint = pickle.load(source)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema {checkpoint.get('schema_version')!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    expected_hash = stable_json_hash(expected_identity)
    if checkpoint.get("identity_sha256") != expected_hash or checkpoint.get("identity") != dict(expected_identity):
        raise ValueError("checkpoint identity does not match this run's immutable configuration")
    return checkpoint


def evaluation_due(step: int, total_steps: int, eval_interval: int) -> tuple[bool, bool]:
    """Return ``(periodic, final)`` for an absolute completed update step."""

    if step < 1 or total_steps < 1 or step > total_steps:
        raise ValueError("invalid evaluation step")
    periodic = eval_interval != 0 and (step == 1 or step % eval_interval == 0)
    final = step == total_steps
    return periodic, final


def evaluation_episode_seeds(training_seed: int, eval_step: int, episode_index: int) -> tuple[int, int]:
    """Derive deterministic, independent environment and policy seeds."""

    if eval_step < 0 or episode_index < 0:
        raise ValueError("eval_step and episode_index must be non-negative")
    sequence = np.random.SeedSequence(
        [training_seed & 0xFFFFFFFF, eval_step & 0xFFFFFFFF, episode_index & 0xFFFFFFFF]
    )
    env_seed, actor_seed = sequence.generate_state(2, dtype=np.uint32)
    return int(env_seed), int(actor_seed)


def discover_official_100m_shards(
    directory: os.PathLike[str] | str,
    env_name: str,
) -> list[str]:
    """Validate and return the exact official 100M train-shard sequence.

    A formal 100M directory must contain exactly the 100 contiguous train files
    and their 100 paired validation files. Any other ``.npz`` is rejected so a
    stale, duplicate-style, or unrelated cache entry cannot enter rotation.
    """

    match = re.fullmatch(r'(?P<base>.+)-singletask(?:-task[0-9]+)?-v0', env_name)
    if match is None:
        raise ValueError(f'cannot derive official 100M file stem from env_name {env_name!r}')
    file_stem = f"{match.group('base')}-v0"
    dataset_dir = Path(directory).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f'100M dataset directory does not exist: {dataset_dir}')

    expected_train = [f'{file_stem}-{index:03d}.npz' for index in range(100)]
    expected_validation = [f'{file_stem}-{index:03d}-val.npz' for index in range(100)]
    expected = set(expected_train) | set(expected_validation)
    actual_paths = [path for path in dataset_dir.iterdir() if path.suffix.lower() == '.npz']
    actual = {path.name for path in actual_paths}
    non_files = sorted(path.name for path in actual_paths if not path.is_file())
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected or non_files or len(actual_paths) != 200:
        def preview(names):
            shown = ', '.join(names[:5])
            return shown + (f', ... (+{len(names) - 5})' if len(names) > 5 else '')

        details = []
        if missing:
            details.append(f'missing {len(missing)} [{preview(missing)}]')
        if unexpected:
            details.append(f'unexpected {len(unexpected)} [{preview(unexpected)}]')
        if non_files:
            details.append(f'non-files {len(non_files)} [{preview(non_files)}]')
        if len(actual_paths) != 200:
            details.append(f'found {len(actual_paths)} .npz entries; expected exactly 200')
        raise ValueError(f'Invalid official 100M shard directory {dataset_dir}: ' + '; '.join(details))
    return [str(dataset_dir / name) for name in expected_train]


def shard_index_for_step(step: int, replace_interval: int, shard_count: int) -> int:
    """Compute the official rotating 100M shard for an absolute update step."""

    if step < 1:
        raise ValueError("step must be positive")
    if replace_interval <= 0:
        raise ValueError("replace_interval must be positive")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    # Upstream starts on shard 0 and switches before steps interval, 2*interval,
    # ... .  Computing from the absolute step makes a re-run idempotent even if
    # preemption occurs between loading the next shard and taking the update.
    return (step // replace_interval) % shard_count
