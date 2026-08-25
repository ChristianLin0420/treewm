#!/usr/bin/env python3
"""Tiny one-GPU-per-rank JAX smoke test used before formal training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def trainer_runtime_software(upstream_dir: Path) -> dict[str, Any]:
    """Read package versions from the trainer's authoritative curated list."""

    resume_path = upstream_dir.resolve() / "utils" / "resume.py"
    spec = importlib.util.spec_from_file_location("_rql_preflight_runtime", resume_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load trainer runtime contract: {resume_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provenance = module.collect_runtime_provenance()
    return {
        "python": provenance["python"],
        "packages": provenance["packages"],
    }


def preflight_cache_key(protocol_lock: Path, upstream_dir: Path) -> str:
    """Key durable success to protocol, runtime source, and package versions."""

    upstream_dir = upstream_dir.resolve()
    files = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "train.slurm",
        protocol_lock.resolve(),
    ]
    files.extend(
        sorted(
            path
            for path in upstream_dir.rglob("*")
            if path.is_file() and path.suffix in {".py", ".txt", ".sh"}
        )
    )
    payload = {
        "runtime_software": trainer_runtime_software(upstream_dir),
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _full_rql_smoke(upstream_dir: Path, output_dir: Path, rank: int) -> dict[str, Any]:
    """Compile one representative rematerialized RQL update and restore it."""

    sys.path.insert(0, str(upstream_dir))
    import flax
    import jax
    import jax.numpy as jnp
    import numpy as np
    import ogbench

    from agents.rql import RQLAgent, get_config
    from utils.resume import atomic_pickle_dump, load_checkpoint, make_checkpoint

    # Exercise OGBench registration plus MuJoCo's EGL renderer without loading
    # a dataset (prepare_data.py already checked all 416 NPZ files).
    env = ogbench.make_env_and_datasets(
        "scene-play-singletask-task1-v0",
        env_only=True,
    )
    try:
        env.reset(seed=rank)
        frame = env.render()
        if frame is None or getattr(frame, "size", 0) == 0:
            raise RuntimeError("OGBench EGL render returned no pixels")
    finally:
        env.close()

    config = get_config()
    config.h = 5
    config.batch_size = 256
    config.gradient_checkpointing = True
    observation_dim = 64
    action_dim = 8
    agent = RQLAgent.create(
        rank,
        jnp.zeros((1, observation_dim), dtype=jnp.float32),
        jnp.zeros((1, action_dim), dtype=jnp.float32),
        config,
    )
    trajectory = config.h + 1
    batch = {
        "observations": jnp.zeros((trajectory, config.batch_size, observation_dim), dtype=jnp.float32),
        "actions": jnp.zeros((trajectory, config.batch_size, action_dim), dtype=jnp.float32),
        "rewards": jnp.zeros((trajectory, config.batch_size), dtype=jnp.float32),
        "terminals": jnp.zeros((trajectory, config.batch_size), dtype=jnp.float32),
        "masks": jnp.ones((trajectory, config.batch_size), dtype=jnp.float32),
    }
    updated_agent, metrics = agent.update(batch)
    metrics = jax.tree_util.tree_map(lambda value: value.block_until_ready(), metrics)
    if not all(np.isfinite(np.asarray(value)).all() for value in jax.tree_util.tree_leaves(metrics)):
        raise RuntimeError("synthetic RQL update produced non-finite metrics")

    identity = {
        "kind": "gpu-preflight",
        "rank": rank,
        "h": 5,
        "batch_size": 256,
        "gradient_checkpointing": True,
    }
    state = flax.serialization.to_state_dict(updated_agent)
    with tempfile.TemporaryDirectory(prefix=f"rank{rank}.", dir=output_dir) as temporary:
        checkpoint_path = Path(temporary) / "checkpoint.pkl"
        atomic_pickle_dump(make_checkpoint({"agent": state}, identity), checkpoint_path)
        restored_payload = load_checkpoint(checkpoint_path, identity)
        restored_agent = flax.serialization.from_state_dict(agent, restored_payload["agent"])
        # Materialize a leaf to ensure deserialization really rebuilt the tree.
        first_leaf = jax.tree_util.tree_leaves(restored_agent.network.params)[0]
        first_leaf.block_until_ready()

    devices = jax.local_devices()
    return {
        "jax_device_count": len(devices),
        "jax_platform": devices[0].platform,
        "jax_device": str(devices[0]),
        "ogbench_egl_render": True,
        "rql_full_update": True,
        "trajectory_steps": trajectory,
        "batch_size": config.batch_size,
        "gradient_checkpointing": bool(config.gradient_checkpointing),
        "checkpoint_restore": True,
    }


def _wandb_auth_readonly() -> bool:
    import wandb

    api = wandb.Api(timeout=30)
    return bool(api.viewer)


def _quick_jax_smoke() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    devices = jax.local_devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"expected one JAX GPU, got {devices}")
    value = jnp.add(jnp.ones((1,), dtype=jnp.float32), 1).block_until_ready()
    if float(value[0]) != 2.0:
        raise RuntimeError("tiny device computation returned the wrong value")
    return {
        "jax_device_count": len(devices),
        "jax_platform": devices[0].platform,
        "jax_device": str(devices[0]),
        "preflight_level": "quick",
    }


def probe(
    output_dir: Path,
    upstream_dir: Path,
    *,
    skip_jax: bool = False,
    quick: bool = False,
) -> int:
    # All 16 ranks may race here; mkdir(exist_ok=True) is safe and must happen
    # before the temporary atomic-checkpoint directory is created.
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_text = os.environ.get("SLURM_PROCID")
    if rank_text is None or not rank_text.isdigit():
        print("SLURM_PROCID is missing or invalid", file=sys.stderr)
        return 2
    rank = int(rank_text)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_devices = [item.strip() for item in visible.split(",") if item.strip()]
    if len(visible_devices) != 1:
        print(f"rank {rank}: expected one CUDA_VISIBLE_DEVICES entry, got {visible!r}", file=sys.stderr)
        return 2

    payload: dict[str, Any] = {
        "status": "ok",
        "rank": rank,
        "local_rank": int(os.environ.get("SLURM_LOCALID", "-1")),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": visible_devices,
        "unix_time": time.time(),
    }
    if not skip_jax:
        try:
            if quick:
                payload.update(_quick_jax_smoke())
            else:
                payload.update(_full_rql_smoke(upstream_dir, output_dir, rank))
                payload["preflight_level"] = "full"
            if payload["jax_device_count"] != 1 or payload["jax_platform"] != "gpu":
                raise RuntimeError(f"expected exactly one JAX GPU, got {payload}")
            if not quick:
                payload["wandb_auth_readonly"] = _wandb_auth_readonly() if rank == 0 else None
        except Exception as exc:
            print(f"rank {rank}: JAX GPU preflight failed: {exc}", file=sys.stderr)
            return 2
    else:
        payload.update({"preflight_level": "quick" if quick else "full"})
        if not quick:
            payload.update({
                "jax_device_count": 1,
                "jax_platform": "skipped",
                "ogbench_egl_render": True,
                "rql_full_update": True,
                "trajectory_steps": 6,
                "batch_size": 256,
                "gradient_checkpointing": True,
                "checkpoint_restore": True,
                "wandb_auth_readonly": True if rank == 0 else None,
            })
        else:
            payload.update({"jax_device_count": 1, "jax_platform": "skipped"})

    atomic_json(output_dir / f"rank.{rank}.json", payload)
    print(f"rank {rank}: GPU preflight OK ({visible_devices[0]})", flush=True)
    return 0


def verify(output_dir: Path, workers: int, level: str = "full") -> int:
    failures: list[str] = []
    payloads: list[dict[str, Any]] = []
    for rank in range(workers):
        path = output_dir / f"rank.{rank}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            failures.append(f"rank {rank}: {exc}")
            continue
        if payload.get("status") != "ok" or payload.get("rank") != rank:
            failures.append(f"rank {rank}: invalid payload")
        if len(payload.get("cuda_visible_devices", [])) != 1:
            failures.append(f"rank {rank}: not bound to exactly one visible GPU")
        if payload.get("jax_device_count") != 1:
            failures.append(f"rank {rank}: JAX did not see exactly one device")
        if payload.get("preflight_level") != level:
            failures.append(f"rank {rank}: expected {level} preflight payload")
        if level == "full":
            for key in ("ogbench_egl_render", "rql_full_update", "gradient_checkpointing", "checkpoint_restore"):
                if payload.get(key) is not True:
                    failures.append(f"rank {rank}: {key} did not pass")
            if payload.get("trajectory_steps") != 6 or payload.get("batch_size") != 256:
                failures.append(f"rank {rank}: synthetic trajectory/batch shape contract drifted")
            if rank == 0 and payload.get("wandb_auth_readonly") is not True:
                failures.append("rank 0: read-only W&B authentication failed")
        payloads.append(payload)
    hosts: dict[str, int] = {}
    host_local_ranks: dict[str, set[int]] = {}
    for payload in payloads:
        host = str(payload.get("hostname"))
        hosts[host] = hosts.get(host, 0) + 1
        host_local_ranks.setdefault(host, set()).add(int(payload.get("local_rank", -1)))
    if sorted(hosts.values()) != [8, 8]:
        failures.append(f"expected two hosts with eight ranks each, got {hosts}")
    for host, local_ranks in host_local_ranks.items():
        if local_ranks != set(range(8)):
            failures.append(
                f"host {host}: expected local ranks 0..7 exactly once, got {sorted(local_ranks)}"
            )
    if failures:
        print("GPU preflight verification failed:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 2
    print(f"GPU preflight: 16 ranks across two nodes, one JAX GPU per rank ({hosts})", flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "upstream_rql",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--quick", action="store_true", help="current-allocation binding/JAX/topology check only")
    parser.add_argument("--verify-level", choices=("quick", "full"), default="full")
    parser.add_argument("--print-cache-key", action="store_true")
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path(__file__).resolve().parent / "protocol.sha256",
    )
    parser.add_argument("--success-sentinel", type=Path)
    parser.add_argument("--cache-key-value")
    parser.add_argument("--skip-jax", action="store_true", help="test-only environment validation")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.print_cache_key:
        print(preflight_cache_key(args.protocol_lock, args.upstream_dir))
        return 0
    if args.output_dir is None:
        print("--output-dir is required unless --print-cache-key is used", file=sys.stderr)
        return 2
    if args.verify:
        status = verify(args.output_dir.resolve(), args.workers, level=args.verify_level)
        if status == 0 and args.success_sentinel is not None:
            if not args.cache_key_value or len(args.cache_key_value) != 64:
                print("--cache-key-value must be a 64-character digest", file=sys.stderr)
                return 2
            atomic_json(
                args.success_sentinel.resolve(),
                {
                    "status": "full_gpu_preflight_complete",
                    "cache_key": args.cache_key_value,
                    "unix_time": time.time(),
                    "workers": args.workers,
                },
            )
        return status
    return probe(
        args.output_dir.resolve(),
        args.upstream_dir.resolve(),
        skip_jax=args.skip_jax,
        quick=args.quick,
    )


if __name__ == "__main__":
    raise SystemExit(main())
