#!/usr/bin/env python3
"""One-GPU-per-rank PyTorch/TreeWM gate for the formal campaign.

The quick probe is deliberately repeated after every requeue because a successor may
land on different nodes.  The fuller EGL/rematerialization/checkpoint/authentication
probe is keyed by protocol, live trainer source, runtime, and this campaign's launch
code, so its durable success sentinel can be shared by array elements safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_WORKERS = 16


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_contract(repo_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(repo_root))
    from treewm.utils.provenance import runtime_fingerprint, trainer_code_fingerprint

    return {
        "code": trainer_code_fingerprint(repo_root),
        "runtime": runtime_fingerprint(),
    }


def cache_key(protocol_lock: Path, repo_root: Path) -> str:
    """Bind a cached full probe to every input that can change its meaning."""

    campaign_dir = Path(__file__).resolve().parent
    files = [
        Path(__file__).resolve(),
        campaign_dir / "train.slurm",
        protocol_lock.resolve(),
    ]
    payload = {
        "contract": _load_contract(repo_root.resolve()),
        "files": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
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
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass


def _quick_torch_smoke() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one visible CUDA device, available={torch.cuda.is_available()} "
            f"count={torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    left = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    value = (left @ torch.eye(4, device=device)).sum()
    torch.cuda.synchronize(device)
    if float(value.item()) != 120.0:
        raise RuntimeError("tiny CUDA computation returned the wrong result")
    return {
        "torch_cuda_device_count": torch.cuda.device_count(),
        "torch_cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "preflight_level": "quick",
    }


def _full_treewm_smoke(repo_root: Path, output_dir: Path, rank: int) -> dict[str, Any]:
    """Exercise the named method, JAX-free GPU graph, remat, and atomic restore."""

    sys.path.insert(0, str(repo_root))
    import numpy as np
    import ogbench
    import torch

    from treewm.models.baselines import build_model
    from treewm.models.treewm import TreeWMConfig
    from treewm.utils.checkpoint import load_checkpoint, save_checkpoint

    env = ogbench.make_env_and_datasets("scene-play-v0", env_only=True)
    try:
        env.reset(seed=rank)
        frame = env.render()
        if frame is None or np.asarray(frame).size == 0:
            raise RuntimeError("OGBench EGL render returned no pixels")
    finally:
        env.close()

    device = torch.device("cuda:0")
    cfg = TreeWMConfig(
        obs_dim=8,
        action_dim=3,
        z_dim=16,
        q_dim=8,
        hidden_dim=32,
        encoder_hidden=32,
        num_layers=2,
        num_heads=4,
        branch_factor=4,
        h_max=8,
        horizons=(2, 4, 8),
        scales=(("short", 2, 1.0),),
        max_depth=8,
    )
    model = build_model("treewm", cfg).to(device).train()
    if model.__class__.__name__ != "TreeWM" or model.default_scorer != "learned":
        raise RuntimeError("formal method resolved to the wrong model/scorer")
    model.set_gradient_checkpointing(True)
    if model.branch_transformer.gradient_checkpointing is not True:
        raise RuntimeError("TreeWM activation rematerialization did not enable")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    observation = torch.randn(4, cfg.obs_dim, device=device)
    children = model.predict_children(model.encode(observation))
    loss = (
        children["latent"].square().mean()
        + children["action_chunk"].square().mean()
        + children["expansion_gain"].square().mean()
    )
    if not torch.isfinite(loss):
        raise RuntimeError("TreeWM smoke loss is non-finite")
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.branch_transformer.parameters()):
        raise RuntimeError("rematerialized transformer produced no gradients")
    optimizer.step()
    torch.cuda.synchronize(device)

    identity = {"kind": "treewm-gpu-preflight", "rank": rank, "arm": "treewm"}
    with tempfile.TemporaryDirectory(prefix=f"rank{rank}.", dir=output_dir) as temp:
        checkpoint = Path(temp) / "latest.pt"
        save_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=1,
            extra={"run_identity": identity, "completed_updates": 1},
        )
        restored = build_model("treewm", cfg).to(device)
        payload = load_checkpoint(
            checkpoint,
            restored,
            map_location="cuda:0",
            restore_rng=False,
            expected_identity=identity,
        )
        if payload.get("completed_updates") != 1:
            raise RuntimeError("atomic checkpoint did not restore the completed update")

    return {
        "treewm_full_update": True,
        "arm": "treewm",
        "model_class": model.__class__.__name__,
        "scorer": model.default_scorer,
        "gradient_checkpointing": True,
        "checkpoint_restore": True,
        "ogbench_egl_render": True,
        **{k: v for k, v in _quick_torch_smoke().items() if k != "preflight_level"},
        "preflight_level": "full",
    }


def _wandb_auth_readonly() -> bool:
    import wandb

    return bool(wandb.Api(timeout=30).viewer)


def probe(
    output_dir: Path,
    repo_root: Path,
    *,
    quick: bool = False,
    skip_gpu: bool = False,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_text = os.environ.get("SLURM_PROCID")
    if rank_text is None or not rank_text.isdigit():
        print("SLURM_PROCID is missing or invalid", file=sys.stderr)
        return 2
    rank = int(rank_text)
    visible = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
    if len(visible) != 1:
        print(f"rank {rank}: expected one visible GPU, got {visible!r}", file=sys.stderr)
        return 2
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "rank": rank,
        "local_rank": int(os.environ.get("SLURM_LOCALID", "-1")),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": visible,
        "unix_time": time.time(),
    }
    try:
        if skip_gpu:
            payload.update(
                {
                    "torch_cuda_device_count": 1,
                    "torch_cuda_device": "test-only",
                    "torch_version": "test-only",
                    "preflight_level": "quick" if quick else "full",
                }
            )
            if not quick:
                payload.update(
                    {
                        "treewm_full_update": True,
                        "arm": "treewm",
                        "model_class": "TreeWM",
                        "scorer": "learned",
                        "gradient_checkpointing": True,
                        "checkpoint_restore": True,
                        "ogbench_egl_render": True,
                        "wandb_auth_readonly": True if rank == 0 else None,
                    }
                )
        elif quick:
            payload.update(_quick_torch_smoke())
        else:
            payload.update(_full_treewm_smoke(repo_root, output_dir, rank))
            payload["wandb_auth_readonly"] = _wandb_auth_readonly() if rank == 0 else None
    except Exception as exc:
        print(f"rank {rank}: TreeWM GPU preflight failed: {exc}", file=sys.stderr)
        return 2
    atomic_json(output_dir / f"rank.{rank}.json", payload)
    print(f"rank {rank}: TreeWM GPU preflight OK ({visible[0]})", flush=True)
    return 0


def verify(output_dir: Path, workers: int, *, level: str) -> int:
    failures: list[str] = []
    payloads: list[dict[str, Any]] = []
    for rank in range(workers):
        try:
            payload = json.loads((output_dir / f"rank.{rank}.json").read_text())
        except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            failures.append(f"rank {rank}: {exc}")
            continue
        if payload.get("status") != "ok" or payload.get("rank") != rank:
            failures.append(f"rank {rank}: invalid payload")
        if len(payload.get("cuda_visible_devices", [])) != 1:
            failures.append(f"rank {rank}: not bound to exactly one visible GPU")
        if payload.get("torch_cuda_device_count") != 1:
            failures.append(f"rank {rank}: PyTorch did not see exactly one GPU")
        if payload.get("preflight_level") != level:
            failures.append(f"rank {rank}: expected {level} preflight")
        if level == "full":
            expected = {
                "treewm_full_update": True,
                "arm": "treewm",
                "model_class": "TreeWM",
                "scorer": "learned",
                "gradient_checkpointing": True,
                "checkpoint_restore": True,
                "ogbench_egl_render": True,
            }
            for key, value in expected.items():
                if payload.get(key) != value:
                    failures.append(f"rank {rank}: {key} did not match {value!r}")
            if rank == 0 and payload.get("wandb_auth_readonly") is not True:
                failures.append("rank 0: read-only W&B authentication failed")
        payloads.append(payload)

    hosts: dict[str, int] = {}
    local_ranks: dict[str, set[int]] = {}
    for payload in payloads:
        host = str(payload.get("hostname"))
        hosts[host] = hosts.get(host, 0) + 1
        local_ranks.setdefault(host, set()).add(int(payload.get("local_rank", -1)))
    if sorted(hosts.values()) != [8, 8]:
        failures.append(f"expected two hosts with eight ranks each, got {hosts}")
    for host, ranks in local_ranks.items():
        if ranks != set(range(8)):
            failures.append(f"host {host}: expected local ranks 0..7, got {sorted(ranks)}")
    if failures:
        print("GPU preflight verification failed:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 2
    print(f"GPU preflight: {workers} ranks, two nodes x eight one-GPU workers", flush=True)
    return 0


def validate_success_sentinel(path: Path, cache_key_value: str, workers: int) -> bool:
    """A nonempty file is not evidence; require the exact full-gate contract."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("schema_version") == 1
        and payload.get("status") == "full_gpu_preflight_complete"
        and payload.get("cache_key") == cache_key_value
        and payload.get("workers") == workers == EXPECTED_WORKERS
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    parser.add_argument("--workers", type=int, default=EXPECTED_WORKERS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-level", choices=("quick", "full"), default="full")
    parser.add_argument("--print-cache-key", action="store_true")
    parser.add_argument("--print-contract", action="store_true")
    parser.add_argument("--protocol-lock", type=Path, default=here / "protocol.sha256")
    parser.add_argument("--success-sentinel", type=Path)
    parser.add_argument("--check-success-sentinel", action="store_true")
    parser.add_argument("--cache-key-value")
    parser.add_argument("--skip-gpu", action="store_true", help="test-only synthetic result")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.print_cache_key:
        print(cache_key(args.protocol_lock, args.repo_root.resolve()))
        return 0
    if args.print_contract:
        print(json.dumps(_load_contract(args.repo_root.resolve()), sort_keys=True))
        return 0
    if args.check_success_sentinel:
        if args.success_sentinel is None or not args.cache_key_value:
            print(
                "--check-success-sentinel requires --success-sentinel and --cache-key-value",
                file=sys.stderr,
            )
            return 2
        return (
            0
            if validate_success_sentinel(
                args.success_sentinel.resolve(), args.cache_key_value, args.workers
            )
            else 2
        )
    if args.output_dir is None:
        print("--output-dir is required for probe/verify", file=sys.stderr)
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
                    "schema_version": 1,
                    "status": "full_gpu_preflight_complete",
                    "cache_key": args.cache_key_value,
                    "workers": args.workers,
                    "unix_time": time.time(),
                },
            )
        return status
    return probe(
        args.output_dir.resolve(),
        args.repo_root.resolve(),
        quick=args.quick,
        skip_gpu=args.skip_gpu,
    )


if __name__ == "__main__":
    raise SystemExit(main())
