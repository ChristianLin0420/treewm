#!/usr/bin/env python3
"""Inspect, recover, or explicitly cancel one authenticated Exp24 transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

# Stable v1 compatibility boundary. Later worktree schemas must retain this
# reader; the authenticated direct snapshot command is also printed in plans.
DISPATCH_CAMPAIGN_ID = "treewm-executable-prefix-formal-v1-launch1"
DISPATCH_POLICY = "stable_v1_minimal_snapshot_cancel_recover_capsule"
PINNED_PYTHON = (
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/envs/treewm-formal-py311/bin/python"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DispatchError(RuntimeError):
    """The stable pre-import emergency-dispatch boundary failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DispatchError(message)


def _require_mutation_bootstrap() -> None:
    _require(
        sys.executable == PINNED_PYTHON
        and sys.version_info[:3] == (3, 11, 15)
        and sys.flags.isolated == 1
        and sys.flags.no_site == 1
        and sys.flags.dont_write_bytecode == 1,
        "cancellation/recovery requires exact pinned Python 3.11 with -I -S -B",
    )


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _absolute(path: Path, label: str) -> Path:
    value = path.absolute()
    _require(
        value.is_absolute()
        and all(part not in {"", ".", ".."} for part in value.parts[1:]),
        f"{label} is not an absolute normalized path",
    )
    return value


def _safe_relative(raw: object, label: str) -> PurePosixPath:
    _require(isinstance(raw, str), f"{label} is not text")
    value = PurePosixPath(raw)
    _require(
        raw == str(value)
        and not value.is_absolute()
        and raw not in {"", "."}
        and all(part not in {"", ".", ".."} for part in value.parts),
        f"{label} is unsafe",
    )
    return value


def _open_absolute_directory(path: Path, label: str) -> int:
    value = _absolute(path, label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(value.anchor, flags)
    try:
        for component in value.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        _require(stat.S_ISDIR(os.fstat(descriptor).st_mode), f"{label} is not a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _require_directory(path: Path, mode: int, label: str) -> None:
    descriptor = _open_absolute_directory(path, label)
    try:
        info = os.fstat(descriptor)
        _require(
            info.st_uid == os.getuid()
            and info.st_gid == os.getgid()
            and info.st_nlink >= 1
            and stat.S_IMODE(info.st_mode) == mode,
            f"{label} ownership/mode differs",
        )
    finally:
        os.close(descriptor)


def _immutable_bytes(path: Path, label: str) -> tuple[bytes, str]:
    parent_fd = _open_absolute_directory(path.parent, f"{label} parent")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        entry = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        _require(
            stat.S_ISREG(before.st_mode)
            and _identity(before) == _identity(entry)
            and before.st_uid == os.getuid()
            and before.st_gid == os.getgid()
            and before.st_nlink == 1
            and stat.S_IMODE(before.st_mode) == 0o444,
            f"{label} identity/mode differs",
        )
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 16 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
        entry_after = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        _require(
            _identity(before) == _identity(after) == _identity(entry_after),
            f"{label} changed while read",
        )
        return b"".join(chunks), digest.hexdigest()
    except OSError as exc:
        raise DispatchError(f"{label} cannot be opened safely: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _immutable_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    payload, digest = _immutable_bytes(path, label)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DispatchError(f"non-finite JSON in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DispatchError(f"{label} is malformed: {exc}") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    return value, digest


def _dispatch_contract(
    submission_root: Path,
) -> tuple[dict[str, Any], Path, PurePosixPath, dict[str, str]] | None:
    """Authenticate stable v1 envelope and only its minimal snapshot capsule."""

    submission_root = _absolute(submission_root, "submission root")
    contract_path = submission_root / "SUBMISSION_CONTRACT.json"
    try:
        contract_path.lstat()
    except FileNotFoundError:
        # A pre-contract --recover has no scheduler authority yet and may use
        # current code. Any other lexical entry/error is fatal.
        return None
    except OSError as exc:
        raise DispatchError(f"submission contract availability is ambiguous: {exc}") from exc
    _require_directory(submission_root, 0o700, "submission root")
    contract, _contract_sha = _immutable_json(contract_path, "submission contract")
    _require(isinstance(contract.get("contract_body_sha256"), str), "contract body hash is absent")
    body = dict(contract)
    claimed_body_sha = body.pop("contract_body_sha256")
    _require(
        SHA256.fullmatch(claimed_body_sha) is not None
        and claimed_body_sha == _stable_hash(body),
        "submission contract body hash differs",
    )
    _require(
        contract.get("schema_version") == 1
        and contract.get("status") == "prepared_scheduler_transaction"
        and contract.get("campaign_id") == DISPATCH_CAMPAIGN_ID
        and contract.get("submission_root") == str(submission_root),
        "submission contract dispatch identity differs",
    )
    snapshot_root = submission_root / "source-snapshot" / "repo"
    _require(contract.get("snapshot_root") == str(snapshot_root), "snapshot dispatch root differs")
    inventory = contract.get("snapshot_inventory")
    _require(
        isinstance(inventory, dict)
        and inventory
        and contract.get("snapshot_inventory_sha256") == _stable_hash(inventory)
        and all(
            isinstance(key, str) and SHA256.fullmatch(str(value)) is not None
            for key, value in inventory.items()
        ),
        "snapshot dispatch inventory binding differs",
    )
    envelope = contract.get("emergency_dispatch")
    _require(isinstance(envelope, dict) and set(envelope) == {
        "schema_version", "campaign_id", "package_relative", "python",
        "targets", "policy",
    }, "emergency dispatch envelope schema differs")
    _require(
        envelope.get("schema_version") == 1
        and envelope.get("campaign_id") == DISPATCH_CAMPAIGN_ID
        and envelope.get("python") == PINNED_PYTHON
        and envelope.get("policy") == DISPATCH_POLICY,
        "emergency dispatch envelope identity differs",
    )
    package_relative = _safe_relative(envelope.get("package_relative"), "dispatch package path")
    expected_targets = sorted(
        str(package_relative / name)
        for name in ("campaign.py", "cancel.py", "manifest.json", "runtime.py")
    )
    raw_targets = envelope.get("targets")
    _require(
        isinstance(raw_targets, dict)
        and list(raw_targets) == expected_targets
        and all(
            inventory.get(relative) == digest
            and SHA256.fullmatch(str(digest)) is not None
            for relative, digest in raw_targets.items()
        ),
        "emergency dispatch target inventory differs",
    )
    _require_directory(submission_root / "source-snapshot", 0o555, "source snapshot parent")
    _require_directory(snapshot_root, 0o555, "snapshot root")
    current = snapshot_root
    for component in package_relative.parts:
        current /= component
        _require_directory(current, 0o555, f"snapshot package directory {component}")
    verified: dict[str, str] = {}
    manifest_value: Mapping[str, Any] | None = None
    for relative in expected_targets:
        path = snapshot_root / Path(relative)
        payload, digest = _immutable_bytes(path, f"emergency target {relative}")
        _require(digest == raw_targets[relative], f"emergency target hash differs: {relative}")
        verified[relative] = digest
        if relative == str(package_relative / "manifest.json"):
            try:
                decoded = json.loads(
                    payload.decode("utf-8"),
                    object_pairs_hook=_pairs,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        DispatchError(f"non-finite snapshot manifest value: {token}")
                    ),
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DispatchError(f"snapshot manifest is malformed: {exc}") from exc
            _require(isinstance(decoded, dict), "snapshot manifest is not an object")
            manifest_value = decoded
    _require(
        manifest_value is not None
        and _stable_hash(manifest_value) == contract.get("manifest_sha256")
        and manifest_value.get("campaign_id") == DISPATCH_CAMPAIGN_ID
        and (manifest_value.get("paths") or {}).get("python") == PINNED_PYTHON,
        "snapshot manifest dispatch binding differs",
    )
    return contract, snapshot_root, package_relative, verified


def _snapshot_dispatch_decision(
    submission_root: Path,
    raw_argv: Sequence[str],
) -> tuple[str, list[str] | None]:
    """Classify absent, already-resident, or exact-snapshot-exec dispatch."""

    dispatch = _dispatch_contract(submission_root)
    if dispatch is None:
        return "contract_absent", None
    _contract, snapshot_root, package_relative, _verified = dispatch
    program = snapshot_root / Path(package_relative) / "cancel.py"
    if Path(__file__).absolute() == program:
        return "already_snapshot_resident", None
    _require("--snapshot-resident" not in raw_argv, "live caller forged snapshot-resident mode")
    return (
        "exec_authenticated_snapshot",
        [PINNED_PYTHON, "-I", "-S", "-B", str(program), "--snapshot-resident", *raw_argv],
    )


def snapshot_dispatch_command(submission_root: Path, raw_argv: Sequence[str]) -> list[str] | None:
    """Return the authenticated snapshot command, or None when no exec is needed."""

    _state, command = _snapshot_dispatch_decision(submission_root, raw_argv)
    return command


def _exec_snapshot(command: Sequence[str]) -> None:
    os.execve(
        command[0],
        list(command),
        {
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    raise AssertionError("snapshot cancel execve unexpectedly returned")


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--test-only", action="store_true")
    modes.add_argument("--cancel", action="store_true")
    modes.add_argument("--recover", action="store_true")
    parser.add_argument("--submission-root", type=Path)
    parser.add_argument("--snapshot-resident", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(raw_argv)
    if args.submission_root is None:
        if args.cancel or args.recover:
            parser.error("mutation/recovery requires --submission-root")
        print(json.dumps({
            "schema_version": 1,
            "status": "no_submission_selected",
            "persistent_writes_performed": False,
            "scheduler_calls": [],
            "policy": "receipt_exact_ids_only_latch_before_scancel",
        }, sort_keys=True, indent=2, allow_nan=False))
        return 0
    if args.cancel or args.recover:
        # No contract/snapshot/package code is read before this mutation guard.
        _require_mutation_bootstrap()
    dispatch_state, dispatch = _snapshot_dispatch_decision(args.submission_root, raw_argv)
    if dispatch is not None:
        _exec_snapshot(dispatch)
    if args.snapshot_resident:
        _require(
            dispatch_state == "already_snapshot_resident",
            "snapshot-resident mode has no authenticated snapshot contract",
        )
    if dispatch_state == "contract_absent":
        _require(
            args.cancel or args.recover,
            "a receipt-bound cancellation plan requires an authenticated snapshot contract",
        )

        # The first absence observation carries no authority.  Serialize against
        # submission, then repeat the stable stdlib-only decision while holding
        # the same external lock used from claim through receipt.  A publication
        # that won the race transfers control to its exact snapshot; only a still
        # pre-contract recovery may continue through current code under the lock.
        import campaign
        import runtime

        locked_dispatch: list[str] | None = None
        locked_result: dict[str, Any] | None = None
        with runtime.transaction_recovery_lock(args.submission_root) as lock_handle:
            locked_state, locked_dispatch = _snapshot_dispatch_decision(
                args.submission_root,
                raw_argv,
            )
            if locked_state == "contract_absent":
                _require(
                    args.recover,
                    "cancellation requires an authenticated snapshot contract and receipt",
                )
                manifest = campaign.load_manifest()
                boundary = runtime.SchedulerBoundary(
                    runner=runtime.default_runner,
                    observer=lambda: {},
                    expected={},
                )
                locked_result = runtime._recover_transaction_locked(
                    manifest,
                    submission_root=args.submission_root,
                    boundary=boundary,
                    lock_handle=lock_handle,
                )
            else:
                _require(
                    locked_state == "exec_authenticated_snapshot"
                    and locked_dispatch is not None,
                    "contract publication race did not resolve to the exact snapshot",
                )
        if locked_dispatch is not None:
            _exec_snapshot(locked_dispatch)
        _require(locked_result is not None, "locked pre-contract recovery produced no result")
        print(json.dumps(locked_result, sort_keys=True, indent=2, allow_nan=False))
        return 0

    # Package-owned code is imported only after dispatch transferred control or
    # authenticated that this file is the bound snapshot implementation.
    import campaign
    import runtime

    if args.snapshot_resident:
        runtime.require(
            (
                args.submission_root
                / "source-snapshot"
                / "repo"
                / runtime.PACKAGE_RELATIVE
                / "cancel.py"
            )
            == Path(__file__).absolute(),
            "snapshot-resident cancel mode was not executed from the bound snapshot",
        )
    manifest = campaign.load_manifest()
    if args.recover:
        contract_path = args.submission_root / "SUBMISSION_CONTRACT.json"
        if contract_path.exists():
            contract, _contract_sha = runtime.authenticated_immutable_json(
                contract_path,
                "recovery submission contract",
            )
            boundary = runtime.boundary_from_submission_contract(manifest["execution"], contract)
        else:
            boundary = runtime.SchedulerBoundary(
                runner=runtime.default_runner,
                observer=lambda: {},
                expected={},
            )
        result = runtime.recover_transaction(
            manifest,
            submission_root=args.submission_root,
            boundary=boundary,
        )
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    plan = runtime.cancellation_plan(args.submission_root)
    if not args.cancel:
        print(json.dumps(
            {**plan, "status": "authenticated_read_only_cancellation_plan"},
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ))
        return 0
    contract, _contract_sha = runtime.authenticated_immutable_json(
        args.submission_root / "SUBMISSION_CONTRACT.json",
        "cancellation submission contract",
    )
    boundary = runtime.boundary_from_submission_contract(manifest["execution"], contract)
    result = runtime.explicit_cancel(
        args.submission_root,
        boundary=boundary,
        execution=manifest["execution"],
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"EXP24_CANCEL_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
