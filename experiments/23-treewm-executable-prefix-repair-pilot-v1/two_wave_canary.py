#!/usr/bin/env python3
"""Explicit login-side controller for the tiny Launch8 two-wave GPU canary.

The default ``--describe`` action is read-only.  Real scheduler mutation requires
both ``--submit-real-gpu-two-wave-canary`` and the exact confirmation phrase.
Scientific submission and all package preflights never invoke this controller.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Mapping, Sequence


CAMPAIGN_ID = "treewm-executable-prefix-repair-pilot-v1-launch8"
PACKAGE_RELATIVE = Path("experiments/23-treewm-executable-prefix-repair-pilot-v1")
CONFIRMATION = "SUBMIT_EXP23_LAUNCH8_REAL_GPU_TWO_WAVE_CANARY"
RUNTIME_PROOF_SCOPE = (
    "isolated lexical pinned-Python execution, location-validated Torch import, one "
    "visible selected CUDA device, a real CUDA tensor operation, and cross-wave "
    "checkpoint transfer; no resolved-interpreter, pyvenv, Torch-distribution, "
    "native-library, driver/runtime byte or version binding, and no scientific-runtime "
    "equivalence claim"
)
CANARY_SOURCE_FILES = (
    "two_wave_canary.py",
    "canary_worker.py",
    "canary_gpu.slurm",
    "canary_report.slurm",
)
CANARY_PARENT_RELATIVE = Path("outputs/exp23-launch8-two-wave-canaries")


class CanarySubmissionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CanarySubmissionError(message)


def _failed_canary_identities(
    manifest: Mapping[str, Any],
) -> tuple[set[Path], set[str], set[str]]:
    launch = manifest.get("launch_contract")
    require(isinstance(launch, Mapping), "canary launch contract differs")
    canary = launch.get("real_gpu_two_wave_canary")
    require(isinstance(canary, Mapping), "real-GPU canary contract differs")
    attempts = canary.get("failed_attempts")
    require(isinstance(attempts, list), "failed canary attempts differ")
    roots: set[Path] = set()
    tokens: set[str] = set()
    job_ids: set[str] = set()
    for attempt in attempts:
        require(isinstance(attempt, Mapping), "failed canary attempt differs")
        raw_root = attempt.get("state_root")
        token = attempt.get("canary_token")
        role_ids = attempt.get("job_ids_by_role")
        require(
            type(raw_root) is str
            and Path(raw_root).is_absolute()
            and Path(raw_root) == Path(raw_root).absolute()
            and all(part not in {"", ".", ".."} for part in Path(raw_root).parts[1:]),
            "failed canary state root differs",
        )
        require(
            type(token) is str
            and len(token) == 16
            and all(character in "0123456789abcdef" for character in token),
            "failed canary token differs",
        )
        require(
            isinstance(role_ids, Mapping)
            and set(role_ids) == {"wave0", "wave1", "report"},
            "failed canary job identities differ",
        )
        flattened: list[str] = []
        for role in ("wave0", "wave1", "report"):
            values = role_ids[role]
            require(
                isinstance(values, list)
                and all(
                    type(value) is str
                    and value.isascii()
                    and value.isdigit()
                    and not value.startswith("0")
                    for value in values
                ),
                "failed canary job identities differ",
            )
            flattened.extend(values)
        require(
            len(flattened) == len(set(flattened)),
            "failed canary job identities are not injective",
        )
        roots.add(Path(raw_root))
        tokens.add(token)
        job_ids.update(flattened)
    require(
        len(roots) == len(tokens) == len(attempts),
        "failed canary identities are not injective",
    )
    return roots, tokens, job_ids


class _CanaryControllerLock:
    """Serialize submit/recovery/cancel over one persistent canary-state inode."""

    def __init__(self, state_root: Path, *, create: bool) -> None:
        self.path = state_root / ".CANARY_CONTROLLER.lock"
        self.create = create
        self.descriptor: int | None = None

    def __enter__(self) -> "_CanaryControllerLock":
        flags = (
            (os.O_RDWR | os.O_CREAT) if self.create else os.O_RDONLY
        ) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise CanarySubmissionError(
                "canary controller lock is unavailable"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            named = self.path.lstat()
            require(
                stat.S_ISREG(opened.st_mode)
                and (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)
                and opened.st_uid == os.getuid()
                and opened.st_nlink == 1
                and (self.create or stat.S_IMODE(opened.st_mode) == 0o600),
                "canary controller lock identity differs",
            )
            if self.create:
                os.fchmod(descriptor, 0o600)
                os.fsync(descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except BaseException:
            os.close(descriptor)
            raise
        self.descriptor = descriptor
        return self

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        assert self.descriptor is not None
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.descriptor)
            self.descriptor = None

    def binding(self) -> dict[str, Any]:
        require(self.descriptor is not None, "canary controller lock is not held")
        opened = os.fstat(self.descriptor)
        named = self.path.lstat()
        require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)
            and opened.st_uid == os.getuid()
            and opened.st_nlink == 1
            and stat.S_IMODE(opened.st_mode) == 0o600,
            "canary controller lock identity changed",
        )
        return {
            "path": str(self.path),
            "device": opened.st_dev,
            "inode": opened.st_ino,
            "uid": opened.st_uid,
            "mode": stat.S_IMODE(opened.st_mode),
        }


def _runner_with_canary_lock_lease(runner, lock: _CanaryControllerLock):
    """Make every scheduler child retain the controller flock through exec."""

    descriptor = lock.descriptor
    assert descriptor is not None
    binding = lock.binding()

    def leased(command, cwd, environment, inherited_fds=()):
        require(lock.binding() == binding, "canary scheduler-call lock lease differs")
        descriptors = tuple(dict.fromkeys([*inherited_fds, descriptor]))
        return runner(command, cwd, environment, descriptors)

    setattr(leased, "controller_lock_binding", binding)
    return leased


def _load_submit(repo_root: Path):
    path = repo_root / PACKAGE_RELATIVE / "submit.py"
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), "Exp23 submit source is unavailable or symlinked")
    spec = importlib.util.spec_from_file_location("_exp23_launch8_canary_submit", path)
    require(spec is not None and spec.loader is not None, "cannot load Exp23 submit module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_campaign(repo_root: Path):
    path = repo_root / PACKAGE_RELATIVE / "campaign.py"
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), "Exp23 campaign source is unavailable or symlinked")
    spec = importlib.util.spec_from_file_location("_exp23_launch8_canary_campaign", path)
    require(spec is not None and spec.loader is not None, "cannot load Exp23 campaign module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_canary_worker(state_root: Path):
    path = state_root / "source" / "canary_worker.py"
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), "canary worker snapshot is unavailable or symlinked")
    spec = importlib.util.spec_from_file_location(
        f"_exp23_launch8_recovery_worker_{os.getpid()}", path
    )
    require(spec is not None and spec.loader is not None, "cannot load canary worker snapshot")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(
        Path(str(module.__file__)).absolute() == path.absolute(),
        "canary worker snapshot import escaped state root",
    )
    return module


def description() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "real_gpu_two_wave_canary_available_not_run",
        "campaign_id": CAMPAIGN_ID,
        "scientific": False,
        "scheduler_mutation_on_describe": False,
        "preflight_invocation": False,
        "gpu_tasks": 2,
        "cpu_report_tasks": 1,
        "graph": ["wave0-held", "afterok-wave1", "afterok-report"],
        "within_wave_requeue": False,
        "runtime_proof_scope": RUNTIME_PROOF_SCOPE,
        "confirmation_phrase": CONFIRMATION,
        "hard_crash_action": "--recover-or-cancel-real-gpu-canary",
    }


def _prepare_state(
    repo_root: Path, state_root: Path, manifest: Mapping[str, Any]
) -> tuple[Path, dict[str, str]]:
    root = state_root.absolute()
    require(
        root == state_root
        and root.is_absolute()
        and all(part not in {"", ".", ".."} for part in root.parts[1:]),
        "canary state root must be an absolute normalized path",
    )
    require(root.name.startswith("exp23-launch8-two-wave-canary-"), "canary state root name differs")
    outputs = (repo_root / "outputs").absolute()
    require(outputs.is_dir() and not outputs.is_symlink(), "repository outputs root is unavailable or symlinked")
    dedicated_parent = (repo_root / CANARY_PARENT_RELATIVE).absolute()
    if not os.path.lexists(dedicated_parent):
        dedicated_parent.mkdir(mode=0o700)
    require(
        dedicated_parent.is_dir()
        and not dedicated_parent.is_symlink()
        and root.parent == dedicated_parent,
        "canary state root is outside its dedicated parent",
    )
    forbidden = [Path(str(manifest["paths"]["run_root"])).absolute()]
    current_lock = Path(str(manifest["paths"]["transaction_lock"]))
    forbidden.append(
        (repo_root / current_lock).absolute()
        if not current_lock.is_absolute()
        else current_lock.absolute()
    )
    for superseded in manifest.get("superseded_launches", []):
        require(isinstance(superseded, Mapping), "superseded launch contract differs")
        forbidden.extend(
            Path(str(superseded[key])).absolute()
            for key in ("run_root", "submission_root")
        )
        if superseded.get("transaction_lock"):
            lock = Path(str(superseded["transaction_lock"]))
            forbidden.append(
                (repo_root / lock).absolute() if not lock.is_absolute() else lock.absolute()
            )
    failed_roots, _failed_tokens, _failed_job_ids = _failed_canary_identities(manifest)
    forbidden.extend(failed_roots)
    require(
        all(
            not root.is_relative_to(path)
            and not path.is_relative_to(root)
            and not dedicated_parent.is_relative_to(path)
            and dedicated_parent != path
            for path in forbidden
        ),
        "canary state namespace overlaps a scientific/superseded/failed-canary namespace",
    )
    require(not os.path.lexists(root), "canary state root already exists")
    root.mkdir(mode=0o700)
    (root / "logs").mkdir(mode=0o700)
    source = root / "source"
    source.mkdir(mode=0o700)
    hashes: dict[str, str] = {}
    for name in CANARY_SOURCE_FILES:
        origin = repo_root / PACKAGE_RELATIVE / name
        info = origin.lstat()
        require(stat.S_ISREG(info.st_mode), f"canary source is unavailable or symlinked: {name}")
        destination = source / name
        source_descriptor = os.open(
            origin,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        destination_descriptor: int | None = None
        try:
            opened = os.fstat(source_descriptor)
            require(
                stat.S_ISREG(opened.st_mode)
                and (opened.st_dev, opened.st_ino, opened.st_size)
                == (info.st_dev, info.st_ino, info.st_size),
                f"canary source raced before copy: {name}",
            )
            chunks: list[bytes] = []
            while True:
                block = os.read(source_descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            payload = b"".join(chunks)
            require(
                os.fstat(source_descriptor).st_size == opened.st_size
                and len(payload) == opened.st_size,
                f"canary source raced during copy: {name}",
            )
            destination_descriptor = os.open(
                destination,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            view = memoryview(payload)
            offset = 0
            while offset < len(view):
                written = os.write(destination_descriptor, view[offset:])
                require(written > 0, f"canary snapshot write stalled: {name}")
                offset += written
            os.fsync(destination_descriptor)
            require(
                os.fstat(destination_descriptor).st_size == len(payload),
                f"canary snapshot size differs: {name}",
            )
            os.lseek(destination_descriptor, 0, os.SEEK_SET)
            copied = bytearray()
            while True:
                block = os.read(destination_descriptor, 1024 * 1024)
                if not block:
                    break
                copied.extend(block)
            require(bytes(copied) == payload, f"canary snapshot bytes differ: {name}")
            os.fchmod(destination_descriptor, 0o444)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        finally:
            if destination_descriptor is not None:
                os.close(destination_descriptor)
            os.close(source_descriptor)
    source.chmod(0o555)
    return root, hashes


def _require_direct_canary_acceptance(
    submit: Any, job_id: str, record: Mapping[str, Any], role: str
) -> None:
    """Canary release requires a direct parseable response, not recovery-only acceptance."""

    stdout = record.get("stdout")
    match = submit.SBATCH_JOB.fullmatch(stdout.strip()) if isinstance(stdout, str) else None
    require(
        match is not None
        and match.group("job_id") == job_id
        and type(record.get("returncode")) is int
        and record.get("returncode") == 0
        and record.get("reconciled_job_ids") == [job_id],
        f"canary {role} lacks an exact direct accepted response",
    )


def submit_real_canary(
    repo_root: Path,
    state_root: Path,
    confirmation: str,
    *,
    scheduler_runner=None,
) -> dict[str, Any]:
    require(confirmation == CONFIRMATION, "real GPU canary confirmation phrase differs")
    require(
        bool(sys.flags.isolated)
        and bool(sys.flags.no_site)
        and bool(sys.flags.dont_write_bytecode)
        and bool(sys.dont_write_bytecode)
        and bool(sys.flags.safe_path),
        "real GPU canary controller requires pinned Python -I -S -B",
    )
    forbidden_environment = sorted(
        key
        for key in os.environ
        if key.startswith("TREEWM_")
        or key in {"RANK", "WORLD_SIZE", "LOCAL_RANK"}
    )
    require(
        not forbidden_environment,
        "real GPU canary inherited forbidden environment: "
        + ", ".join(forbidden_environment),
    )
    require(
        not any(name == "treewm" or name.startswith("treewm.") for name in sys.modules),
        "treewm was imported before canary package validation",
    )
    submit = _load_submit(repo_root)
    campaign = _load_campaign(repo_root)
    runner = submit._default_scheduler_runner if scheduler_runner is None else scheduler_runner
    manifest = submit.read_json(repo_root / PACKAGE_RELATIVE / "manifest.json")
    require(manifest.get("campaign_id") == CAMPAIGN_ID, "canary manifest campaign differs")
    require(
        manifest.get("status") == "sealed_launch_ready_unsubmitted"
        and manifest.get("formal_validation") is False,
        "real GPU canary requires the final sealed Launch8 package",
    )
    validated_manifest, _weight_lock = campaign.load_contract(repo_root)
    require(
        submit.exact_json_equal(validated_manifest, manifest),
        "canary live manifest validation differs",
    )
    protocol_sha256 = campaign.verify_protocol_lock(repo_root / PACKAGE_RELATIVE)
    pinned = Path(str(manifest["paths"]["python"])).resolve(strict=True)
    require(Path(sys.executable).resolve(strict=True) == pinned, "canary controller interpreter is not pinned")
    _failed_roots, failed_tokens, failed_job_ids = _failed_canary_identities(manifest)
    token = secrets.token_hex(8)
    require(
        type(token) is str
        and len(token) == 16
        and all(character in "0123456789abcdef" for character in token),
        "new canary token syntax differs",
    )
    require(token not in failed_tokens, "new canary token reuses a failed canary token")
    root, source_hashes = _prepare_state(repo_root, state_root, manifest)
    live_source_hashes = {
        name: submit.file_sha256(repo_root / PACKAGE_RELATIVE / name)
        for name in CANARY_SOURCE_FILES
    }
    require(
        submit.exact_json_equal(live_source_hashes, source_hashes)
        and campaign.verify_protocol_lock(repo_root / PACKAGE_RELATIVE)
        == protocol_sha256,
        "canary source/protocol generation changed while snapshotting",
    )
    worker = root / "source" / "canary_worker.py"
    gpu_script = root / "source" / "canary_gpu.slurm"
    report_script = root / "source" / "canary_report.slurm"
    execution = manifest["execution"]
    control_plane = submit._scheduler_contract(execution.get("scheduler_control_plane"))
    fallback = submit._scheduler_fallback_config(control_plane)
    authorization = fallback["source_control_plane"]
    sbatch = str(execution["sbatch"])
    squeue = str(execution.get("squeue") or (Path(sbatch).parent / "squeue"))
    scontrol = str(execution["scontrol"])
    scancel = str(execution["scancel"])
    for path, label in ((sbatch, "sbatch"), (squeue, "squeue"), (scontrol, "scontrol"), (scancel, "scancel")):
        submit._regular_nonsymlink(Path(path), f"canary {label}")
        require(os.access(path, os.X_OK), f"canary {label} is not executable")

    names = {
        "wave0": f"exp23-launch8-canary-{token}-wave0",
        "wave1": f"exp23-launch8-canary-{token}-wave1",
        "report": f"exp23-launch8-canary-{token}-report",
    }
    comment = f"treewm-exp23-canary:{token}"
    known: dict[str, list[str]] = {role: [] for role in names}
    active_role = "wave0"
    observations: list[dict[str, Any]] = []
    controller_lock = _CanaryControllerLock(root, create=True)
    controller_lock.__enter__()
    try:
        runner = _runner_with_canary_lock_lease(runner, controller_lock)
        controller_lock_binding = controller_lock.binding()
        submit.exclusive_json(
            root / "CANARY_CONTROLLER_IDENTITY.json",
            {
                "schema_version": 1,
                "status": "canary_controller_claimed",
                "campaign_id": CAMPAIGN_ID,
                "scientific": False,
                "state_root": str(root),
                "canary_token": token,
                "job_names": names,
                "scheduler_comment": comment,
                "controller_lock": controller_lock_binding,
                "source_sha256": source_hashes,
                "package_protocol_sha256": protocol_sha256,
                "scheduler_control_plane": authorization,
                "scheduler_control_plane_sha256": submit.stable_hash(
                    authorization
                ),
            },
        )
        for role, name in names.items():
            submit._assert_job_absent(
                squeue,
                name,
                comment,
                root,
                runner,
                control_plane,
                fallback,
                authorization,
                observations,
            )
        wave0_command = [
            sbatch,
            "--parsable",
            "--export=NONE",
            "--hold",
            f"--job-name={names['wave0']}",
            f"--comment={comment}",
            f"--output={root / 'logs/wave0_%j.out'}",
            str(gpu_script),
            str(worker),
            str(root),
            "wave0",
        ]
        wave0_calling_sha256 = submit.exclusive_json(
            root / "CANARY_WAVE0_CALLING.json",
            {
                "schema_version": 1,
                "status": "canary_scheduler_calling",
                "campaign_id": CAMPAIGN_ID,
                "state_root": str(root),
                "canary_token": token,
                "role": "wave0",
                "job_name": names["wave0"],
                "scheduler_comment": comment,
                "command": wave0_command,
                "controller_lock": controller_lock_binding,
            },
        )
        wave0_id, wave0_record = submit._submit_one(
            wave0_command,
            job_name=names["wave0"],
            comment=comment,
            squeue=squeue,
            cwd=root,
            runner=runner,
            control_plane=control_plane,
            fallback=fallback,
            expected_observation=authorization,
            observations=observations,
        )
        require(
            wave0_id not in failed_job_ids,
            "canary scheduler reused a failed canary job ID",
        )
        _require_direct_canary_acceptance(submit, wave0_id, wave0_record, "wave0")
        wave0_record = {**wave0_record, "calling_sha256": wave0_calling_sha256}
        known["wave0"].append(wave0_id)
        wave0_hold = submit._accepted_wave0_hold_evidence(
            scontrol=scontrol,
            job_id=wave0_id,
            job_name=names["wave0"],
            comment=comment,
            cwd=root,
            runner=runner,
            control_plane=control_plane,
            expected_observation=authorization,
            observations=observations,
        )
        submit.exclusive_json(
            root / "CANARY_WAVE0_SUBMITTED.json",
            {
                "schema_version": 1,
                "status": "canary_wave0_submitted_held",
                "job_id": wave0_id,
                "accepted_submission_record": wave0_record,
                "accepted_hold": wave0_hold,
            },
        )
        wave1_command = [
            sbatch,
            "--parsable",
            "--export=NONE",
            f"--dependency=afterok:{wave0_id}",
            "--kill-on-invalid-dep=yes",
            f"--job-name={names['wave1']}",
            f"--comment={comment}",
            f"--output={root / 'logs/wave1_%j.out'}",
            str(gpu_script),
            str(worker),
            str(root),
            "wave1",
        ]
        active_role = "wave1"
        wave1_calling_sha256 = submit.exclusive_json(
            root / "CANARY_WAVE1_CALLING.json",
            {
                "schema_version": 1,
                "status": "canary_scheduler_calling",
                "campaign_id": CAMPAIGN_ID,
                "state_root": str(root),
                "canary_token": token,
                "role": "wave1",
                "job_name": names["wave1"],
                "scheduler_comment": comment,
                "command": wave1_command,
                "controller_lock": controller_lock_binding,
            },
        )
        wave1_id, wave1_record = submit._submit_one(
            wave1_command,
            job_name=names["wave1"],
            comment=comment,
            squeue=squeue,
            cwd=root,
            runner=runner,
            control_plane=control_plane,
            fallback=fallback,
            expected_observation=authorization,
            observations=observations,
        )
        require(
            wave1_id not in failed_job_ids,
            "canary scheduler reused a failed canary job ID",
        )
        _require_direct_canary_acceptance(submit, wave1_id, wave1_record, "wave1")
        require(
            wave1_id != wave0_id,
            "canary scheduler assigned one job ID to multiple roles",
        )
        wave1_record = {**wave1_record, "calling_sha256": wave1_calling_sha256}
        known["wave1"].append(wave1_id)
        wave1_dependency = submit._accepted_dependency_evidence(
            scontrol=scontrol,
            job_id=wave1_id,
            predecessor_id=wave0_id,
            job_name=names["wave1"],
            role="wave1",
            predecessor_kind="scalar",
            comment=comment,
            cwd=root,
            runner=runner,
            control_plane=control_plane,
            expected_observation=authorization,
            observations=observations,
        )
        submit.exclusive_json(
            root / "CANARY_WAVE1_SUBMITTED.json",
            {
                "schema_version": 1,
                "status": "canary_wave1_submitted",
                "job_id": wave1_id,
                "accepted_submission_record": wave1_record,
                "accepted_dependency": wave1_dependency,
            },
        )
        report_command = [
            sbatch,
            "--parsable",
            "--export=NONE",
            f"--dependency=afterok:{wave1_id}",
            "--kill-on-invalid-dep=yes",
            f"--job-name={names['report']}",
            f"--comment={comment}",
            f"--output={root / 'logs/report_%j.out'}",
            str(report_script),
            str(worker),
            str(root),
        ]
        active_role = "report"
        report_calling_sha256 = submit.exclusive_json(
            root / "CANARY_REPORT_CALLING.json",
            {
                "schema_version": 1,
                "status": "canary_scheduler_calling",
                "campaign_id": CAMPAIGN_ID,
                "state_root": str(root),
                "canary_token": token,
                "role": "report",
                "job_name": names["report"],
                "scheduler_comment": comment,
                "command": report_command,
                "controller_lock": controller_lock_binding,
            },
        )
        report_id, report_record = submit._submit_one(
            report_command,
            job_name=names["report"],
            comment=comment,
            squeue=squeue,
            cwd=root,
            runner=runner,
            control_plane=control_plane,
            fallback=fallback,
            expected_observation=authorization,
            observations=observations,
        )
        require(
            report_id not in failed_job_ids,
            "canary scheduler reused a failed canary job ID",
        )
        _require_direct_canary_acceptance(submit, report_id, report_record, "report")
        require(
            report_id not in {wave0_id, wave1_id},
            "canary scheduler assigned one job ID to multiple roles",
        )
        report_record = {**report_record, "calling_sha256": report_calling_sha256}
        known["report"].append(report_id)
        report_dependency = submit._accepted_dependency_evidence(
            scontrol=scontrol,
            job_id=report_id,
            predecessor_id=wave1_id,
            job_name=names["report"],
            role="report",
            predecessor_kind="scalar",
            comment=comment,
            cwd=root,
            runner=runner,
            control_plane=control_plane,
            expected_observation=authorization,
            observations=observations,
        )
        submit.exclusive_json(
            root / "CANARY_REPORT_SUBMITTED.json",
            {
                "schema_version": 1,
                "status": "canary_report_submitted",
                "job_id": report_id,
                "accepted_submission_record": report_record,
                "accepted_dependency": report_dependency,
            },
        )
        accepted_submission_records = {
            "wave0": wave0_record,
            "wave1": wave1_record,
            "report": report_record,
        }
        accepted_scheduler_evidence = {
            "wave0_accepted_hold": wave0_hold,
            "wave1_accepted_dependency": wave1_dependency,
            "report_accepted_dependency": report_dependency,
        }
        auth = {
            "schema_version": 1,
            "status": "authorized_two_wave_gpu_canary",
            "campaign_id": CAMPAIGN_ID,
            "canary_token": token,
            "state_root": str(root),
            "controller_identity_sha256": submit.file_sha256(
                root / "CANARY_CONTROLLER_IDENTITY.json"
            ),
            "controller_lock": controller_lock_binding,
            "worker_sha256": source_hashes["canary_worker.py"],
            "source_sha256": source_hashes,
            "package_protocol_sha256": protocol_sha256,
            "job_ids": {"wave0": wave0_id, "wave1": wave1_id, "report": report_id},
            "job_names": names,
            "dependencies": {
                "wave0": "none",
                "wave1": f"afterok:{wave0_id}",
                "report": f"afterok:{wave1_id}",
            },
            "scheduler_comment": comment,
            "scheduler_executables": {"submit": sbatch, "control": scontrol},
            "scheduler_control_plane": authorization,
            "scheduler_control_plane_sha256": submit.stable_hash(authorization),
            "accepted_submission_records_sha256": submit.stable_hash(
                accepted_submission_records
            ),
            "accepted_scheduler_evidence_sha256": submit.stable_hash(
                accepted_scheduler_evidence
            ),
            "within_wave_requeue": False,
        }
        auth_sha256 = submit.exclusive_json(root / "CANARY_AUTHORIZATION.json", auth)
        receipt = {
            "schema_version": 1,
            "status": "two_wave_gpu_canary_ready_to_release",
            "campaign_id": CAMPAIGN_ID,
            "scientific": False,
            "state_root": str(root),
            "canary_token": token,
            "controller_identity_sha256": auth["controller_identity_sha256"],
            "controller_lock": auth["controller_lock"],
            "authorization_sha256": auth_sha256,
            "job_ids": auth["job_ids"],
            "job_names": auth["job_names"],
            "dependencies": auth["dependencies"],
            "scheduler_comment": auth["scheduler_comment"],
            "scheduler_executables": auth["scheduler_executables"],
            "scheduler_control_plane": auth["scheduler_control_plane"],
            "scheduler_control_plane_sha256": auth[
                "scheduler_control_plane_sha256"
            ],
            "source_sha256": source_hashes,
            "package_protocol_sha256": protocol_sha256,
            "accepted_submission_records_sha256": auth[
                "accepted_submission_records_sha256"
            ],
            "accepted_scheduler_evidence_sha256": auth[
                "accepted_scheduler_evidence_sha256"
            ],
            "wave0_accepted_hold": wave0_hold,
            "wave1_accepted_dependency": wave1_dependency,
            "report_accepted_dependency": report_dependency,
            "accepted_submission_records": accepted_submission_records,
        }
        receipt_sha256 = submit.exclusive_json(
            root / "CANARY_SUBMISSION_RECEIPT.json", receipt
        )
        ready_to_release_sha256 = submit.exclusive_json(
            root / "CANARY_READY_TO_RELEASE.json",
            {
                "schema_version": 1,
                "status": "canary_ready_to_release",
                "campaign_id": CAMPAIGN_ID,
                "authorization_sha256": auth_sha256,
                "receipt_sha256": receipt_sha256,
                "job_ids": auth["job_ids"],
                "dependencies": auth["dependencies"],
                "accepted_submission_records_sha256": auth[
                    "accepted_submission_records_sha256"
                ],
                "accepted_scheduler_evidence_sha256": auth[
                    "accepted_scheduler_evidence_sha256"
                ],
            },
        )
        release_command = [scontrol, "release", wave0_id]
        release_calling_sha256 = submit.exclusive_json(
            root / "CANARY_WAVE0_RELEASE_CALLING.json",
            {
                "schema_version": 1,
                "status": "canary_scheduler_calling",
                "campaign_id": CAMPAIGN_ID,
                "state_root": str(root),
                "canary_token": token,
                "role": "wave0_release",
                "job_name": names["wave0"],
                "scheduler_comment": comment,
                "command": release_command,
                "controller_lock": controller_lock_binding,
                "authorization_sha256": auth_sha256,
                "receipt_sha256": receipt_sha256,
            },
        )
        release = submit._release_authorized_wave0(
            scontrol=scontrol,
            job_id=wave0_id,
            job_name=names["wave0"],
            comment=comment,
            cwd=root,
            runner=runner,
            control_plane=control_plane,
            expected_observation=authorization,
            observations=observations,
        )
        submit.exclusive_json(
            root / "CANARY_WAVE0_RELEASED.json",
            {
                "schema_version": 1,
                "status": "canary_wave0_released",
                "campaign_id": CAMPAIGN_ID,
                "authorization_sha256": auth_sha256,
                "receipt_sha256": receipt_sha256,
                "ready_to_release_sha256": ready_to_release_sha256,
                "calling_sha256": release_calling_sha256,
                "wave0_job_id": wave0_id,
                "wave0_release": release,
            },
        )
        return {
            **receipt,
            "status": "two_wave_gpu_canary_submitted_and_released",
            "receipt_sha256": receipt_sha256,
            "wave0_release": release,
        }
    except BaseException as exc:
        cancel_context = {
            "schema_version": 1,
            "status": "canary_scheduler_calling",
            "campaign_id": CAMPAIGN_ID,
            "state_root": str(root),
            "canary_token": token,
            "role": "recovery_cancel",
            "controller_lock": controller_lock_binding,
        }
        abort_evidence = submit._initial_exception_reconcile_and_cancel(
            exception_job_ids=getattr(exc, "job_ids", ()),
            active_role=active_role,
            prior_claimed_ids_by_role=known,
            role_names=names,
            squeue=squeue,
            scancel=scancel,
            scheduler_comment=comment,
            cancel_directory=root,
            cancel_calling_prefix="CANARY_RECOVERY_CANCEL_CALLING",
            cancel_result_prefix="CANARY_RECOVERY_CANCEL_RESULT",
            snapshot_root=root,
            scheduler_runner=runner,
            control_plane=control_plane,
            scheduler_fallback=fallback,
            expected_observation=authorization,
            scheduler_observations=observations,
            cancel_context=cancel_context,
        )
        abort = {
            "schema_version": 1,
            "status": "two_wave_gpu_canary_aborted",
            "campaign_id": CAMPAIGN_ID,
            "state_root": str(root),
            "canary_token": token,
            "controller_lock": controller_lock_binding,
            "error": repr(exc),
            "known_job_ids": abort_evidence["known_job_ids"],
            "job_ids_by_role": abort_evidence["job_ids_by_role"],
            "cancellation_authority_job_ids": abort_evidence[
                "cancellation_authority_job_ids"
            ],
            "cancellation_authority_job_ids_by_role": abort_evidence[
                "cancellation_authority_job_ids_by_role"
            ],
            "reconciliation_errors": abort_evidence["reconciliation_errors"],
            "cancellation": abort_evidence["cancellation"],
            "cancellation_error": abort_evidence["cancellation_error"],
            "cancel_attempt_history": abort_evidence["cancel_attempt_history"],
        }
        try:
            submit.exclusive_json(root / "CANARY_ABORTED.json", abort)
        except BaseException:
            pass
        raise CanarySubmissionError(f"real GPU two-wave canary submission aborted: {exc}") from exc
    finally:
        controller_lock.__exit__(None, None, None)


def _validated_canary_postsubmission_prefix(
    submit: Any,
    worker: Any,
    root: Path,
    *,
    identity: Mapping[str, Any],
    protocol_sha256: str,
    scheduler_executables: Mapping[str, str],
    scheduler_control_plane_sha256: str,
) -> dict[str, Any]:
    """Authenticate each extant auth/receipt/READY/release prefix independently."""

    scheduler_control_plane = identity.get("scheduler_control_plane")
    require(
        isinstance(scheduler_control_plane, Mapping)
        and identity.get("scheduler_control_plane_sha256")
        == scheduler_control_plane_sha256
        and submit.stable_hash(scheduler_control_plane)
        == scheduler_control_plane_sha256,
        "canary controller scheduler control-plane binding differs",
    )

    names = {
        "wave0": "CANARY_WAVE0_SUBMITTED.json",
        "wave1": "CANARY_WAVE1_SUBMITTED.json",
        "report": "CANARY_REPORT_SUBMITTED.json",
    }
    roles = ("wave0", "wave1", "report")
    journal_values: dict[str, dict[str, Any]] = {}
    journal_gap = False
    for role, filename in names.items():
        path = root / filename
        if not os.path.lexists(path):
            journal_gap = True
            continue
        require(not journal_gap, "canary submitted journals are not a contiguous prefix")
        submit._regular_nonsymlink(path, f"canary {role} submitted journal")
        require(
            stat.S_IMODE(path.lstat().st_mode) == 0o444,
            f"canary {role} submitted journal mode differs",
        )
        record = submit.read_json(path)
        expected_fields = (
            {
                "schema_version",
                "status",
                "job_id",
                "accepted_submission_record",
                "accepted_hold",
            }
            if role == "wave0"
            else {
                "schema_version",
                "status",
                "job_id",
                "accepted_submission_record",
                "accepted_dependency",
            }
        )
        require(
            set(record) == expected_fields
            and type(record.get("schema_version")) is int
            and record.get("schema_version") == 1
            and record.get("status")
            == (
                "canary_wave0_submitted_held"
                if role == "wave0"
                else f"canary_{role}_submitted"
            )
            and isinstance(record.get("job_id"), str)
            and submit.JOB_ID.fullmatch(record["job_id"]) is not None,
            f"canary {role} submitted journal differs",
        )
        journal_values[role] = record

    durable_jobs = {
        role: (
            journal_values[role]["job_id"]
            if role in journal_values
            else str(999999990 + index)
        )
        for index, role in enumerate(roles)
    }
    journal_context = {
        "canary_token": identity["canary_token"],
        "job_ids": durable_jobs,
        "job_names": identity["job_names"],
        "scheduler_comment": identity["scheduler_comment"],
        "scheduler_executables": dict(scheduler_executables),
        "scheduler_control_plane": scheduler_control_plane,
        "scheduler_control_plane_sha256": scheduler_control_plane_sha256,
        "controller_lock": identity["controller_lock"],
    }
    calling_names = {
        "wave0": "CANARY_WAVE0_CALLING.json",
        "wave1": "CANARY_WAVE1_CALLING.json",
        "report": "CANARY_REPORT_CALLING.json",
    }
    expected_commands = worker._expected_submission_commands(root, journal_context)
    calling_hashes: dict[str, str] = {}
    calling_gap = False
    for role in roles:
        calling_path = root / calling_names[role]
        if not os.path.lexists(calling_path):
            calling_gap = True
            continue
        require(not calling_gap, "canary calling records are not a contiguous prefix")
        submit._regular_nonsymlink(
            calling_path, f"canary {role} scheduler-call intent"
        )
        require(
            stat.S_IMODE(calling_path.lstat().st_mode) == 0o444,
            f"canary {role} scheduler-call intent mode differs",
        )
        require(
            submit.exact_json_equal(
                submit.read_json(calling_path),
                {
                "schema_version": 1,
                "status": "canary_scheduler_calling",
                "campaign_id": CAMPAIGN_ID,
                "state_root": str(root),
                "canary_token": identity["canary_token"],
                "role": role,
                "job_name": identity["job_names"][role],
                "scheduler_comment": identity["scheduler_comment"],
                "command": expected_commands[role],
                "controller_lock": identity["controller_lock"],
                },
            ),
            f"canary {role} scheduler-call intent differs",
        )
        calling_hashes[role] = submit.file_sha256(calling_path)
    require(
        set(journal_values).issubset(set(calling_hashes))
        and len(calling_hashes) <= len(journal_values) + 1,
        "canary calling/submitted prefix differs",
    )
    suffix = (
        "CANARY_AUTHORIZATION.json",
        "CANARY_SUBMISSION_RECEIPT.json",
        "CANARY_READY_TO_RELEASE.json",
        "CANARY_WAVE0_RELEASE_CALLING.json",
        "CANARY_WAVE0_RELEASED.json",
    )
    presence = [os.path.lexists(root / name) for name in suffix]
    require(
        all(not presence[index] or all(presence[:index]) for index in range(len(presence))),
        "canary post-submission durable prefix has a gap",
    )
    hashes: dict[str, Any] = {
        "authorization": None,
        "receipt": None,
        "ready_to_release": None,
        "release_calling": None,
        "released": None,
        "submitted_job_ids_by_role": {
            role: [journal_values[role]["job_id"]]
            if role in journal_values
            else []
            for role in roles
        },
        "calling_intent_sha256_by_role": calling_hashes,
    }
    if journal_values:
        for role, record in journal_values.items():
            worker._validate_submission_record(
                record["accepted_submission_record"],
                role=role,
                root=root,
                authorization=journal_context,
            )
            worker._validate_scheduler_evidence(
                record[
                    "accepted_hold" if role == "wave0" else "accepted_dependency"
                ],
                role=role,
                root=root,
                authorization=journal_context,
            )
    if not presence[0]:
        return hashes
    require(
        set(journal_values) == {"wave0", "wave1", "report"},
        "canary authorization lacks the complete accepted-job journals",
    )
    jobs = {role: journal_values[role]["job_id"] for role in names}
    require(
        all(submit.JOB_ID.fullmatch(value) is not None for value in jobs.values())
        and len(set(jobs.values())) == 3,
        "canary authorization durable job IDs differ",
    )
    token = identity["canary_token"]
    expected_names = identity["job_names"]
    expected_dependencies = {
        "wave0": "none",
        "wave1": f"afterok:{jobs['wave0']}",
        "report": f"afterok:{jobs['wave1']}",
    }
    accepted_records = {
        role: journal_values[role]["accepted_submission_record"] for role in names
    }
    accepted_evidence = {
        "wave0_accepted_hold": journal_values["wave0"]["accepted_hold"],
        "wave1_accepted_dependency": journal_values["wave1"]["accepted_dependency"],
        "report_accepted_dependency": journal_values["report"]["accepted_dependency"],
    }
    auth_path = root / "CANARY_AUTHORIZATION.json"
    submit._regular_nonsymlink(auth_path, "canary authorization")
    require(stat.S_IMODE(auth_path.lstat().st_mode) == 0o444, "canary authorization mode differs")
    auth = submit.read_json(auth_path)
    require(
        set(auth)
        == {
            "schema_version",
            "status",
            "campaign_id",
            "canary_token",
            "state_root",
            "controller_identity_sha256",
            "controller_lock",
            "worker_sha256",
            "source_sha256",
            "package_protocol_sha256",
            "job_ids",
            "job_names",
            "dependencies",
            "scheduler_comment",
            "scheduler_executables",
            "scheduler_control_plane",
            "scheduler_control_plane_sha256",
            "accepted_submission_records_sha256",
            "accepted_scheduler_evidence_sha256",
            "within_wave_requeue",
        }
        and type(auth.get("schema_version")) is int
        and auth.get("schema_version") == 1
        and auth.get("status") == "authorized_two_wave_gpu_canary"
        and auth.get("campaign_id") == CAMPAIGN_ID
        and auth.get("canary_token") == token
        and auth.get("state_root") == str(root)
        and auth.get("controller_identity_sha256")
        == submit.file_sha256(root / "CANARY_CONTROLLER_IDENTITY.json")
        and submit.exact_json_equal(
            auth.get("controller_lock"), identity["controller_lock"]
        )
        and auth.get("worker_sha256") == identity["source_sha256"]["canary_worker.py"]
        and submit.exact_json_equal(
            auth.get("source_sha256"), identity["source_sha256"]
        )
        and auth.get("package_protocol_sha256") == protocol_sha256
        and submit.exact_json_equal(auth.get("job_ids"), jobs)
        and submit.exact_json_equal(auth.get("job_names"), expected_names)
        and submit.exact_json_equal(
            auth.get("dependencies"), expected_dependencies
        )
        and auth.get("scheduler_comment") == identity["scheduler_comment"]
        and submit.exact_json_equal(
            auth.get("scheduler_executables"), scheduler_executables
        )
        and submit.exact_json_equal(
            auth.get("scheduler_control_plane"), scheduler_control_plane
        )
        and auth.get("scheduler_control_plane_sha256")
        == scheduler_control_plane_sha256
        and auth.get("accepted_submission_records_sha256")
        == submit.stable_hash(accepted_records)
        and auth.get("accepted_scheduler_evidence_sha256")
        == submit.stable_hash(accepted_evidence)
        and auth.get("within_wave_requeue") is False,
        "canary authorization durable semantics differ",
    )
    for role in names:
        worker._validate_submission_record(
            accepted_records[role], role=role, root=root, authorization=auth
        )
        worker._validate_scheduler_evidence(
            accepted_evidence[
                "wave0_accepted_hold"
                if role == "wave0"
                else f"{role}_accepted_dependency"
            ],
            role=role,
            root=root,
            authorization=auth,
        )
    hashes["authorization"] = submit.file_sha256(auth_path)
    if not presence[1]:
        return hashes
    receipt_path = root / "CANARY_SUBMISSION_RECEIPT.json"
    submit._regular_nonsymlink(receipt_path, "canary submission receipt")
    require(stat.S_IMODE(receipt_path.lstat().st_mode) == 0o444, "canary receipt mode differs")
    receipt = submit.read_json(receipt_path)
    require(
        submit.exact_json_equal(
            receipt,
            {
            "schema_version": 1,
            "status": "two_wave_gpu_canary_ready_to_release",
            "campaign_id": CAMPAIGN_ID,
            "scientific": False,
            "state_root": str(root),
            "canary_token": token,
            "controller_identity_sha256": auth["controller_identity_sha256"],
            "controller_lock": auth["controller_lock"],
            "authorization_sha256": hashes["authorization"],
            "job_ids": jobs,
            "job_names": expected_names,
            "dependencies": expected_dependencies,
            "scheduler_comment": identity["scheduler_comment"],
            "scheduler_executables": dict(scheduler_executables),
            "scheduler_control_plane": scheduler_control_plane,
            "scheduler_control_plane_sha256": scheduler_control_plane_sha256,
            "source_sha256": identity["source_sha256"],
            "package_protocol_sha256": protocol_sha256,
            "accepted_submission_records_sha256": auth[
                "accepted_submission_records_sha256"
            ],
            "accepted_scheduler_evidence_sha256": auth[
                "accepted_scheduler_evidence_sha256"
            ],
            "wave0_accepted_hold": accepted_evidence["wave0_accepted_hold"],
            "wave1_accepted_dependency": accepted_evidence[
                "wave1_accepted_dependency"
            ],
            "report_accepted_dependency": accepted_evidence[
                "report_accepted_dependency"
            ],
            "accepted_submission_records": accepted_records,
            },
        ),
        "canary durable receipt differs",
    )
    hashes["receipt"] = submit.file_sha256(receipt_path)
    if not presence[2]:
        return hashes
    ready_path = root / "CANARY_READY_TO_RELEASE.json"
    submit._regular_nonsymlink(ready_path, "canary ready-to-release record")
    require(stat.S_IMODE(ready_path.lstat().st_mode) == 0o444, "canary READY mode differs")
    require(
        submit.exact_json_equal(
            submit.read_json(ready_path),
            {
            "schema_version": 1,
            "status": "canary_ready_to_release",
            "campaign_id": CAMPAIGN_ID,
            "authorization_sha256": hashes["authorization"],
            "receipt_sha256": hashes["receipt"],
            "job_ids": jobs,
            "dependencies": expected_dependencies,
            "accepted_submission_records_sha256": auth[
                "accepted_submission_records_sha256"
            ],
            "accepted_scheduler_evidence_sha256": auth[
                "accepted_scheduler_evidence_sha256"
            ],
            },
        ),
        "canary ready-to-release durable semantics differ",
    )
    hashes["ready_to_release"] = submit.file_sha256(ready_path)
    if not presence[3]:
        return hashes
    calling_path = root / "CANARY_WAVE0_RELEASE_CALLING.json"
    submit._regular_nonsymlink(calling_path, "canary wave-zero release intent")
    require(stat.S_IMODE(calling_path.lstat().st_mode) == 0o444, "canary release intent mode differs")
    require(
        submit.exact_json_equal(
            submit.read_json(calling_path),
            {
            "schema_version": 1,
            "status": "canary_scheduler_calling",
            "campaign_id": CAMPAIGN_ID,
            "state_root": str(root),
            "canary_token": token,
            "role": "wave0_release",
            "job_name": expected_names["wave0"],
            "scheduler_comment": identity["scheduler_comment"],
            "command": [scheduler_executables["control"], "release", jobs["wave0"]],
            "controller_lock": identity["controller_lock"],
            "authorization_sha256": hashes["authorization"],
            "receipt_sha256": hashes["receipt"],
            },
        ),
        "canary wave-zero release intent differs",
    )
    hashes["release_calling"] = submit.file_sha256(calling_path)
    if not presence[4]:
        return hashes
    release_record, released_sha256 = worker._validate_release_record(root, auth)
    require(
        release_record.get("calling_sha256") == hashes["release_calling"],
        "canary wave-zero release result/calling bytes differ",
    )
    hashes["released"] = released_sha256
    return hashes


def _validated_canary_recovery_result(
    submit: Any,
    root: Path,
    value: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    protocol_sha256: str,
    role_names: Mapping[str, str],
    squeue: str,
    scancel: str,
    controller_lock: Mapping[str, Any],
    scheduler_executables: Mapping[str, str],
    scheduler_control_plane_sha256: str,
    expected_control_plane: Mapping[str, Any],
    scheduler_fallback: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "status",
        "campaign_id",
        "state_root",
        "canary_token",
        "controller_identity_sha256",
        "package_protocol_sha256",
        "source_sha256",
        "claimed_job_ids",
        "claimed_job_ids_by_role",
        "live_verified_job_ids",
        "live_verified_job_ids_by_role",
        "calling_intent_sha256_by_role",
        "pre_cancel_census_rounds",
        "cancelled_live_job_ids",
        "cancel_calling_sha256",
        "cancellation",
        "cancel_attempt_history",
        "post_cancel_census_rounds",
        "post_cancel_active_job_ids_by_role",
        "durable_prefix_sha256",
        "scheduler_control_plane_observations",
        "scheduler_calls",
        "controller_lock",
        "new_jobs_created",
    }
    require(set(value) == fields, "prior canary recovery fields differ")
    require(
        type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("state_root") == str(root)
        and value.get("canary_token") == identity["canary_token"]
        and value.get("controller_identity_sha256")
        == submit.file_sha256(root / "CANARY_CONTROLLER_IDENTITY.json")
        and value.get("package_protocol_sha256") == protocol_sha256
        and submit.exact_json_equal(
            value.get("source_sha256"), identity["source_sha256"]
        )
        and submit.exact_json_equal(
            value.get("controller_lock"), controller_lock
        )
        and type(value.get("new_jobs_created")) is int
        and value.get("new_jobs_created") == 0,
        "prior canary recovery identity differs",
    )
    aborted_claims, _abort_history = _validated_canary_abort(
        submit,
        root,
        canary_token=identity["canary_token"],
        controller_lock=controller_lock,
        scancel=scancel,
        expected_control_plane=expected_control_plane,
        scheduler_fallback=scheduler_fallback,
    )
    worker = _load_canary_worker(root)
    durable_prefix = _validated_canary_postsubmission_prefix(
        submit,
        worker,
        root,
        identity=identity,
        protocol_sha256=protocol_sha256,
        scheduler_executables=scheduler_executables,
        scheduler_control_plane_sha256=scheduler_control_plane_sha256,
    )

    def role_map(
        raw: object, label: str, *, require_single_current: bool
    ) -> dict[str, list[str]]:
        require(
            isinstance(raw, Mapping)
            and set(raw) == {"wave0", "wave1", "report"},
            f"canary {label} role map differs",
        )
        result: dict[str, list[str]] = {}
        for role in ("wave0", "wave1", "report"):
            values = raw[role]
            require(isinstance(values, list), f"canary {label} {role} IDs differ")
            require(
                all(
                    isinstance(item, str)
                    and submit.JOB_ID.fullmatch(item) is not None
                    for item in values
                ),
                f"canary {label} {role} IDs differ",
            )
            normalized = list(values)
            require(
                normalized == sorted(set(normalized), key=int)
                and (not require_single_current or len(normalized) <= 1),
                f"canary {label} {role} IDs differ",
            )
            result[role] = normalized
        require(
            len({item for values in result.values() for item in values})
            == sum(len(values) for values in result.values()),
            f"canary {label} assigns one scheduler ID to multiple roles",
        )
        return result

    claimed = role_map(
        value.get("claimed_job_ids_by_role"),
        "claimed",
        require_single_current=False,
    )
    live = role_map(
        value.get("live_verified_job_ids_by_role"),
        "live",
        require_single_current=True,
    )
    post = role_map(
        value.get("post_cancel_active_job_ids_by_role"),
        "post-cancel",
        require_single_current=True,
    )
    submitted_claims = durable_prefix["submitted_job_ids_by_role"]
    require(
        isinstance(submitted_claims, Mapping)
        and set(submitted_claims) == {"wave0", "wave1", "report"},
        "canary recovery durable submitted claims differ",
    )
    expected_claimed = {
        role: sorted(
            set(aborted_claims[role]) | set(submitted_claims[role]), key=int
        )
        for role in ("wave0", "wave1", "report")
    }
    require(
        submit.exact_json_equal(claimed, expected_claimed),
        "canary recovery claimed IDs differ from durable provenance",
    )
    claimed_flat = sorted({item for ids in claimed.values() for item in ids}, key=int)
    live_flat = sorted({item for ids in live.values() for item in ids}, key=int)
    require(
        value.get("claimed_job_ids") == claimed_flat
        and value.get("live_verified_job_ids") == live_flat
        and value.get("cancelled_live_job_ids") == live_flat,
        "canary recovery flat/role IDs differ",
    )
    calling = value.get("calling_intent_sha256_by_role")
    require(
        isinstance(calling, Mapping)
        and submit.exact_json_equal(
            calling, durable_prefix["calling_intent_sha256_by_role"]
        )
        and all(
            isinstance(digest, str)
            and len(digest) == 64
            and set(digest) <= set("0123456789abcdef")
            for digest in calling.values()
        ),
        "canary recovery calling hashes differ",
    )

    def rounds(raw: object, label: str) -> list[dict[str, Any]]:
        require(isinstance(raw, list), f"canary {label} census differs")
        result = []
        for index, row in enumerate(raw):
            require(
                isinstance(row, Mapping)
                and type(row.get("round")) is int
                and row.get("round") == index,
                f"canary {label} census round differs",
            )
            result.append(
                {
                    "round": index,
                    "job_ids_by_role": role_map(
                        row.get("job_ids_by_role"),
                        label,
                        require_single_current=True,
                    ),
                }
            )
        return result

    observations = submit._validated_scheduler_attempt_ledger(
        value.get("scheduler_control_plane_observations"),
        expected_control_plane=expected_control_plane,
        fallback=scheduler_fallback,
    )
    pre, reconstructed_live, attempt_cursor = submit._validated_recovery_census_rounds(
        value.get("pre_cancel_census_rounds"),
        attempts=observations,
        role_names=role_names,
        squeue=squeue,
        comment=identity["scheduler_comment"],
        label="canary pre-cancel",
        expected_start=0,
    )
    require(
        submit.exact_json_equal(reconstructed_live, live),
        "canary pre-cancel census did not settle",
    )
    cancellation = value.get("cancellation")
    cancel_context = {
        "schema_version": 1,
        "status": "canary_scheduler_calling",
        "campaign_id": CAMPAIGN_ID,
        "state_root": str(root),
        "canary_token": identity["canary_token"],
        "role": "recovery_cancel",
        "controller_lock": dict(controller_lock),
    }
    disk_history = submit._validated_recovery_cancel_history(
        root,
        calling_prefix="CANARY_RECOVERY_CANCEL_CALLING",
        result_prefix="CANARY_RECOVERY_CANCEL_RESULT",
        context=cancel_context,
        scancel=scancel,
        expected_control_plane=expected_control_plane,
        fallback=scheduler_fallback,
    )
    stored_history = value.get("cancel_attempt_history")
    require(
        isinstance(stored_history, list)
        and len(stored_history) <= len(disk_history)
        and submit.exact_json_equal(
            stored_history, disk_history[: len(stored_history)]
        ),
        "canary recovery cancellation attempt history differs",
    )
    last_attempt = stored_history[-1] if stored_history else None
    require(
        value.get("cancel_calling_sha256")
        == (last_attempt["calling_sha256"] if last_attempt else None)
        and submit.exact_json_equal(
            cancellation,
            last_attempt["cancellation"] if last_attempt else None,
        ),
        "canary recovery terminal cancellation mirrors differ",
    )
    require(
        type(value.get("scheduler_calls")) is int
        and value.get("scheduler_calls") == len(observations),
        "canary recovery scheduler-call count differs",
    )
    if live_flat:
        require(
            value.get("status") == "canary_recovered_terminal_after_cancel_attempts"
            and last_attempt is not None
            and last_attempt["job_ids"] == live_flat
            and last_attempt["cancellation"] is not None
            and isinstance(value.get("cancel_calling_sha256"), str)
            and isinstance(cancellation, Mapping)
            and cancellation.get("job_ids") == live_flat
            and cancellation.get("command") == [scancel, *live_flat],
            "canary recovery cancellation evidence differs",
        )
        cancellation_attempts = submit._validated_scheduler_attempt_ledger(
            cancellation.get("scheduler_attempts"),
            expected_control_plane=expected_control_plane,
            fallback=scheduler_fallback,
        )
        require(
            submit.exact_json_equal(
                observations[
                    attempt_cursor : attempt_cursor + len(cancellation_attempts)
                ],
                cancellation_attempts,
            ),
            "canary cancellation attempts are not in the scheduler ledger",
        )
        attempt_cursor += len(cancellation_attempts)
        post_rounds, reconstructed_post, attempt_cursor = (
            submit._validated_recovery_census_rounds(
                value.get("post_cancel_census_rounds"),
                attempts=observations,
                role_names=role_names,
                squeue=squeue,
                comment=identity["scheduler_comment"],
                label="canary post-cancel",
                expected_start=attempt_cursor,
            )
        )
        empty = {"wave0": [], "wave1": [], "report": []}
        require(
            submit.exact_json_equal(reconstructed_post, post)
            and submit.exact_json_equal(post, empty),
            "canary recovery jobs remain active after cancellation",
        )
    else:
        require(
            value.get("status")
            == (
                "canary_recovered_terminal_after_cancel_attempts"
                if stored_history
                else "canary_recovered_terminal_no_active_jobs"
            )
            and value.get("post_cancel_census_rounds") == []
            and submit.exact_json_equal(
                post, {"wave0": [], "wave1": [], "report": []}
            ),
            "canary recovery no-active evidence differs",
        )
    require(
        attempt_cursor == len(observations),
        "canary recovery scheduler ledger has unbound attempts",
    )
    require(
        submit.exact_json_equal(
            value.get("durable_prefix_sha256"), durable_prefix
        ),
        "canary recovery durable post-submission prefix differs",
    )
    allowed_queries = {
        (
            squeue,
            "--noheader",
            f"--name={role_names[role]}",
            "--format=%A|%j|%u|%T|%k",
        )
        for role in ("wave0", "wave1", "report")
    }
    require(
        all(
            tuple(attempt["command"]) in allowed_queries
            or attempt["command"] == [scancel, *live_flat]
            for attempt in observations
        ),
        "canary recovery scheduler command escaped exact identities",
    )
    return dict(value)


def _validated_canary_abort(
    submit: Any,
    root: Path,
    *,
    canary_token: str,
    controller_lock: Mapping[str, Any],
    scancel: str,
    expected_control_plane: Mapping[str, Any],
    scheduler_fallback: Mapping[str, Any],
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Bind an initial exception-path abort to its immutable cancel attempts."""

    context = {
        "schema_version": 1,
        "status": "canary_scheduler_calling",
        "campaign_id": CAMPAIGN_ID,
        "state_root": str(root),
        "canary_token": canary_token,
        "role": "recovery_cancel",
        "controller_lock": dict(controller_lock),
    }
    history = submit._validated_recovery_cancel_history(
        root,
        calling_prefix="CANARY_RECOVERY_CANCEL_CALLING",
        result_prefix="CANARY_RECOVERY_CANCEL_RESULT",
        context=context,
        scancel=scancel,
        expected_control_plane=expected_control_plane,
        fallback=scheduler_fallback,
    )
    empty = {"wave0": [], "wave1": [], "report": []}
    path = root / "CANARY_ABORTED.json"
    if not os.path.lexists(path):
        return empty, history
    submit._regular_nonsymlink(path, "canary initial abort")
    require(
        stat.S_IMODE(path.lstat().st_mode) == 0o444,
        "canary initial abort mode differs",
    )
    value = submit.read_json(path)
    require(
        set(value)
        == {
            "schema_version",
            "status",
            "campaign_id",
            "state_root",
            "canary_token",
            "controller_lock",
            "error",
            "known_job_ids",
            "job_ids_by_role",
            "cancellation_authority_job_ids",
            "cancellation_authority_job_ids_by_role",
            "reconciliation_errors",
            "cancellation",
            "cancellation_error",
            "cancel_attempt_history",
        }
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "two_wave_gpu_canary_aborted"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("state_root") == str(root)
        and value.get("canary_token") == canary_token
        and submit.exact_json_equal(
            value.get("controller_lock"), controller_lock
        )
        and isinstance(value.get("error"), str)
        and isinstance(value.get("reconciliation_errors"), list)
        and all(isinstance(item, str) for item in value["reconciliation_errors"])
        and (
            value.get("cancellation_error") is None
            or isinstance(value["cancellation_error"], str)
        ),
        "canary initial abort identity differs",
    )
    raw_roles = value.get("job_ids_by_role")
    require(
        isinstance(raw_roles, Mapping)
        and set(raw_roles) == {"wave0", "wave1", "report"},
        "canary initial abort role IDs differ",
    )
    roles: dict[str, list[str]] = {}
    seen: set[str] = set()
    for role in ("wave0", "wave1", "report"):
        raw = raw_roles[role]
        require(isinstance(raw, list), f"canary initial abort {role} IDs differ")
        require(
            all(
                isinstance(item, str)
                and submit.JOB_ID.fullmatch(item) is not None
                for item in raw
            ),
            f"canary initial abort {role} IDs differ",
        )
        ids = list(raw)
        require(
            ids == sorted(set(ids), key=int)
            and not seen.intersection(ids),
            f"canary initial abort {role} IDs differ",
        )
        roles[role] = ids
        seen.update(ids)
    require(
        value.get("known_job_ids") == sorted(seen, key=int),
        "canary initial abort flat/role IDs differ",
    )
    raw_authority = value.get("cancellation_authority_job_ids_by_role")
    require(
        isinstance(raw_authority, Mapping)
        and set(raw_authority) == {"wave0", "wave1", "report"},
        "canary initial abort cancellation authority differs",
    )
    authority: dict[str, list[str]] = {}
    authority_flat: set[str] = set()
    for role in ("wave0", "wave1", "report"):
        raw = raw_authority[role]
        require(
            isinstance(raw, list)
            and all(
                isinstance(item, str)
                and submit.JOB_ID.fullmatch(item) is not None
                for item in raw
            )
            and raw == sorted(set(raw), key=int)
            and len(raw) <= 1
            and set(raw).issubset(roles[role]),
            f"canary initial abort {role} cancellation authority differs",
        )
        authority[role] = list(raw)
        authority_flat.update(authority[role])
    require(
        value.get("cancellation_authority_job_ids")
        == sorted(authority_flat, key=int),
        "canary initial abort flat cancellation authority differs",
    )
    abort_history = value.get("cancel_attempt_history")
    require(
        isinstance(abort_history, list)
        and len(abort_history) <= len(history)
        and submit.exact_json_equal(
            abort_history, history[: len(abort_history)]
        )
        and submit.exact_json_equal(
            value.get("cancellation"),
            abort_history[-1]["cancellation"] if abort_history else None,
        ),
        "canary initial abort cancellation history differs",
    )
    submit._validated_successful_cancellation(
        value,
        scancel,
        list(value["cancellation_authority_job_ids"]),
    )
    return roles, history


def recover_or_cancel_real_canary(
    repo_root: Path,
    state_root: Path,
    confirmation: str,
    *,
    scheduler_runner=None,
) -> dict[str, Any]:
    """Reconcile/cancel a hard-crashed canary without ever creating a job."""

    require(confirmation == CONFIRMATION, "real GPU canary confirmation phrase differs")
    require(
        bool(sys.flags.isolated)
        and bool(sys.flags.no_site)
        and bool(sys.flags.dont_write_bytecode)
        and bool(sys.dont_write_bytecode)
        and bool(sys.flags.safe_path),
        "real GPU canary recovery requires pinned Python -I -S -B",
    )
    require(
        not any(
            key.startswith("TREEWM_")
            or key in {"RANK", "WORLD_SIZE", "LOCAL_RANK"}
            for key in os.environ
        ),
        "real GPU canary recovery inherited forbidden environment",
    )
    require(
        not any(name == "treewm" or name.startswith("treewm.") for name in sys.modules),
        "treewm was imported before canary recovery package validation",
    )
    submit = _load_submit(repo_root)
    campaign = _load_campaign(repo_root)
    manifest = submit.read_json(repo_root / PACKAGE_RELATIVE / "manifest.json")
    require(
        manifest.get("campaign_id") == CAMPAIGN_ID
        and manifest.get("status") == "sealed_launch_ready_unsubmitted"
        and manifest.get("formal_validation") is False,
        "canary recovery requires the final sealed Launch8 package",
    )
    validated_manifest, _weight_lock = campaign.load_contract(repo_root)
    require(
        submit.exact_json_equal(validated_manifest, manifest),
        "canary recovery manifest validation differs",
    )
    root = state_root.absolute()
    require(
        root == state_root
        and root.parent == (repo_root / CANARY_PARENT_RELATIVE).absolute()
        and root.name.startswith("exp23-launch8-two-wave-canary-"),
        "canary recovery state root differs",
    )
    failed_roots, failed_tokens, failed_job_ids = _failed_canary_identities(
        manifest
    )
    require(
        root not in failed_roots,
        "failed canary state root cannot be recovered or read",
    )
    protocol_sha256 = campaign.verify_protocol_lock(repo_root / PACKAGE_RELATIVE)
    pinned = Path(str(manifest["paths"]["python"])).resolve(strict=True)
    require(Path(sys.executable).resolve(strict=True) == pinned, "canary recovery interpreter is not pinned")
    require(
        not root.parent.is_symlink() and root.parent.is_dir(),
        "canary recovery state parent differs",
    )
    require(
        not root.is_symlink() and root.is_dir(),
        "canary recovery state root differs",
    )
    runner = submit._default_scheduler_runner if scheduler_runner is None else scheduler_runner
    with _CanaryControllerLock(root, create=False) as controller_lock:
        runner = _runner_with_canary_lock_lease(runner, controller_lock)
        identity = submit.read_json(root / "CANARY_CONTROLLER_IDENTITY.json")
        require(
            set(identity)
            == {
                "schema_version",
                "status",
                "campaign_id",
                "scientific",
                "state_root",
                "canary_token",
                "job_names",
                "scheduler_comment",
                "controller_lock",
                "source_sha256",
                "package_protocol_sha256",
                "scheduler_control_plane",
                "scheduler_control_plane_sha256",
            }
            and type(identity.get("schema_version")) is int
            and identity.get("schema_version") == 1
            and identity.get("status") == "canary_controller_claimed"
            and identity.get("campaign_id") == CAMPAIGN_ID
            and identity.get("scientific") is False
            and identity.get("state_root") == str(root)
            and isinstance(identity.get("canary_token"), str)
            and len(identity["canary_token"]) == 16
            and identity.get("package_protocol_sha256") == protocol_sha256
            and set(identity.get("source_sha256") or {}) == set(CANARY_SOURCE_FILES),
            "canary controller identity differs",
        )
        require(
            submit.exact_json_equal(
                identity.get("controller_lock"), controller_lock.binding()
            ),
            "canary recovery controller-lock lineage differs",
        )
        require(
            identity["canary_token"] not in failed_tokens,
            "canary recovery identity reuses a failed canary token",
        )
        require(
            all(
                submit.file_sha256(root / "source" / name) == digest
                and submit.file_sha256(repo_root / PACKAGE_RELATIVE / name) == digest
                for name, digest in identity["source_sha256"].items()
            ),
            "canary recovery source bytes differ",
        )
        token = identity["canary_token"]
        expected_names = {
            "wave0": f"exp23-launch8-canary-{token}-wave0",
            "wave1": f"exp23-launch8-canary-{token}-wave1",
            "report": f"exp23-launch8-canary-{token}-report",
        }
        require(
            identity.get("job_names") == expected_names
            and identity.get("scheduler_comment") == f"treewm-exp23-canary:{token}",
            "canary recovery scheduler identity differs",
        )
        execution = manifest["execution"]
        control_plane = submit._scheduler_contract(
            execution.get("scheduler_control_plane")
        )
        fallback = submit._scheduler_fallback_config(control_plane)
        authorization = fallback["source_control_plane"]
        require(
            submit.exact_json_equal(
                identity.get("scheduler_control_plane"), authorization
            )
            and identity.get("scheduler_control_plane_sha256")
            == submit.stable_hash(authorization),
            "canary recovery scheduler control-plane identity differs",
        )
        squeue = str(execution.get("squeue") or (Path(str(execution["sbatch"])).parent / "squeue"))
        scancel = str(execution["scancel"])
        scheduler_executables = {
            "submit": str(execution["sbatch"]),
            "control": str(execution["scontrol"]),
        }
        scheduler_control_plane_sha256 = submit.stable_hash(authorization)
        for path, label in ((squeue, "squeue"), (scancel, "scancel")):
            submit._regular_nonsymlink(Path(path), f"canary recovery {label}")
            require(os.access(path, os.X_OK), f"canary recovery {label} is not executable")
        aborted_ids_by_role, existing_cancel_history = _validated_canary_abort(
            submit,
            root,
            canary_token=token,
            controller_lock=controller_lock.binding(),
            scancel=scancel,
            expected_control_plane=authorization,
            scheduler_fallback=fallback,
        )
        require(
            all(
                not failed_job_ids.intersection(attempt["job_ids"])
                for attempt in existing_cancel_history
            ),
            "canary recovery cancellation history reuses a failed canary job ID",
        )
        require(
            not failed_job_ids.intersection(
                item
                for values in aborted_ids_by_role.values()
                for item in values
            ),
            "canary recovery abort reuses a failed canary job ID",
        )
        prior_path = root / "CANARY_RECOVERY_CANCELLED.json"
        if os.path.lexists(prior_path):
            submit._regular_nonsymlink(prior_path, "terminal canary recovery result")
            require(
                stat.S_IMODE(prior_path.lstat().st_mode) == 0o444,
                "terminal canary recovery result mode differs",
            )
            prior = _validated_canary_recovery_result(
                submit,
                root,
                submit.read_json(prior_path),
                identity=identity,
                protocol_sha256=protocol_sha256,
                role_names=expected_names,
                squeue=squeue,
                scancel=scancel,
                controller_lock=controller_lock.binding(),
                scheduler_executables=scheduler_executables,
                scheduler_control_plane_sha256=scheduler_control_plane_sha256,
                expected_control_plane=authorization,
                scheduler_fallback=fallback,
            )
            require(
                not failed_job_ids.intersection(prior["claimed_job_ids"]),
                "canary recovery result reuses a failed canary job ID",
            )
            require(
                not failed_job_ids.intersection(prior["live_verified_job_ids"])
                and not failed_job_ids.intersection(
                    prior["cancelled_live_job_ids"]
                ),
                "canary recovery result used a failed canary live job ID",
            )
            residual_identity = {
                "campaign_id": CAMPAIGN_ID,
                "state_root": str(root),
                "canary_token": token,
                "controller_identity_sha256": prior[
                    "controller_identity_sha256"
                ],
                "package_protocol_sha256": protocol_sha256,
                "source_sha256": identity["source_sha256"],
                "durable_prefix_sha256": prior["durable_prefix_sha256"],
                "controller_lock": controller_lock.binding(),
            }
            cancel_context = {
                "schema_version": 1,
                "status": "canary_scheduler_calling",
                "campaign_id": CAMPAIGN_ID,
                "state_root": str(root),
                "canary_token": token,
                "role": "recovery_cancel",
                "controller_lock": controller_lock.binding(),
            }
            residual_chain = submit._validated_residual_recovery_chain(
                root,
                filename_prefix="CANARY_RECOVERY_RECONCILED",
                initial_terminal_path=prior_path,
                initial_cancel_history_length=len(
                    prior["cancel_attempt_history"]
                ),
                identity=residual_identity,
                status_prefix="canary_recovered_residual",
                cancel_directory=root,
                cancel_calling_prefix="CANARY_RECOVERY_CANCEL_CALLING",
                cancel_result_prefix="CANARY_RECOVERY_CANCEL_RESULT",
                cancel_context=cancel_context,
                role_names=expected_names,
                comment=identity["scheduler_comment"],
                squeue=squeue,
                scancel=scancel,
                expected_control_plane=authorization,
                scheduler_fallback=fallback,
            )
            require(
                all(
                    not failed_job_ids.intersection(
                        generation["live_verified_job_ids"]
                    )
                    and not failed_job_ids.intersection(
                        generation["cancelled_live_job_ids"]
                    )
                    for generation in residual_chain
                ),
                "canary residual recovery used a failed canary live job ID",
            )
            bound_history_length = len(
                (
                    residual_chain[-1]["cancel_attempt_history"]
                    if residual_chain
                    else prior["cancel_attempt_history"]
                )
            )
            revalidation_observations: list[dict[str, Any]] = []
            revalidation_rounds, revalidation_active = submit._recovery_census_rounds(
                squeue=squeue,
                role_names=expected_names,
                comment=identity["scheduler_comment"],
                cwd=root,
                runner=runner,
                control_plane=control_plane,
                fallback=fallback,
                expected_observation=authorization,
                observations=revalidation_observations,
            )
            revalidation_ids = sorted(
                {
                    item
                    for values in revalidation_active.values()
                    for item in values
                },
                key=int,
            )
            require(
                not failed_job_ids.intersection(revalidation_ids),
                "canary residual recovery found a failed canary live job ID",
            )
            # Historical response/journal IDs are provenance only.  The fresh
            # exact-name/comment/owner census is the sole mutation authority,
            # including when a delayed scheduler acceptance was not present in
            # the historical response claims.
            append_residual = bool(revalidation_ids) or (
                len(existing_cancel_history) > bound_history_length
            )
            if append_residual:
                residual_history = existing_cancel_history
                residual_cancellation: Mapping[str, Any] | None = None
                residual_calling_sha256: str | None = None
                post_rounds: list[dict[str, Any]] = []
                post_active = {"wave0": [], "wave1": [], "report": []}
                if revalidation_ids:
                    residual_history, residual_cancellation = (
                        submit._append_recovery_cancel_attempt(
                            root,
                            calling_prefix="CANARY_RECOVERY_CANCEL_CALLING",
                            result_prefix="CANARY_RECOVERY_CANCEL_RESULT",
                            context=cancel_context,
                            scancel=scancel,
                            job_ids=revalidation_ids,
                            cwd=root,
                            runner=runner,
                            control_plane=control_plane,
                            fallback=fallback,
                            expected_observation=authorization,
                            observations=revalidation_observations,
                        )
                    )
                    residual_calling_sha256 = residual_history[-1][
                        "calling_sha256"
                    ]
                    post_rounds, post_active = submit._recovery_census_rounds(
                        squeue=squeue,
                        role_names=expected_names,
                        comment=identity["scheduler_comment"],
                        cwd=root,
                        runner=runner,
                        control_plane=control_plane,
                        fallback=fallback,
                        expected_observation=authorization,
                        observations=revalidation_observations,
                    )
                    require(
                        post_active
                        == {"wave0": [], "wave1": [], "report": []},
                        "residual canary jobs remain active after cancellation",
                    )
                generation = len(residual_chain)
                previous_path = (
                    root
                    / f"CANARY_RECOVERY_RECONCILED_{generation - 1:04d}.json"
                    if generation
                    else prior_path
                )
                submit.exclusive_json(
                    root
                    / f"CANARY_RECOVERY_RECONCILED_{generation:04d}.json",
                    {
                        "schema_version": 1,
                        "record": "residual_reconciliation",
                        **residual_identity,
                        "generation": generation,
                        "previous_terminal_name": previous_path.name,
                        "previous_terminal_sha256": submit.file_sha256(
                            previous_path
                        ),
                        "status": (
                            "canary_recovered_residual_terminal_after_residual_cancel"
                            if revalidation_ids
                            else "canary_recovered_residual_terminal_no_active_jobs"
                        ),
                        "live_verified_job_ids": revalidation_ids,
                        "live_verified_job_ids_by_role": revalidation_active,
                        "pre_cancel_census_rounds": revalidation_rounds,
                        "cancelled_live_job_ids": revalidation_ids,
                        "cancel_history_length_before": bound_history_length,
                        "cancel_attempt_history": residual_history,
                        "cancel_calling_sha256": residual_calling_sha256,
                        "cancellation": residual_cancellation,
                        "post_cancel_census_rounds": post_rounds,
                        "post_cancel_active_job_ids_by_role": post_active,
                        "scheduler_control_plane_observations": revalidation_observations,
                        "scheduler_calls": len(revalidation_observations),
                        "new_jobs_created": 0,
                    },
                )
                residual_chain = submit._validated_residual_recovery_chain(
                    root,
                    filename_prefix="CANARY_RECOVERY_RECONCILED",
                    initial_terminal_path=prior_path,
                    initial_cancel_history_length=len(
                        prior["cancel_attempt_history"]
                    ),
                    identity=residual_identity,
                    status_prefix="canary_recovered_residual",
                    cancel_directory=root,
                    cancel_calling_prefix="CANARY_RECOVERY_CANCEL_CALLING",
                    cancel_result_prefix="CANARY_RECOVERY_CANCEL_RESULT",
                    cancel_context=cancel_context,
                    role_names=expected_names,
                    comment=identity["scheduler_comment"],
                    squeue=squeue,
                    scancel=scancel,
                    expected_control_plane=authorization,
                    scheduler_fallback=fallback,
                )
            return {
                **prior,
                "reused_recovery": True,
                "residual_reconciliation_chain": residual_chain,
                "revalidation_census_rounds": revalidation_rounds,
                "revalidation_scheduler_control_plane_observations": revalidation_observations,
                "recovery_invocation_scheduler_calls": len(revalidation_observations),
            }
        journal_names = {
            "wave0": "CANARY_WAVE0_SUBMITTED.json",
            "wave1": "CANARY_WAVE1_SUBMITTED.json",
            "report": "CANARY_REPORT_SUBMITTED.json",
        }
        claimed: dict[str, set[str]] = {role: set() for role in expected_names}
        for role, values in aborted_ids_by_role.items():
            claimed[role].update(values)
        journal_values: dict[str, dict[str, Any]] = {}
        journal_gap = False
        for role, filename in journal_names.items():
            path = root / filename
            if not os.path.lexists(path):
                journal_gap = True
                continue
            require(not journal_gap, "canary submitted journals are not a contiguous prefix")
            submit._regular_nonsymlink(path, f"canary recovery {role} submitted journal")
            require(
                stat.S_IMODE(path.lstat().st_mode) == 0o444,
                f"canary recovery {role} submitted journal mode differs",
            )
            record = submit.read_json(path)
            expected_fields = (
                {
                    "schema_version",
                    "status",
                    "job_id",
                    "accepted_submission_record",
                    "accepted_hold",
                }
                if role == "wave0"
                else {
                    "schema_version",
                    "status",
                    "job_id",
                    "accepted_submission_record",
                    "accepted_dependency",
                }
            )
            expected_status = (
                "canary_wave0_submitted_held"
                if role == "wave0"
                else f"canary_{role}_submitted"
            )
            job_id = record.get("job_id")
            require(
                set(record) == expected_fields
                and type(record.get("schema_version")) is int
                and record.get("schema_version") == 1
                and record.get("status") == expected_status
                and isinstance(job_id, str)
                and submit.JOB_ID.fullmatch(job_id) is not None,
                f"canary recovery {role} journal differs",
            )
            claimed[role].add(job_id)
            journal_values[role] = record

        worker = _load_canary_worker(root)
        # Dependency-bearing CALLING records were created from the exact IDs in
        # the contiguous accepted-submission journal prefix.  Extra IDs retained
        # by an abort record are forensic claims only and must never influence
        # reconstruction of those scheduler commands.
        context_jobs = {
            role: (
                journal_values[role]["job_id"]
                if role in journal_values
                else str(999999990 + index)
            )
            for index, role in enumerate(("wave0", "wave1", "report"))
        }
        recovery_context = {
            "canary_token": token,
            "job_ids": context_jobs,
            "job_names": expected_names,
            "scheduler_comment": identity["scheduler_comment"],
            "scheduler_executables": {
                **scheduler_executables,
            },
            "scheduler_control_plane": identity["scheduler_control_plane"],
            "scheduler_control_plane_sha256": scheduler_control_plane_sha256,
            "controller_lock": controller_lock.binding(),
        }
        calling_names = {
            "wave0": "CANARY_WAVE0_CALLING.json",
            "wave1": "CANARY_WAVE1_CALLING.json",
            "report": "CANARY_REPORT_CALLING.json",
        }
        calling_hashes: dict[str, str] = {}
        calling_gap = False
        expected_commands = worker._expected_submission_commands(root, recovery_context)
        for role in ("wave0", "wave1", "report"):
            path = root / calling_names[role]
            if not os.path.lexists(path):
                calling_gap = True
                continue
            require(not calling_gap, "canary calling records are not a contiguous prefix")
            submit._regular_nonsymlink(path, f"canary recovery {role} calling record")
            require(
                stat.S_IMODE(path.lstat().st_mode) == 0o444,
                f"canary recovery {role} calling record mode differs",
            )
            calling = submit.read_json(path)
            require(
                submit.exact_json_equal(
                    calling,
                    {
                    "schema_version": 1,
                    "status": "canary_scheduler_calling",
                    "campaign_id": CAMPAIGN_ID,
                    "state_root": str(root),
                    "canary_token": token,
                    "role": role,
                    "job_name": expected_names[role],
                    "scheduler_comment": identity["scheduler_comment"],
                    "command": expected_commands[role],
                    "controller_lock": controller_lock.binding(),
                    },
                ),
                f"canary recovery {role} calling record differs",
            )
            calling_hashes[role] = submit.file_sha256(path)
        require(
            set(journal_values).issubset(set(calling_hashes))
            and len(calling_hashes) <= len(journal_values) + 1,
            "canary calling/submitted prefix differs",
        )
        for role, record in journal_values.items():
            worker._validate_submission_record(
                record["accepted_submission_record"],
                role=role,
                root=root,
                authorization=recovery_context,
            )
            worker._validate_scheduler_evidence(
                record[
                    "accepted_hold" if role == "wave0" else "accepted_dependency"
                ],
                role=role,
                root=root,
                authorization=recovery_context,
            )
        for filename in ("CANARY_AUTHORIZATION.json", "CANARY_SUBMISSION_RECEIPT.json"):
            path = root / filename
            if os.path.lexists(path):
                value = submit.read_json(path)
                jobs = value.get("job_ids")
                require(
                    isinstance(jobs, Mapping)
                    and set(jobs) == set(expected_names)
                    and all(
                        isinstance(item, str)
                        and submit.JOB_ID.fullmatch(item) is not None
                        for item in jobs.values()
                    ),
                    f"canary recovery {filename} job IDs differ",
                )
                for role, job_id in jobs.items():
                    claimed[role].add(job_id)
        require(
            len({item for values in claimed.values() for item in values})
            == sum(len(values) for values in claimed.values()),
            "canary recovery assigns one durable job ID to multiple roles",
        )
        require(
            not failed_job_ids.intersection(
                item for values in claimed.values() for item in values
            ),
            "canary recovery durable prefix reuses a failed canary job ID",
        )
        durable_prefix_sha256 = _validated_canary_postsubmission_prefix(
            submit,
            worker,
            root,
            identity=identity,
            protocol_sha256=protocol_sha256,
            scheduler_executables=scheduler_executables,
            scheduler_control_plane_sha256=scheduler_control_plane_sha256,
        )
        observations: list[dict[str, Any]] = []
        pre_cancel_rounds, active = submit._recovery_census_rounds(
            squeue=squeue,
            role_names=expected_names,
            comment=identity["scheduler_comment"],
            cwd=root,
            runner=runner,
            control_plane=control_plane,
            fallback=fallback,
            expected_observation=authorization,
            observations=observations,
        )
        # Durable response/journal claims remain provenance only.  Cancellation
        # targets are derived exclusively from this fresh settled census.
        active_ids = sorted(
            {job_id for values in active.values() for job_id in values}, key=int
        )
        require(
            not failed_job_ids.intersection(active_ids),
            "canary recovery found a failed canary live job ID",
        )
        cancel_context = {
            "schema_version": 1,
            "status": "canary_scheduler_calling",
            "campaign_id": CAMPAIGN_ID,
            "state_root": str(root),
            "canary_token": token,
            "role": "recovery_cancel",
            "controller_lock": controller_lock.binding(),
        }
        cancel_history = existing_cancel_history
        cancellation = None
        cancel_calling_sha256 = None
        post_cancel_rounds: list[dict[str, Any]] = []
        post_cancel_active = {"wave0": [], "wave1": [], "report": []}
        if active_ids:
            cancel_history, cancellation = submit._append_recovery_cancel_attempt(
                root,
                calling_prefix="CANARY_RECOVERY_CANCEL_CALLING",
                result_prefix="CANARY_RECOVERY_CANCEL_RESULT",
                context=cancel_context,
                scancel=scancel,
                job_ids=active_ids,
                cwd=root,
                runner=runner,
                control_plane=control_plane,
                fallback=fallback,
                expected_observation=authorization,
                observations=observations,
            )
            post_cancel_rounds, post_cancel_active = submit._recovery_census_rounds(
                squeue=squeue,
                role_names=expected_names,
                comment=identity["scheduler_comment"],
                cwd=root,
                runner=runner,
                control_plane=control_plane,
                fallback=fallback,
                expected_observation=authorization,
                observations=observations,
            )
            require(
                post_cancel_active == {"wave0": [], "wave1": [], "report": []},
                "canary jobs remain active after exact cancellation",
            )
        if cancel_history:
            cancel_calling_sha256 = cancel_history[-1]["calling_sha256"]
            cancellation = cancel_history[-1]["cancellation"]
        claimed_by_role = {
            role: sorted(values, key=int) for role, values in claimed.items()
        }
        claimed_ids = sorted(
            {item for values in claimed_by_role.values() for item in values}, key=int
        )
        result = {
            "schema_version": 1,
            "status": (
                "canary_recovered_terminal_after_cancel_attempts"
                if cancel_history
                else "canary_recovered_terminal_no_active_jobs"
            ),
            "campaign_id": CAMPAIGN_ID,
            "state_root": str(root),
            "canary_token": token,
            "controller_identity_sha256": submit.file_sha256(
                root / "CANARY_CONTROLLER_IDENTITY.json"
            ),
            "package_protocol_sha256": protocol_sha256,
            "source_sha256": identity["source_sha256"],
            "claimed_job_ids": claimed_ids,
            "claimed_job_ids_by_role": claimed_by_role,
            "live_verified_job_ids": active_ids,
            "live_verified_job_ids_by_role": active,
            "calling_intent_sha256_by_role": calling_hashes,
            "pre_cancel_census_rounds": pre_cancel_rounds,
            "cancelled_live_job_ids": active_ids,
            "cancel_calling_sha256": cancel_calling_sha256,
            "cancellation": cancellation,
            "cancel_attempt_history": cancel_history,
            "post_cancel_census_rounds": post_cancel_rounds,
            "post_cancel_active_job_ids_by_role": post_cancel_active,
            "durable_prefix_sha256": durable_prefix_sha256,
            "scheduler_control_plane_observations": observations,
            "new_jobs_created": 0,
            "scheduler_calls": len(observations),
            "controller_lock": controller_lock.binding(),
        }
        submit.exclusive_json(prior_path, result)
        validated = _validated_canary_recovery_result(
            submit,
            root,
            submit.read_json(prior_path),
            identity=identity,
            protocol_sha256=protocol_sha256,
            role_names=expected_names,
            squeue=squeue,
            scancel=scancel,
            controller_lock=controller_lock.binding(),
            scheduler_executables=scheduler_executables,
            scheduler_control_plane_sha256=scheduler_control_plane_sha256,
            expected_control_plane=authorization,
            scheduler_fallback=fallback,
        )
        return {**validated, "reused_recovery": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--describe", action="store_true", help="read-only description (default)")
    action.add_argument(
        "--submit-real-gpu-two-wave-canary",
        action="store_true",
        help="perform the explicit non-scientific three-job scheduler mutation",
    )
    action.add_argument(
        "--recover-or-cancel-real-gpu-canary",
        action="store_true",
        help="reconcile and cancel an existing hard-crashed canary without creating jobs",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--confirmation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not (
            args.submit_real_gpu_two_wave_canary
            or args.recover_or_cancel_real_gpu_canary
        ):
            require(args.state_root is None and args.confirmation is None, "--describe accepts no mutation arguments")
            print(json.dumps(description(), sort_keys=True, indent=2))
            return 0
        require(args.state_root is not None, "real GPU canary requires --state-root")
        if args.submit_real_gpu_two_wave_canary:
            value = submit_real_canary(
                args.repo_root.absolute(),
                args.state_root.absolute(),
                str(args.confirmation),
            )
        else:
            value = recover_or_cancel_real_canary(
                args.repo_root.absolute(),
                args.state_root.absolute(),
                str(args.confirmation),
            )
        print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"Exp23 canary controller error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
