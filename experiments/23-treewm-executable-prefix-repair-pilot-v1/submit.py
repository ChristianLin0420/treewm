#!/usr/bin/env python3
"""Read-only preflight, or explicitly submit, the sealed twenty-cell Exp23 pilot.

The default action is the same as ``--test-only``.  Submission is deliberately a
separate, explicit mode.  No preflight path creates a directory, a bytecode file, a
snapshot, or a scheduler process.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence


sys.dont_write_bytecode = True

PACKAGE_RELATIVE = Path("experiments/23-treewm-executable-prefix-repair-pilot-v1")
CAMPAIGN_ID = "treewm-executable-prefix-repair-pilot-v1"
PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
AUDITS = (
    ("weight", "weight_audit.py", ("--summary-only",), "EXP23_WEIGHT_AUDIT_SUMMARY="),
    ("prefix_target", "prefix_target_audit.py", (), "EXP23_PREFIX_TARGET_AUDIT="),
    ("resolved_config", "resolved_config_audit.py", (), "EXP23_RESOLVED_CONFIG_AUDIT="),
    ("causal_parity", "causal_parity_audit.py", (), "EXP23_CAUSAL_PARITY_AUDIT="),
)
AUDIT_LOCKS = {
    "weight": "weight_audit.lock.json",
    "prefix_target": "prefix_target.lock.json",
    "resolved_config": "resolved_config.lock.json",
    "causal_parity": "causal_parity.lock.json",
}
SBATCH_JOB = re.compile(r"^(?P<job_id>[0-9]+)$")
JOB_ID = re.compile(r"^[1-9][0-9]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA1 = re.compile(r"^[0-9a-f]{40}$")
RECEIPT_FIELDS = frozenset(
    {
        "schema_version", "status", "campaign_id", "submission_root",
        "snapshot_root", "submission_sha256", "train_array_job_id",
        "report_job_id", "array", "dependency",
    }
)
FORBIDDEN_DISTRIBUTED_ENVIRONMENT = frozenset(
    {"RANK", "WORLD_SIZE", "LOCAL_RANK", "TREEWM_STOP_AFTER_UPDATE"}
)
SAFE_CHILD_ENVIRONMENT = frozenset(
    {
        "PATH",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "LANG",
        "LC_ALL",
        "TZ",
        "TMPDIR",
        "CUDA_VISIBLE_DEVICES",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
    }
)

# Stdlib-only child bootstrap. ``-I -S`` prevents sitecustomize and every ``.pth``
# file from running; the two hash-bound distribution roots are appended directly.
ISOLATED_RUN_CODE = r"""
import hashlib, json, os, runpy, stat, sys
root, vsite, bsite, expected_sha, expected_size, expected_python, inventory_json, script = sys.argv[1:9]
if not (sys.flags.isolated and sys.flags.no_site): raise SystemExit('isolation flags absent')
if os.path.normpath(os.path.abspath(sys.executable)) != expected_python: raise SystemExit('lexical interpreter differs')
target = os.path.realpath(sys.executable)
st = os.stat(target)
if st.st_size != int(expected_size): raise SystemExit('interpreter size differs')
h = hashlib.sha256()
with open(target, 'rb') as fh:
    for block in iter(lambda: fh.read(16 * 1024 * 1024), b''): h.update(block)
if h.hexdigest() != expected_sha: raise SystemExit('interpreter hash differs')
if any('site-packages' in value for value in sys.path): raise SystemExit('site path loaded before bootstrap')
for value in (root, vsite, bsite):
    if not os.path.isdir(value) or os.path.islink(value): raise SystemExit('bootstrap root unavailable')
if inventory_json:
    inventory = json.loads(inventory_json)
    if not isinstance(inventory, dict) or not inventory: raise SystemExit('snapshot inventory absent')
    expected_files = set()
    expected_dirs = set()
    for relative, digest in inventory.items():
        parts = relative.split('/')
        if not parts or any(part in ('', '.', '..') for part in parts): raise SystemExit('unsafe inventory path')
        if not isinstance(digest, str) or len(digest) != 64: raise SystemExit('invalid inventory hash')
        expected_files.add(relative)
        expected_dirs.update('/'.join(parts[:index]) for index in range(1, len(parts)))
    if os.stat(root).st_mode & 0o222: raise SystemExit('snapshot root writable')
    actual_files, actual_dirs = set(), set()
    for directory, names, files in os.walk(root, followlinks=False):
        for name in names:
            path = os.path.join(directory, name)
            info = os.lstat(path)
            relative = os.path.relpath(path, root)
            if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o222: raise SystemExit('unsafe snapshot directory')
            actual_dirs.add(relative)
        for name in files:
            path = os.path.join(directory, name)
            info = os.lstat(path)
            relative = os.path.relpath(path, root)
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222: raise SystemExit('unsafe snapshot file')
            if relative not in inventory: raise SystemExit('unclaimed snapshot file')
            digest = hashlib.sha256()
            with open(path, 'rb') as fh:
                for block in iter(lambda: fh.read(16 * 1024 * 1024), b''): digest.update(block)
            if digest.hexdigest() != inventory[relative]: raise SystemExit('snapshot file hash differs')
            actual_files.add(relative)
    if actual_files != expected_files or actual_dirs != expected_dirs: raise SystemExit('snapshot coverage differs')
sys.path.insert(0, root)
sys.path.extend((vsite, bsite))
if os.path.islink(script) or not stat.S_ISREG(os.stat(script).st_mode): raise SystemExit('target script invalid')
if os.path.commonpath((os.path.realpath(script), os.path.realpath(root))) != os.path.realpath(root): raise SystemExit('target script escapes root')
sys.argv = [script, *sys.argv[9:]]
runpy.run_path(script, run_name='__main__')
"""

ISOLATED_AUDIT_CODE = r"""
import os, subprocess, sys
root, vsite, bsite, expected_sha, expected_size, expected_python, inventory_json, basic, script = sys.argv[1:10]
real_run = subprocess.run
def isolated_run(command, *args, **kwargs):
    values = list(command)
    if values and os.path.normpath(os.path.abspath(str(values[0]))) == expected_python:
        if len(values) < 2: raise RuntimeError('Python child has no script')
        values = [values[0], '-I', '-S', '-B', '-c', basic, root, vsite, bsite,
                  expected_sha, expected_size, expected_python, inventory_json, str(values[1]), *map(str, values[2:])]
    return real_run(values, *args, **kwargs)
subprocess.run = isolated_run
sys.argv = [sys.argv[0], root, vsite, bsite, expected_sha, expected_size, expected_python, inventory_json, script, *sys.argv[10:]]
exec(compile(basic, '<treewm-isolated-bootstrap>', 'exec'))
"""


class SubmissionError(RuntimeError):
    """The prospective launch does not satisfy the sealed contract."""


class SchedulerSubmissionError(SubmissionError):
    def __init__(
        self,
        message: str,
        job_ids: Sequence[str] = (),
        job_ids_by_role: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        super().__init__(message)
        self.job_ids = tuple(job_ids)
        self.job_ids_by_role = {
            role: tuple(values)
            for role, values in (job_ids_by_role or {}).items()
        }


class CommitRecoveryRequired(SchedulerSubmissionError):
    """Both jobs and READY are durable; retry may only commit the receipt."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SubmissionError(message)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    SubmissionError(f"non-finite JSON value in {source}: {token}")
                ),
            )
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionError(f"cannot read JSON {source}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {source}")
    return value


def _regular_nonsymlink(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise SubmissionError(f"{label} is unavailable: {exc}") from exc
    require(stat.S_ISREG(info.st_mode), f"{label} is not a regular nonsymlink file")
    return info


def _directory_nonsymlink(path: Path, label: str) -> Path:
    lexical = path.absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            component = current.lstat()
        except OSError as exc:
            raise SubmissionError(f"{label} path component is unavailable: {current}: {exc}") from exc
        require(not stat.S_ISLNK(component.st_mode), f"{label} has a symlink path component: {current}")
    try:
        info = path.lstat()
    except OSError as exc:
        raise SubmissionError(f"{label} is unavailable: {exc}") from exc
    require(stat.S_ISDIR(info.st_mode), f"{label} is not a nonsymlink directory")
    return path.resolve(strict=True)


def _safe_relative(value: str | Path, label: str) -> Path:
    relative = Path(value)
    require(
        not relative.is_absolute()
        and relative.parts
        and all(part not in ("", ".", "..") for part in relative.parts),
        f"{label} is not a safe repository-relative path: {value}",
    )
    return relative


def _contained_regular_no_symlinks(root: Path, relative: Path, label: str) -> Path:
    """Reject a symlink at every path component, not merely at the leaf."""

    root_resolved = _directory_nonsymlink(root, f"{label} root")
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise SubmissionError(f"{label} is unavailable: {exc}") from exc
        if index + 1 == len(relative.parts):
            require(stat.S_ISREG(info.st_mode), f"{label} is not a regular nonsymlink file")
        else:
            require(stat.S_ISDIR(info.st_mode), f"{label} has a symlink/non-directory parent: {current}")
    resolved = current.resolve(strict=True)
    require(resolved.is_relative_to(root_resolved), f"{label} escapes repository containment")
    return current


def _open_relative_regular(root: Path, relative: Path, label: str) -> tuple[int, os.stat_result]:
    """Open a repository file through O_NOFOLLOW directory descriptors."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow
    descriptors: list[int] = []
    try:
        descriptor = os.open(root, directory_flags)
        descriptors.append(descriptor)
        for part in relative.parts[:-1]:
            descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            descriptors.append(descriptor)
        source_fd = os.open(relative.name, os.O_RDONLY | nofollow, dir_fd=descriptor)
        info = os.fstat(source_fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(source_fd)
            raise SubmissionError(f"{label} is not a regular file")
        return source_fd, info
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _load_module(name: str, path: Path, *, containment_root: Path) -> ModuleType:
    _regular_nonsymlink(path, name)
    resolved = path.resolve(strict=True)
    root = containment_root.resolve(strict=True)
    require(resolved.is_relative_to(root), f"{name} escapes its repository root")
    unique = f"_treewm_exp23_{name}_{os.getpid()}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(unique, resolved)
    require(spec is not None and spec.loader is not None, f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique] = module
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(unique, None)
        raise
    module_path = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    require(module_path == resolved, f"{name} import escaped containment")
    return module


def load_campaign(repo_root: str | Path) -> ModuleType:
    root = Path(repo_root).resolve(strict=True)
    return _load_module(
        "campaign", root / PACKAGE_RELATIVE / "campaign.py", containment_root=root
    )


def reject_inherited_environment(environ: Mapping[str, str] | None = None) -> None:
    """Reject every externally supplied TreeWM/distributed control variable.

    The scheduler receives ``--export=NONE`` and workers construct their own exact
    environment from a launch JSON.  Consequently the submitter has no legitimate
    inherited ``TREEWM_*`` variable.
    """

    environment = os.environ if environ is None else environ
    forbidden = sorted(FORBIDDEN_DISTRIBUTED_ENVIRONMENT.intersection(environment))
    unknown_treewm = sorted(
        key for key in environment if key.startswith("TREEWM_")
    )
    failures = sorted(set(forbidden + unknown_treewm))
    require(not failures, "forbidden inherited environment: " + ", ".join(failures))


def verify_submit_interpreter(
    manifest: Mapping[str, Any], executable: str | Path | None = None
) -> str:
    """Bind the lexical venv entry point, its target, and isolated runtime sites.

    The sealed interpreter is intentionally a venv symlink.  Resolving only the leaf
    is insufficient: invoking its base target directly has different site-package
    semantics even though both paths resolve to the same executable.
    """

    expected = Path(str(manifest["paths"]["python"]))
    actual = Path(executable or sys.executable)
    require(expected.is_absolute(), "pinned Python path is not absolute")
    try:
        expected_info = expected.lstat()
    except OSError as exc:
        raise SubmissionError(f"pinned Python is unavailable: {exc}") from exc
    require(
        stat.S_ISLNK(expected_info.st_mode) or stat.S_ISREG(expected_info.st_mode),
        "pinned Python is neither a regular file nor a symlink",
    )
    target = expected.resolve(strict=True)
    _regular_nonsymlink(target, "resolved pinned Python")
    require(os.access(expected, os.X_OK), "pinned Python is not executable")
    require(
        os.path.normpath(os.path.abspath(str(actual))) == str(expected),
        f"submit must use the exact lexical pinned Python {expected}; actual is {actual}",
    )
    require(
        actual.resolve(strict=True) == target,
        "submit interpreter target differs from the sealed venv target",
    )
    return str(expected)


def interpreter_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    python = Path(verify_submit_interpreter(manifest))
    venv_root = python.parent.parent
    pyvenv = venv_root / "pyvenv.cfg"
    _regular_nonsymlink(pyvenv, "pinned pyvenv.cfg")
    values: dict[str, str] = {}
    for line in pyvenv.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    require("home" in values and Path(values["home"]).is_absolute(), "pyvenv home is absent")
    base_root = Path(values["home"]).parent
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    venv_site = venv_root / "lib" / version / "site-packages"
    base_site = base_root / "lib" / version / "site-packages"
    for path, label in ((venv_site, "venv site-packages"), (base_site, "base site-packages")):
        try:
            site_info = path.lstat()
        except OSError as exc:
            raise SubmissionError(f"{label} is unavailable: {exc}") from exc
        require(stat.S_ISDIR(site_info.st_mode), f"{label} is not a nonsymlink directory")
    target = python.resolve(strict=True)
    info = target.stat()
    base_executable = Path(str(getattr(sys, "_base_executable", target)))
    require(base_executable.resolve(strict=True) == target, "base interpreter target differs")
    return {
        "lexical_executable": str(python),
        "lexical_symlink_target": os.readlink(python) if python.is_symlink() else None,
        "resolved_executable": str(target),
        "resolved_executable_sha256": file_sha256(target),
        "resolved_executable_size": info.st_size,
        "base_executable": str(base_executable),
        "venv_site_packages": str(venv_site),
        "base_site_packages": str(base_site),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


def activate_isolated_runtime(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Append only the two bound package roots; never process ``.pth`` files."""

    require(bool(sys.flags.isolated) and bool(sys.flags.no_site), "orchestration requires Python -I -S")
    identity = interpreter_contract(manifest)
    site_paths = (identity["venv_site_packages"], identity["base_site_packages"])
    existing_sites = [item for item in sys.path if "site-packages" in item]
    require(not existing_sites or existing_sites == list(site_paths), "unexpected site-package path before bootstrap")
    for value in site_paths:
        if value not in sys.path:
            sys.path.append(value)
    require(
        [item for item in sys.path if "site-packages" in item] == list(site_paths),
        "isolated runtime site-package order differs",
    )
    return identity


def isolated_python_command(
    argv: Sequence[str],
    repo_root: Path,
    identity: Mapping[str, Any],
    *,
    intercept_python_children: bool,
    snapshot_inventory: Mapping[str, str] | None = None,
) -> list[str]:
    require(len(argv) >= 2, "isolated Python command has no target script")
    python = str(identity["lexical_executable"])
    require(str(argv[0]) == python, "isolated child interpreter differs")
    prefix = [
        python,
        "-I",
        "-S",
        "-B",
        "-c",
        ISOLATED_AUDIT_CODE if intercept_python_children else ISOLATED_RUN_CODE,
        str(repo_root),
        str(identity["venv_site_packages"]),
        str(identity["base_site_packages"]),
        str(identity["resolved_executable_sha256"]),
        str(identity["resolved_executable_size"]),
        python,
        canonical_json(snapshot_inventory) if snapshot_inventory is not None else "",
    ]
    if intercept_python_children:
        prefix.append(ISOLATED_RUN_CODE)
    return [*prefix, str(argv[1]), *map(str, argv[2:])]


def _git_command(root: Path, *arguments: str) -> str:
    git = Path("/usr/bin/git")
    _regular_nonsymlink(git, "pinned git")
    require(os.access(git, os.X_OK), "pinned git is not executable")
    completed = subprocess.run(
        [
            str(git),
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            *arguments,
        ],
        cwd=root,
        env={
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "LC_ALL": "C",
        },
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(completed.returncode == 0, f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def git_provenance(repo_root: Path) -> dict[str, Any]:
    """Require an immutable, clean, pushed source provenance before mutation."""

    root = _directory_nonsymlink(repo_root, "repository root")
    head = _git_command(root, "rev-parse", "HEAD")
    origin_main = _git_command(root, "rev-parse", "origin/main")
    status = _git_command(root, "status", "--porcelain=v1", "--untracked-files=all")
    branch = _git_command(root, "rev-parse", "--abbrev-ref", "HEAD")
    remote = _git_command(root, "config", "--get", "remote.origin.url")
    object_format = _git_command(root, "rev-parse", "--show-object-format")
    oid_pattern = GIT_SHA1 if object_format == "sha1" else SHA256 if object_format == "sha256" else None
    require(oid_pattern is not None, f"unsupported git object format: {object_format}")
    require(oid_pattern.fullmatch(head) is not None, "git HEAD is malformed")
    require(oid_pattern.fullmatch(origin_main) is not None, "git origin/main is malformed")
    require(not status, "explicit submission requires a clean worktree including untracked files")
    require(head == origin_main, "explicit submission requires HEAD == origin/main")
    return {
        "head": head,
        "origin_main": origin_main,
        "branch": branch,
        "remote_origin": remote,
        "object_format": object_format,
        "worktree_status": "clean",
        "worktree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _child_environment(
    extra: Mapping[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = os.environ if environ is None else environ
    result = {key: source[key] for key in SAFE_CHILD_ENVIRONMENT if key in source}
    result.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            # Audits are read-only.  Do not let optional libraries populate user or
            # repository caches as a side effect of import.
            # /proc/self/fd is reported as a writable directory (so Matplotlib does
            # not create a temporary fallback) but cannot retain named cache files.
            "MPLCONFIGDIR": "/proc/self/fd",
            "XDG_CACHE_HOME": "/proc/self/fd",
            "TORCH_HOME": "/proc/self/fd",
            "HF_HOME": "/proc/self/fd",
            "WANDB_CACHE_DIR": "/proc/self/fd",
            "WANDB_CONFIG_DIR": "/proc/self/fd",
        }
    )
    if extra:
        for key, value in extra.items():
            require(key != "TREEWM_STOP_AFTER_UPDATE", "staged stop is forbidden")
            result[str(key)] = str(value)
    return result


def _output_tree_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Hash every entry and byte beneath the prospective output root."""
    run_root = Path(str(manifest["paths"]["run_root"]))
    rows: list[dict[str, Any]] = []
    if not os.path.lexists(run_root):
        return stable_hash({"exists": False, "entries": []})
    root = _directory_nonsymlink(run_root, "run root")
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        relative = str(path.relative_to(root))
        require(not stat.S_ISLNK(info.st_mode), f"output tree contains symlink: {relative}")
        if stat.S_ISREG(info.st_mode):
            rows.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": stat.S_IMODE(info.st_mode),
                    "size": info.st_size,
                    "mtime_ns": info.st_mtime_ns,
                    "sha256": file_sha256(path),
                }
            )
        elif stat.S_ISDIR(info.st_mode):
            rows.append(
                {
                    "path": relative,
                    "kind": "directory",
                    "mode": stat.S_IMODE(info.st_mode),
                    "mtime_ns": info.st_mtime_ns,
                }
            )
        else:
            raise SubmissionError(f"output tree contains special file: {relative}")
    return stable_hash(rows)


def _scientific_output_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Hash only cell outputs, allowing intentional submission state creation."""

    run_root = Path(str(manifest["paths"]["run_root"]))
    rows: list[dict[str, Any]] = []
    for setting in manifest["design"]["settings"]:
        path = run_root / str(setting)
        if not os.path.lexists(path):
            rows.append({"setting": setting, "missing": True})
            continue
        _directory_nonsymlink(path, f"scientific setting output {setting}")
        for child in sorted(path.rglob("*")):
            info = child.lstat()
            require(not stat.S_ISLNK(info.st_mode), f"scientific output contains symlink: {child}")
            if stat.S_ISREG(info.st_mode):
                rows.append({"path": str(child.relative_to(run_root)), "sha256": file_sha256(child), "size": info.st_size})
            elif stat.S_ISDIR(info.st_mode):
                rows.append({"path": str(child.relative_to(run_root)), "kind": "directory"})
            else:
                raise SubmissionError(f"scientific output contains special file: {child}")
    return stable_hash(rows)


def _namespace_is_fresh(manifest: Mapping[str, Any], submission_root: Path) -> bool:
    run_root = Path(str(manifest["paths"]["run_root"]))
    if submission_root.exists():
        return False
    if not run_root.exists():
        return True
    if run_root.is_symlink() or not run_root.is_dir():
        return False
    # An empty run root is harmless.  Any prior scientific setting, terminal marker,
    # receipt, or cancellation latch makes this prospective namespace non-fresh.
    return next(run_root.iterdir(), None) is None


def _parse_audit_stdout(raw: bytes, prefix: str, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SubmissionError(f"{label} audit stdout is not UTF-8") from exc
    matches = [line[len(prefix) :] for line in text.splitlines() if line.startswith(prefix)]
    require(len(matches) == 1, f"{label} audit emitted an ambiguous result")
    try:
        value = json.loads(
            matches[0],
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SubmissionError(f"{label} audit emitted non-finite JSON: {token}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise SubmissionError(f"{label} audit result is invalid JSON: {exc}") from exc
    require(isinstance(value, dict), f"{label} audit result is not an object")
    return value


def _verify_audit_result(
    label: str,
    result: Mapping[str, Any],
    lock: Mapping[str, Any],
    manifest: Mapping[str, Any],
    stdout: bytes,
) -> dict[str, Any]:
    if label == "weight":
        identity = lock["result_identity"]
        require(result.get("artifact_sha256") == identity["artifact_sha256"], "weight audit artifact differs")
        require(result.get("rows_sha256") == identity["rows_sha256"], "weight audit rows differ")
        require(result.get("summary_sha256") == identity["summary_sha256"], "weight audit summary differs")
        require(result.get("row_count") == identity["row_count"] == 40, "weight audit row count differs")
        artifact = str(result["artifact_sha256"])
    else:
        require(dict(result) == dict(lock), f"{label} audit replay differs byte-semantically from its lock")
        artifact = str(result.get("artifact_sha256", ""))
    require(SHA256.fullmatch(artifact) is not None, f"{label} audit artifact is malformed")
    if label == "causal_parity":
        binding = manifest["causal_parity_contract"]
        require(hashlib.sha256(stdout).hexdigest() == binding["stdout_sha256"], "causal audit stdout hash differs")
        require(len(stdout) == int(binding["stdout_bytes"]), "causal audit stdout byte count differs")
    return {
        "artifact_sha256": artifact,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stdout_bytes": len(stdout),
    }


AuditRunner = Callable[[Sequence[str], Path, Mapping[str, str], float], subprocess.CompletedProcess[bytes]]


def _default_runner(
    command: Sequence[str], cwd: Path, environment: Mapping[str, str], timeout: float
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SubmissionError(f"preflight command timed out: {command[1]}") from exc


def rerun_audit_locks(
    repo_root: Path,
    campaign: ModuleType,
    manifest: Mapping[str, Any],
    *,
    runner: AuditRunner = _default_runner,
    timeout: float = 7_200,
    snapshot_inventory: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    package = repo_root / PACKAGE_RELATIVE
    records: dict[str, Any] = {}
    results: dict[str, Any] = {}
    python = str(manifest["paths"]["python"])
    try:
        python_info = Path(python).lstat()
    except OSError as exc:
        raise SubmissionError(f"pinned Python is unavailable: {exc}") from exc
    require(
        stat.S_ISREG(python_info.st_mode) or stat.S_ISLNK(python_info.st_mode),
        "pinned Python is not a regular file or sealed symlink",
    )
    _regular_nonsymlink(Path(python).resolve(strict=True), "resolved pinned Python")
    require(os.access(python, os.X_OK), "pinned Python is not executable")
    identity = interpreter_contract(manifest)
    environment = _child_environment()
    for label, program, arguments, prefix in AUDITS:
        lock = read_json(package / AUDIT_LOCKS[label])
        raw_command = [python, str(package / program), *arguments]
        if label in {"weight", "prefix_target"}:
            raw_command.extend(["--project-root", str(repo_root)])
        command = isolated_python_command(
            raw_command,
            repo_root,
            identity,
            intercept_python_children=True,
            snapshot_inventory=snapshot_inventory,
        )
        completed = runner(command, repo_root, environment, timeout)
        require(
            completed.returncode == 0,
            f"{label} audit replay failed ({completed.returncode}): "
            + completed.stderr.decode("utf-8", "replace")[-4000:],
        )
        value = _parse_audit_stdout(completed.stdout, prefix, label)
        records[label] = _verify_audit_result(label, value, lock, manifest, completed.stdout)
        results[label] = value
    return records, results


def _compose_one(
    campaign: ModuleType,
    manifest: Mapping[str, Any],
    weight_lock: Mapping[str, Any],
    cell: Any,
    root: Path,
    protocol: str,
    expected: Mapping[str, Any],
    interpreter: Mapping[str, Any],
    snapshot_inventory: Mapping[str, str] | None,
    runner: AuditRunner,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    launch = campaign.trainer_command(
        manifest,
        weight_lock,
        cell,
        repo_root=root,
        package_protocol_sha256=protocol,
        verify_recipe_files=True,
    )
    argv = launch.get("argv")
    require(isinstance(argv, list) and len(argv) >= 3, f"cell{cell.index}: trainer argv is invalid")
    require(argv[0] == manifest["paths"]["python"], f"cell{cell.index}: interpreter differs")
    require(Path(str(argv[1])).resolve(strict=True) == root / "scripts/train.py", f"cell{cell.index}: trainer path escapes snapshot")
    require("resume=auto" in argv and "train.steps=25000" in argv, f"cell{cell.index}: scratch-to-25k contract differs")
    require(not any("TREEWM_STOP_AFTER_UPDATE" in str(item) for item in argv), f"cell{cell.index}: staged stop leaked into argv")
    environment = _child_environment(
        {str(key): str(value) for key, value in launch["environment"].items()}
    )
    composition_command = isolated_python_command(
        [*argv, "--cfg", "job", "--resolve"],
        root,
        interpreter,
        intercept_python_children=False,
        snapshot_inventory=snapshot_inventory,
    )
    completed = runner(composition_command, root, environment, timeout)
    require(
        completed.returncode == 0,
        f"cell{cell.index}: direct Hydra composition failed ({completed.returncode}): "
        + completed.stderr.decode("utf-8", "replace")[-4000:],
    )
    try:
        from omegaconf import OmegaConf

        config = OmegaConf.to_container(
            OmegaConf.create(completed.stdout.decode("utf-8")), resolve=True
        )
    except Exception as exc:
        raise SubmissionError(f"cell{cell.index}: direct Hydra stdout is invalid: {exc}") from exc
    require(isinstance(config, dict), f"cell{cell.index}: resolved config is not an object")
    require(config == expected["resolved_config"], f"cell{cell.index}: resolved config differs from frozen audit")
    require(campaign.stable_hash(config) == expected["resolved_config_sha256"], f"cell{cell.index}: resolved config hash differs")
    required_hashes = {
        "manifest_sha256",
        "source_sha256",
        "runtime_sha256",
        "package_protocol_sha256",
        "config_override_sha256",
        "run_protocol_sha256",
        "input_contract_sha256",
        "data_manifest_sha256",
        "train_manifest_sha256",
        "validation_manifest_sha256",
        "calibration_sha256",
        "future_recipe_sha256",
        "weight_audit_artifact_sha256",
        "prefix_target_artifact_sha256",
        "resolved_config_artifact_sha256",
        "causal_parity_artifact_sha256",
        "actual_final_evaluation_rows_sha256",
    }
    require(required_hashes.issubset(launch["hashes"]), f"cell{cell.index}: launch omits a required binding")
    require(all(SHA256.fullmatch(str(launch["hashes"][key])) for key in required_hashes), f"cell{cell.index}: launch binding is malformed")
    require(launch.get("launch_sha256") == campaign.stable_hash({key: value for key, value in launch.items() if key != "launch_sha256"}), f"cell{cell.index}: launch hash differs")
    return launch, {
        "index": int(cell.index),
        "resolved_config_sha256": expected["resolved_config_sha256"],
        "launch_sha256": launch["launch_sha256"],
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
    }


def direct_hydra_matrix(
    repo_root: Path,
    campaign: ModuleType,
    manifest: Mapping[str, Any],
    weight_lock: Mapping[str, Any],
    protocol: str,
    resolved_lock: Mapping[str, Any],
    *,
    runner: AuditRunner = _default_runner,
    timeout: float = 300,
    workers: int = 4,
    snapshot_inventory: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells = campaign.expand_matrix(manifest)
    expected_rows = resolved_lock.get("matrix")
    require(
        len(cells) == 20
        and isinstance(expected_rows, list)
        and len(expected_rows) == 20
        and [row.get("index") for row in expected_rows] == list(range(20)),
        "resolved-config matrix is incomplete",
    )
    launches: list[dict[str, Any] | None] = [None] * 20
    records: list[dict[str, Any] | None] = [None] * 20
    interpreter = interpreter_contract(manifest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _compose_one,
                campaign,
                manifest,
                weight_lock,
                cell,
                repo_root,
                protocol,
                expected_rows[cell.index],
                interpreter,
                snapshot_inventory,
                runner,
                timeout,
            ): cell.index
            for cell in cells
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            launch, record = future.result()
            launches[index] = launch
            records[index] = record
    require(all(value is not None for value in launches), "direct Hydra launch matrix is incomplete")
    concrete_launches = [value for value in launches if value is not None]
    concrete_records = [value for value in records if value is not None]
    require(len({row["launch_sha256"] for row in concrete_launches}) == 20, "launch identities are not unique")
    return concrete_launches, concrete_records


def static_preflight(
    repo_root: Path,
    submission_root: Path,
    *,
    rerun_audits: bool = True,
    runner: AuditRunner = _default_runner,
) -> dict[str, Any]:
    reject_inherited_environment()
    root = _directory_nonsymlink(repo_root, "repository root")
    bootstrap_manifest = read_json(root / PACKAGE_RELATIVE / "manifest.json")
    runtime_interpreter = activate_isolated_runtime(bootstrap_manifest)
    require(
        not any(name == "treewm" or name.startswith("treewm.") for name in sys.modules),
        "treewm was imported before repository containment was established",
    )
    campaign = load_campaign(root)
    manifest = campaign.read_json(root / PACKAGE_RELATIVE / "manifest.json")
    require(manifest == bootstrap_manifest, "manifest changed during isolated bootstrap")
    weight_lock = campaign.read_json(root / PACKAGE_RELATIVE / "weight_audit.lock.json")
    campaign.validate_manifest(manifest, weight_lock, root)
    protocol = campaign.verify_protocol_lock(root / PACKAGE_RELATIVE)
    source_before = campaign.source_contract(root)
    require(source_before["source_sha256"] == manifest["core_binding"]["trainer_code_fingerprint"], "source fingerprint differs")
    output_before = _output_tree_fingerprint(manifest)
    scientific_before = _scientific_output_fingerprint(manifest)
    require(_namespace_is_fresh(manifest, submission_root), "prospective output namespace is not fresh")
    if rerun_audits:
        audit_records, audit_results = rerun_audit_locks(
            root, campaign, manifest, runner=runner
        )
        resolved_lock = audit_results["resolved_config"]
    else:
        audit_records = {"status": "not_rerun_in_isolated_unit_test"}
        resolved_lock = campaign.read_json(root / PACKAGE_RELATIVE / "resolved_config.lock.json")
    launches, composition = direct_hydra_matrix(
        root,
        campaign,
        manifest,
        weight_lock,
        protocol,
        resolved_lock,
        runner=runner,
    )
    source_after = campaign.source_contract(root)
    protocol_after = campaign.verify_protocol_lock(root / PACKAGE_RELATIVE)
    output_after = _output_tree_fingerprint(manifest)
    scientific_after = _scientific_output_fingerprint(manifest)
    require(source_after == source_before, "trainer source/runtime changed during preflight")
    require(protocol_after == protocol, "package protocol changed during preflight")
    require(output_after == output_before, "preflight changed the prospective scientific output tree")
    require(scientific_after == scientific_before, "preflight changed prospective cell outputs")
    require(_namespace_is_fresh(manifest, submission_root), "preflight changed the prospective output namespace")
    return {
        "schema_version": 1,
        "status": (
            "read_only_preflight_verified"
            if rerun_audits
            else "isolated_unit_test_preflight_unverified"
        ),
        "campaign_id": manifest["campaign_id"],
        "manifest": manifest,
        "weight_lock": weight_lock,
        "package_protocol_sha256": protocol,
        "source_contract": source_before,
        "orchestration_interpreter": runtime_interpreter,
        "audit_replays": audit_records,
        "direct_hydra_compositions": composition,
        "launches": launches,
        "full_output_fingerprint_before": output_before,
        "full_output_fingerprint_after": output_after,
        "scientific_output_fingerprint_before": scientific_before,
        "scientific_output_fingerprint_after": scientific_after,
        "namespace_fresh": True,
        "writes_performed": 0,
        "scheduler_calls": 0,
    }


def snapshot_inventory(
    repo_root: Path, campaign: ModuleType, protocol: str
) -> dict[str, str]:
    root = repo_root.resolve(strict=True)
    source = campaign.source_contract(root)
    inventory: dict[str, str] = {}
    for raw_relative, claimed in source["source_files"].items():
        relative = _safe_relative(raw_relative, "trainer fingerprint path")
        path = _contained_regular_no_symlinks(root, relative, f"trainer source {relative}")
        actual = file_sha256(path)
        require(actual == claimed, f"trainer fingerprint byte drift: {relative}")
        inventory[str(relative)] = actual
    package = root / PACKAGE_RELATIVE
    require(campaign.protocol_sha256(package) == protocol, "protocol changed while inventorying snapshot")
    for raw_relative in campaign.PROTOCOL_FILES:
        relative = PACKAGE_RELATIVE / _safe_relative(raw_relative, "protocol path")
        path = _contained_regular_no_symlinks(root, relative, f"protocol file {relative}")
        digest = file_sha256(path)
        prior = inventory.get(str(relative))
        require(prior in (None, digest), f"snapshot union has conflicting hashes: {relative}")
        inventory[str(relative)] = digest
    supplemental = getattr(campaign, "SNAPSHOT_IMPORT_FILES", {})
    if supplemental:
        require(isinstance(supplemental, Mapping), "SNAPSHOT_IMPORT_FILES must map paths to exact SHA256 values")
        for raw_relative, claimed in supplemental.items():
            require(SHA256.fullmatch(str(claimed)) is not None, "supplemental snapshot SHA256 is malformed")
            relative = _safe_relative(raw_relative, "supplemental snapshot import path")
            path = _contained_regular_no_symlinks(root, relative, f"supplemental import file {relative}")
            digest = file_sha256(path)
            require(digest == claimed, f"supplemental snapshot byte drift: {relative}")
            prior = inventory.get(str(relative))
            require(prior in (None, digest), f"snapshot union has conflicting hashes: {relative}")
            inventory[str(relative)] = digest
    lock_relative = PACKAGE_RELATIVE / "protocol.sha256"
    _contained_regular_no_symlinks(root, lock_relative, "protocol lock")
    inventory[str(lock_relative)] = file_sha256(root / lock_relative)
    return dict(sorted(inventory.items()))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _transaction_lock_path(submission_root: Path) -> Path:
    absolute = submission_root.absolute()
    require(
        absolute.name == "submission" and absolute.parent.name == "state",
        "transaction lock requires the sealed <run_root>/state/submission layout",
    )
    token = hashlib.sha256(str(absolute).encode("utf-8")).hexdigest()[:16]
    # The run root is intentionally absent before a fresh submission, so the lock
    # lives in its already-existing parent rather than creating scientific output.
    return absolute.parents[2] / f".exp23-{token}.transaction.lock"


class _TransactionLock:
    """Process-lifetime exclusion between a live submitter and crash recovery.

    The lock inode is deliberately persistent and outside the prospective run root.
    Unlinking a lock file after release would let two callers lock different inodes.
    ``flock`` itself is released automatically if the owning process dies, which is
    the only evidence recovery uses to distinguish a crash from a paused submitter.
    """

    def __init__(self, submission_root: Path) -> None:
        self.path = _transaction_lock_path(submission_root)
        self.descriptor: int | None = None

    def __enter__(self) -> "_TransactionLock":
        parent = _directory_nonsymlink(self.path.parent, "transaction-lock parent")
        require(parent == self.path.parent.absolute(), "transaction-lock parent changed")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise SubmissionError(f"cannot open transaction lock: {exc}") from exc
        try:
            opened = os.fstat(descriptor)
            named = self.path.lstat()
            require(stat.S_ISREG(opened.st_mode), "transaction lock is not a regular file")
            require((opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino), "transaction lock path raced")
            require(opened.st_uid == os.getuid() and opened.st_nlink == 1, "transaction lock ownership differs")
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            _fsync_directory(self.path.parent)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SubmissionError(
                    "submission transaction is active; recovery is forbidden until its owner exits"
                ) from exc
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


def _mkdir_chain_no_symlinks(path: Path, *, leaf_mode: int = 0o700) -> Path:
    destination = path.absolute()
    require(not os.path.lexists(destination), f"directory claim already exists: {destination}")
    missing: list[Path] = []
    current = destination
    while not os.path.lexists(current):
        missing.append(current)
        require(current.parent != current, "cannot claim filesystem root")
        current = current.parent
    resolved = _directory_nonsymlink(current, f"ancestor of {destination}")
    require(resolved == current.absolute(), f"directory claim has a symlinked ancestor: {current}")
    for directory in reversed(missing):
        directory.mkdir(mode=leaf_mode if directory == destination else 0o700)
        _fsync_directory(directory.parent)
    return _directory_nonsymlink(destination, f"claimed directory {destination}")


def _mkdir_parents_nonsymlink(root: Path, relative_parent: Path) -> None:
    current = root
    for part in relative_parent.parts:
        current = current / part
        try:
            current.mkdir(mode=0o700)
            _fsync_directory(current.parent)
        except FileExistsError:
            info = current.lstat()
            require(stat.S_ISDIR(info.st_mode), f"snapshot parent is unsafe: {current}")


def _copy_verified(
    source_root: Path,
    relative: Path,
    destination: Path,
    expected_sha256: str,
) -> None:
    source_fd, _opened = _open_relative_regular(
        source_root, relative, f"snapshot source {relative}"
    )
    target_fd: int | None = None
    try:
        target_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        while block := os.read(source_fd, 16 * 1024 * 1024):
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(target_fd, view)
                require(written > 0, f"short snapshot write: {destination}")
                view = view[written:]
        os.fsync(target_fd)
        require(digest.hexdigest() == expected_sha256, f"snapshot source bytes changed: {relative}")
    finally:
        os.close(source_fd)
        if target_fd is not None:
            os.close(target_fd)
    require(file_sha256(destination) == expected_sha256, f"copied snapshot bytes differ: {destination}")


def verify_snapshot_files(snapshot_root: Path, inventory: Mapping[str, str]) -> None:
    root = _directory_nonsymlink(snapshot_root, "snapshot root")
    root_info = snapshot_root.lstat()
    require(root_info.st_mode & 0o222 == 0, "snapshot root is writable")
    expected = set(inventory)
    expected_directories = {
        str(parent)
        for raw_relative in inventory
        for parent in list(_safe_relative(raw_relative, "snapshot verification path").parents)[:-1]
    }
    actual: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        info = path.lstat()
        relative = str(path.relative_to(root))
        require(not stat.S_ISLNK(info.st_mode), f"snapshot contains symlink: {relative}")
        require(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode), f"snapshot contains special file: {relative}")
        if stat.S_ISREG(info.st_mode):
            actual.add(relative)
            require(relative in inventory, f"snapshot has an unclaimed file: {relative}")
            require(file_sha256(path) == inventory[relative], f"snapshot byte drift: {relative}")
            require(info.st_mode & 0o222 == 0, f"snapshot file is writable: {relative}")
        else:
            require(info.st_mode & 0o222 == 0, f"snapshot directory is writable: {relative}")
            actual_directories.add(relative)
    require(actual == expected, "snapshot file coverage differs from exact union")
    require(actual_directories == expected_directories, "snapshot directory coverage differs from exact union")


def create_source_snapshot(
    repo_root: Path, destination: Path, inventory: Mapping[str, str]
) -> Path:
    require(not destination.exists(), "snapshot destination already exists")
    parent = destination.parent
    _mkdir_chain_no_symlinks(parent)
    temporary = parent / f".repo.tmp.{os.getpid()}.{time.time_ns()}"
    temporary.mkdir(mode=0o700)
    _fsync_directory(parent)
    try:
        for raw_relative, digest in inventory.items():
            relative = _safe_relative(raw_relative, "snapshot inventory path")
            _mkdir_parents_nonsymlink(temporary, relative.parent)
            _copy_verified(repo_root, relative, temporary / relative, digest)
        for path in temporary.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        directories = sorted(
            (path for path in temporary.rglob("*") if path.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        )
        for directory in directories:
            directory.chmod(0o555)
        temporary.chmod(0o555)
        os.replace(temporary, destination)
        _fsync_directory(parent)
        parent.chmod(0o555)
        _fsync_directory(parent.parent)
    except BaseException:
        if temporary.exists():
            for path in temporary.rglob("*"):
                try:
                    path.chmod(0o700 if path.is_dir() else 0o600)
                except OSError:
                    pass
            try:
                temporary.chmod(0o700)
                shutil.rmtree(temporary)
            except OSError:
                pass
        raise
    verify_snapshot_files(destination, inventory)
    return destination


def exclusive_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o444) -> str:
    payload = (json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=False)
        _fsync_directory(path.parent.parent)
    else:
        _directory_nonsymlink(path.parent, f"parent of {path}")
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, f"short artifact write: {temporary}")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    return hashlib.sha256(payload).hexdigest()


def append_journal(submission_root: Path, ordinal: int, name: str, value: Mapping[str, Any]) -> Path:
    require(re.fullmatch(r"[A-Z0-9_]+", name) is not None, "journal record name is invalid")
    path = submission_root / "journal" / f"{ordinal:04d}_{name}.json"
    exclusive_json(path, {"schema_version": 1, "record": name.lower(), **dict(value)})
    return path


SchedulerRunner = Callable[[Sequence[str], Path, Mapping[str, str]], subprocess.CompletedProcess[str]]


def _default_scheduler_runner(
    command: Sequence[str], cwd: Path, environment: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _scheduler_environment() -> dict[str, str]:
    # Absolute Slurm clients use their compiled control-plane configuration.  Never
    # forward SLURM_CONF, library injection, Python, TreeWM, rank, or user payload.
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
    }


def _reconcile_job_ids(
    squeue: str,
    job_name: str,
    comment: str,
    cwd: Path,
    runner: SchedulerRunner,
) -> list[str]:
    completed = runner(
        [squeue, "--noheader", f"--name={job_name}", "--format=%A|%j|%u|%T|%k"],
        cwd,
        _scheduler_environment(),
    )
    require(completed.returncode == 0, f"cannot reconcile scheduler job {job_name}: {completed.stderr.strip()}")
    expected_user = pwd.getpwuid(os.getuid()).pw_name
    values: set[str] = set()
    for raw_line in completed.stdout.splitlines():
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.strip().split("|", 4)]
        require(len(fields) == 5, f"scheduler reconciliation returned malformed metadata for {job_name}")
        job_id, actual_name, user, state, actual_comment = fields
        require(JOB_ID.fullmatch(job_id) is not None, f"scheduler reconciliation returned malformed ID for {job_name}")
        require(actual_name == job_name, f"scheduler reconciliation name differs: {actual_name}")
        require(user == expected_user, f"scheduler reconciliation owner differs for {job_name}")
        require(bool(state), f"scheduler reconciliation state is empty for {job_name}")
        require(actual_comment == comment, f"scheduler reconciliation token differs for {job_name}")
        values.add(job_id)
    return sorted(values, key=int)


def _assert_job_absent(
    squeue: str,
    job_name: str,
    comment: str,
    cwd: Path,
    runner: SchedulerRunner,
) -> None:
    require(
        not _reconcile_job_ids(squeue, job_name, comment, cwd, runner),
        f"scheduler already contains transaction job {job_name}",
    )


def _submit_one(
    command: Sequence[str],
    *,
    job_name: str,
    comment: str,
    squeue: str,
    cwd: Path,
    runner: SchedulerRunner,
) -> tuple[str, dict[str, Any]]:
    completed = runner(command, cwd, _scheduler_environment())
    response = completed.stdout.strip()
    match = SBATCH_JOB.fullmatch(response)
    reconciled: list[str] = []
    if completed.returncode != 0:
        # Slurm may accept a job and lose the client response.  Never continue the
        # DAG after an error response; the outer transaction reconciles and cancels.
        parsed = [match.group("job_id")] if match is not None else []
        try:
            found = _reconcile_job_ids(squeue, job_name, comment, cwd, runner)
        except BaseException as exc:
            raise SchedulerSubmissionError(
                f"sbatch failed for {job_name}; reconciliation failed: {exc}",
                parsed,
            ) from exc
        raise SchedulerSubmissionError(
            f"sbatch failed for {job_name}; reconciled IDs={found}: {completed.stderr.strip()}",
            sorted(set([*parsed, *found]), key=int),
        )
    if match is None:
        reconciled = _reconcile_job_ids(squeue, job_name, comment, cwd, runner)
        if len(reconciled) != 1:
            raise SchedulerSubmissionError(
                f"sbatch response for {job_name} is ambiguous and reconciliation found {len(reconciled)} jobs: {response!r} {completed.stderr.strip()}",
                reconciled,
            )
        job_id = reconciled[0]
    else:
        job_id = match.group("job_id")
        # A parseable response is still checked by exact unique name.  This catches
        # fake/stale stdout before the dependent job is created.
        try:
            reconciled = _reconcile_job_ids(squeue, job_name, comment, cwd, runner)
        except BaseException as exc:
            raise SchedulerSubmissionError(
                f"scheduler could not reconcile exact submitted ID {job_id}: {exc}",
                [job_id],
            ) from exc
        if reconciled != [job_id]:
            raise SchedulerSubmissionError(
                f"scheduler did not reconcile exact submitted ID {job_id}",
                sorted(set([job_id, *reconciled]), key=int),
            )
    return job_id, {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "reconciled_job_ids": reconciled,
    }


def _cancel_exact(
    scancel: str,
    job_ids: Sequence[str],
    cwd: Path,
    runner: SchedulerRunner,
) -> dict[str, Any]:
    exact = sorted(set(job_ids), key=int)
    require(exact and all(JOB_ID.fullmatch(value) for value in exact), "refusing non-exact cancellation target")
    command = [scancel, *exact]
    completed = runner(command, cwd, _scheduler_environment())
    require(completed.returncode == 0, f"partial-submission cancellation failed: {completed.stderr.strip()}")
    return {"job_ids": exact, "command": command, "stdout": completed.stdout, "stderr": completed.stderr}


def _assert_live_transaction(submission_root: Path, *, ready: bool = False) -> None:
    """Reject a cancellation/abort marker at every scheduler linearization point."""

    forbidden = [
        submission_root / "CANCEL_REQUESTED.json",
        submission_root / "SUBMISSION_RECEIPT.json",
        submission_root / "journal" / "9000_RECOVERY_CANCELLED.json",
        submission_root / "journal" / "9999_ABORTED.json",
        submission_root / "journal" / "9998_OUTER_ABORTED.json",
    ]
    if not ready:
        forbidden.append(submission_root / "journal" / "0005_READY_TO_COMMIT.json")
    require(
        not any(os.path.lexists(path) for path in forbidden),
        "submission transaction acquired a terminal/cancellation marker",
    )


def _snapshot_preflight_in_process(
    snapshot_root: Path,
    protocol: str,
    inventory: Mapping[str, str],
    *,
    runner: AuditRunner,
) -> dict[str, Any]:
    package = snapshot_root / PACKAGE_RELATIVE
    manifest = read_json(package / "manifest.json")
    runtime_interpreter = activate_isolated_runtime(manifest)
    require(
        not any(name == "treewm" or name.startswith("treewm.") for name in sys.modules),
        "treewm was imported before snapshot containment was established",
    )
    campaign = load_campaign(snapshot_root)
    weight_lock = campaign.read_json(package / "weight_audit.lock.json")
    campaign.validate_manifest(manifest, weight_lock, snapshot_root)
    require(campaign.verify_protocol_lock(package) == protocol, "snapshot protocol differs")
    source_before = campaign.source_contract(snapshot_root)
    require(source_before["source_sha256"] == manifest["core_binding"]["trainer_code_fingerprint"], "snapshot trainer fingerprint differs")
    output_before = _output_tree_fingerprint(manifest)
    scientific_before = _scientific_output_fingerprint(manifest)
    audit_records, audit_results = rerun_audit_locks(
        snapshot_root,
        campaign,
        manifest,
        runner=runner,
        snapshot_inventory=inventory,
    )
    resolved = audit_results["resolved_config"]
    launches, compositions = direct_hydra_matrix(
        snapshot_root,
        campaign,
        manifest,
        weight_lock,
        protocol,
        resolved,
        runner=runner,
        snapshot_inventory=inventory,
    )
    source_after = campaign.source_contract(snapshot_root)
    output_after = _output_tree_fingerprint(manifest)
    scientific_after = _scientific_output_fingerprint(manifest)
    require(source_after == source_before, "snapshot audits/composition changed trainer source/runtime")
    require(campaign.verify_protocol_lock(package) == protocol, "snapshot audits/composition changed protocol")
    require(output_after == output_before, "snapshot audits/composition changed scientific outputs")
    require(scientific_after == scientific_before, "snapshot audits/composition changed cell outputs")
    for name, module in sorted(sys.modules.items()):
        if name != "treewm" and not name.startswith("treewm."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        require(
            Path(str(module_file)).resolve(strict=True).is_relative_to(snapshot_root),
            f"snapshot preflight imported {name} outside snapshot",
        )
    return {
        "manifest": manifest,
        "launches": launches,
        "compositions": compositions,
        "verification": {
            "audit_replays": audit_records,
            "full_output_fingerprint_before": output_before,
            "full_output_fingerprint_after": output_after,
            "scientific_output_fingerprint_before": scientific_before,
            "scientific_output_fingerprint_after": scientific_after,
            "source_sha256": source_before["source_sha256"],
            "runtime_sha256": source_before["runtime_sha256"],
            "orchestration_interpreter": runtime_interpreter,
            "import_containment": "all_treewm_modules_inside_snapshot",
        },
    }


def _snapshot_preflight(
    snapshot_root: Path,
    protocol: str,
    inventory: Mapping[str, str],
    *,
    runner: AuditRunner,
) -> dict[str, Any]:
    """Run the complete copied-tree verification in a clean isolated process."""

    package = snapshot_root / PACKAGE_RELATIVE
    manifest = read_json(package / "manifest.json")
    python = verify_submit_interpreter(manifest)
    program = package / "submit.py"
    _regular_nonsymlink(program, "snapshot submit verifier")
    command = [
        python,
        "-I",
        "-S",
        "-B",
        str(program),
        "--_snapshot-preflight",
        "--snapshot-root",
        str(snapshot_root),
        "--protocol-sha256",
        protocol,
        "--inventory-json",
        canonical_json(inventory),
    ]
    completed = runner(command, snapshot_root, _child_environment(), 14_400)
    require(
        completed.returncode == 0,
        "isolated snapshot preflight failed: "
        + completed.stderr.decode("utf-8", "replace")[-8000:],
    )
    value = _parse_audit_stdout(
        completed.stdout, "EXP23_SNAPSHOT_PREFLIGHT=", "snapshot preflight"
    )
    require(value.get("manifest") == manifest, "isolated snapshot manifest differs")
    require(len(value.get("launches") or []) == 20, "isolated snapshot launch matrix differs")
    verification = value.get("verification") or {}
    require(
        verification.get("scientific_output_fingerprint_before")
        == verification.get("scientific_output_fingerprint_after"),
        "isolated snapshot preflight changed outputs",
    )
    require(
        verification.get("import_containment")
        == "all_treewm_modules_inside_snapshot",
        "isolated snapshot import containment is absent",
    )
    return value


def _submission_contract(
    *,
    campaign: ModuleType,
    manifest: Mapping[str, Any],
    protocol: str,
    source: Mapping[str, Any],
    snapshot_root: Path,
    submission_root: Path,
    inventory: Mapping[str, str],
    launch_rows: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    compositions: Sequence[Mapping[str, Any]],
    snapshot_preflight: Mapping[str, Any],
    git: Mapping[str, Any],
) -> dict[str, Any]:
    audit_bindings = {
        "weight_audit_artifact_sha256": manifest["weight_audit"]["artifact_sha256"],
        "prefix_target_artifact_sha256": manifest["prefix_target_contract"]["artifact_sha256"],
        "resolved_config_artifact_sha256": manifest["resolved_config_contract"]["artifact_sha256"],
        "causal_parity_artifact_sha256": manifest["causal_parity_contract"]["artifact_sha256"],
    }
    rows: list[dict[str, Any]] = []
    for index, (launch, launch_row) in enumerate(zip(preflight["launches"], launch_rows, strict=True)):
        require(index == launch["cell"]["index"], "launch order differs")
        rows.append(
            {
                "index": index,
                "path": f"launches/cell-{index:02d}.json",
                "launch_sha256": launch["launch_sha256"],
                "launch_file_sha256": launch_row["launch_file_sha256"],
                "setting_id": launch["cell"]["setting"],
                "arm_id": launch["cell"]["arm"],
                "seed": launch["cell"]["seed"],
                **audit_bindings,
            }
        )
    contract: dict[str, Any] = {
        "schema_version": 1,
        "status": "sealed_for_submission",
        "campaign_id": manifest["campaign_id"],
        "formal_validation": False,
        "submission_root": str(submission_root.resolve()),
        "snapshot_root": str(snapshot_root.resolve()),
        "package_protocol_sha256": protocol,
        "manifest_sha256": campaign.manifest_sha256(manifest),
        "trainer_code_fingerprint": source["source_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "orchestration_interpreter": dict(preflight["orchestration_interpreter"]),
        **audit_bindings,
        "snapshot_inventory": dict(inventory),
        "snapshot_inventory_sha256": stable_hash(inventory),
        "live_audit_replays": preflight["audit_replays"],
        "snapshot_audit_replays": snapshot_preflight["audit_replays"],
        "direct_hydra_compositions": list(compositions),
        "scientific_output_fingerprint_before": preflight["scientific_output_fingerprint_before"],
        "scientific_output_fingerprint_after": preflight["scientific_output_fingerprint_after"],
        "full_output_fingerprint_before": preflight["full_output_fingerprint_before"],
        "full_output_fingerprint_after": preflight["full_output_fingerprint_after"],
        "snapshot_full_output_fingerprint_before": snapshot_preflight[
            "full_output_fingerprint_before"
        ],
        "snapshot_full_output_fingerprint_after": snapshot_preflight[
            "full_output_fingerprint_after"
        ],
        "snapshot_scientific_output_fingerprint_before": snapshot_preflight[
            "scientific_output_fingerprint_before"
        ],
        "snapshot_scientific_output_fingerprint_after": snapshot_preflight[
            "scientific_output_fingerprint_after"
        ],
        "git_provenance": dict(git),
        "launches": rows,
        "array": "0-19%20",
        "fresh_start": True,
    }
    require(len(rows) == 20 and [row["index"] for row in rows] == list(range(20)), "submission launch matrix differs")
    return contract


def _submit_campaign_impl(
    repo_root: Path,
    submission_root: Path,
    preflight: Mapping[str, Any],
    *,
    claim_token: str,
    audit_runner: AuditRunner = _default_runner,
    scheduler_runner: SchedulerRunner = _default_scheduler_runner,
) -> dict[str, Any]:
    reject_inherited_environment()
    require(preflight.get("status") == "read_only_preflight_verified", "submission lacks a verified preflight")
    require(
        set(preflight.get("audit_replays") or {})
        == {"weight", "prefix_target", "resolved_config", "causal_parity"},
        "submission requires all four independently replayed audit locks",
    )
    require(preflight["scientific_output_fingerprint_before"] == preflight["scientific_output_fingerprint_after"], "preflight output fingerprint differs")
    manifest = preflight["manifest"]
    require(
        activate_isolated_runtime(manifest) == preflight.get("orchestration_interpreter"),
        "orchestration interpreter changed after preflight",
    )
    git = git_provenance(repo_root)
    require(_namespace_is_fresh(manifest, submission_root), "submission namespace is no longer fresh")
    root = repo_root.resolve(strict=True)
    submission_root = submission_root.absolute()
    expected_submission_root = Path(str(manifest["paths"]["run_root"])) / "state" / "submission"
    require(submission_root == expected_submission_root.absolute(), "submission root differs from sealed run namespace")
    _mkdir_chain_no_symlinks(submission_root)
    append_journal(
        submission_root,
        0,
        "CLAIMED",
        {
            "campaign_id": manifest["campaign_id"],
            "submission_root": str(submission_root),
            "claim_token": claim_token,
            "scientific_output_fingerprint": preflight["scientific_output_fingerprint_after"],
        },
    )
    protocol = str(preflight["package_protocol_sha256"])
    live_campaign = load_campaign(root)
    inventory = snapshot_inventory(root, live_campaign, protocol)
    snapshot_root = submission_root / "source-snapshot" / "repo"
    create_source_snapshot(root, snapshot_root, inventory)
    append_journal(
        submission_root,
        1,
        "SNAPSHOT_SEALED",
        {
            "snapshot_root": str(snapshot_root),
            "inventory_sha256": stable_hash(inventory),
            "file_count": len(inventory),
        },
    )
    copied_preflight = _snapshot_preflight(
        snapshot_root, protocol, inventory, runner=audit_runner
    )
    snapshot_manifest = copied_preflight["manifest"]
    launches = copied_preflight["launches"]
    compositions = copied_preflight["compositions"]
    snapshot_verification = copied_preflight["verification"]
    require(snapshot_manifest == manifest, "snapshot manifest differs from preflight")
    require(
        snapshot_verification.get("source_sha256")
        == preflight["source_contract"]["source_sha256"]
        and snapshot_verification.get("runtime_sha256")
        == preflight["source_contract"]["runtime_sha256"],
        "snapshot source/runtime differs from live preflight",
    )
    require(
        snapshot_verification.get("orchestration_interpreter")
        == preflight.get("orchestration_interpreter"),
        "snapshot orchestration interpreter differs from live preflight",
    )
    require(
        set(snapshot_verification.get("audit_replays") or {})
        == {"weight", "prefix_target", "resolved_config", "causal_parity"},
        "snapshot did not replay all four audit locks",
    )
    # Live and copied command derivations must differ only in the repository-root
    # path embedded in the trainer argv.  The copied launches are authoritative.
    require(len(launches) == 20, "snapshot launch matrix is incomplete")
    require(
        all(
            launch["hashes"]["source_sha256"]
            == snapshot_verification["source_sha256"]
            and launch["hashes"]["runtime_sha256"]
            == snapshot_verification["runtime_sha256"]
            for launch in launches
        ),
        "snapshot launch source/runtime bindings differ",
    )
    launches_dir = submission_root / "launches"
    launches_dir.mkdir(mode=0o700)
    launch_rows: list[dict[str, Any]] = []
    for index, launch in enumerate(launches):
        path = launches_dir / f"cell-{index:02d}.json"
        digest = exclusive_json(path, launch)
        launch_rows.append({"index": index, "launch_file_sha256": digest})
    logs = submission_root / "logs"
    logs.mkdir(mode=0o700)
    _fsync_directory(submission_root)
    contract = _submission_contract(
        campaign=live_campaign,
        manifest=snapshot_manifest,
        protocol=protocol,
        source=preflight["source_contract"],
        snapshot_root=snapshot_root,
        submission_root=submission_root,
        inventory=inventory,
        launch_rows=launch_rows,
        preflight={**dict(preflight), "launches": launches},
        compositions=compositions,
        snapshot_preflight=snapshot_verification,
        git=git,
    )
    contract_path = submission_root / "SUBMISSION_CONTRACT.json"
    submission_sha256 = exclusive_json(contract_path, contract)
    append_journal(
        submission_root,
        2,
        "CONTRACT_SEALED",
        {"submission_sha256": submission_sha256, "launch_count": 20},
    )
    verify_snapshot_files(snapshot_root, inventory)
    require(file_sha256(contract_path) == submission_sha256, "submission contract changed before scheduler call")
    require(_scientific_output_fingerprint(manifest) == preflight["scientific_output_fingerprint_after"], "snapshot/preparation changed scientific outputs")

    execution = manifest["execution"]
    sbatch = str(execution["sbatch"])
    scancel = str(execution["scancel"])
    squeue = str(execution.get("squeue") or (Path(sbatch).parent / "squeue"))
    for path, label in ((sbatch, "sbatch"), (scancel, "scancel"), (squeue, "squeue")):
        _regular_nonsymlink(Path(path), label)
        require(os.access(path, os.X_OK), f"{label} is not executable")
    token = submission_sha256[:16]
    train_name = f"exp23-{token}-train"
    report_name = f"exp23-{token}-report"
    scheduler_comment = f"treewm-exp23:{submission_sha256}"
    train_script = snapshot_root / PACKAGE_RELATIVE / "train.slurm"
    report_script = snapshot_root / PACKAGE_RELATIVE / "report.slurm"
    _regular_nonsymlink(train_script, "snapshot training Slurm")
    _regular_nonsymlink(report_script, "snapshot report Slurm")
    train_command = [
        sbatch,
        "--parsable",
        "--export=NONE",
        "--array=0-19%20",
        f"--job-name={train_name}",
        f"--comment={scheduler_comment}",
        f"--output={logs / 'train_%A_%a.out'}",
        str(train_script),
        str(snapshot_root),
        str(submission_root),
        submission_sha256,
    ]
    known_ids: list[str] = []
    known_ids_by_role: dict[str, list[str]] = {"train": [], "report": []}
    active_role = "train"
    ready_to_commit = False
    records: dict[str, Any] = {}
    try:
        _assert_live_transaction(submission_root)
        _assert_job_absent(squeue, train_name, scheduler_comment, snapshot_root, scheduler_runner)
        _assert_job_absent(squeue, report_name, scheduler_comment, snapshot_root, scheduler_runner)
        train_id, train_record = _submit_one(
            train_command,
            job_name=train_name,
            comment=scheduler_comment,
            squeue=squeue,
            cwd=snapshot_root,
            runner=scheduler_runner,
        )
        known_ids.append(train_id)
        known_ids_by_role["train"].append(train_id)
        records["train"] = train_record
        append_journal(submission_root, 3, "TRAIN_SUBMITTED", {"job_id": train_id, **train_record})
        _assert_live_transaction(submission_root)
        report_command = [
            sbatch,
            "--parsable",
            "--export=NONE",
            f"--dependency=afterok:{train_id}",
            "--kill-on-invalid-dep=yes",
            f"--job-name={report_name}",
            f"--comment={scheduler_comment}",
            f"--output={logs / 'report_%j.out'}",
            str(report_script),
            str(snapshot_root),
            str(submission_root),
            submission_sha256,
        ]
        active_role = "report"
        report_id, report_record = _submit_one(
            report_command,
            job_name=report_name,
            comment=scheduler_comment,
            squeue=squeue,
            cwd=snapshot_root,
            runner=scheduler_runner,
        )
        known_ids.append(report_id)
        known_ids_by_role["report"].append(report_id)
        records["report"] = report_record
        append_journal(submission_root, 4, "REPORT_SUBMITTED", {"job_id": report_id, **report_record})
        _assert_live_transaction(submission_root)
        receipt = {
            "schema_version": 1,
            "status": "submitted",
            "campaign_id": manifest["campaign_id"],
            "submission_root": str(submission_root),
            "snapshot_root": str(snapshot_root),
            "submission_sha256": submission_sha256,
            "train_array_job_id": train_id,
            "report_job_id": report_id,
            "array": "0-19%20",
            "dependency": f"afterok:{train_id}",
        }
        append_journal(submission_root, 5, "READY_TO_COMMIT", receipt)
        ready_to_commit = True
        _assert_live_transaction(submission_root, ready=True)
        # The receipt is the transaction's final atomic commit point.  No fallible
        # journal write follows it.
        exclusive_json(submission_root / "SUBMISSION_RECEIPT.json", receipt)
        return {**receipt, "scheduler_calls": 6, "snapshot_files": len(inventory)}
    except BaseException as exc:
        if ready_to_commit:
            raise CommitRecoveryRequired(
                f"receipt commit failed after durable READY_TO_COMMIT: {exc}",
                sorted(set(known_ids), key=int),
            ) from exc
        # The response itself may have been lost after Slurm accepted a job.  Reconcile
        # both transaction-unique names, then cancel only the resulting exact IDs.
        known_ids.extend(getattr(exc, "job_ids", ()))
        known_ids_by_role[active_role].extend(getattr(exc, "job_ids", ()))
        reconciliation_errors: list[str] = []
        for role, name in (("train", train_name), ("report", report_name)):
            try:
                reconciled = _reconcile_job_ids(
                    squeue, name, scheduler_comment, snapshot_root, scheduler_runner
                )
                known_ids.extend(reconciled)
                known_ids_by_role[role].extend(reconciled)
            except BaseException as reconcile_exc:
                reconciliation_errors.append(f"{name}: {reconcile_exc}")
        cancellation: Mapping[str, Any] | None = None
        cancellation_error: str | None = None
        if known_ids:
            try:
                cancellation = _cancel_exact(
                    scancel, sorted(set(known_ids), key=int), snapshot_root, scheduler_runner
                )
            except BaseException as cancel_exc:
                cancellation_error = repr(cancel_exc)
        try:
            append_journal(
                submission_root,
                9999,
                "ABORTED",
                {
                    "error": repr(exc),
                    "known_job_ids": sorted(set(known_ids), key=int),
                    "job_ids_by_role": {
                        role: sorted(set(values), key=int)
                        for role, values in known_ids_by_role.items()
                    },
                    "submission_sha256": submission_sha256,
                    "reconciliation_errors": reconciliation_errors,
                    "cancellation": cancellation,
                    "cancellation_error": cancellation_error,
                },
            )
        except BaseException:
            pass
        detail = repr(exc)
        if reconciliation_errors:
            detail += "; reconciliation=" + "; ".join(reconciliation_errors)
        if cancellation_error:
            detail += "; cancellation=" + cancellation_error
        raise SchedulerSubmissionError(
            f"submission transaction aborted: {detail}",
            sorted(set(known_ids), key=int),
            {
                role: sorted(set(values), key=int)
                for role, values in known_ids_by_role.items()
            },
        ) from exc


def _submit_campaign_locked(
    repo_root: Path,
    submission_root: Path,
    preflight: Mapping[str, Any],
    *,
    audit_runner: AuditRunner = _default_runner,
    scheduler_runner: SchedulerRunner = _default_scheduler_runner,
) -> dict[str, Any]:
    """Guard every post-claim failure with a durable abort journal record."""

    claim_token = hashlib.sha256(
        f"{os.getpid()}:{time.time_ns()}:".encode("ascii") + os.urandom(32)
    ).hexdigest()
    try:
        return _submit_campaign_impl(
            repo_root,
            submission_root,
            preflight,
            audit_runner=audit_runner,
            scheduler_runner=scheduler_runner,
            claim_token=claim_token,
        )
    except BaseException as exc:
        if isinstance(exc, CommitRecoveryRequired):
            raise
        claimed_by_this_process = False
        claimed_path = submission_root / "journal" / "0000_CLAIMED.json"
        if os.path.lexists(claimed_path):
            try:
                claimed_by_this_process = read_json(claimed_path).get("claim_token") == claim_token
            except BaseException:
                claimed_by_this_process = False
        if claimed_by_this_process:
            try:
                _directory_nonsymlink(submission_root, "aborted submission root")
                receipt = submission_root / "SUBMISSION_RECEIPT.json"
                abort_path = submission_root / "journal" / "9998_OUTER_ABORTED.json"
                role_values = getattr(exc, "job_ids_by_role", {})
                role_ids = {
                    role: sorted(set(map(str, role_values.get(role, ()))), key=int)
                    for role in ("train", "report")
                }
                contract_path = submission_root / "SUBMISSION_CONTRACT.json"
                contract_sha256 = (
                    file_sha256(contract_path)
                    if os.path.lexists(contract_path)
                    and stat.S_ISREG(contract_path.lstat().st_mode)
                    else None
                )
                value = {
                    "schema_version": 1,
                    "record": "outer_aborted",
                    "error": repr(exc),
                    "receipt_committed": os.path.lexists(receipt),
                    "known_job_ids": list(getattr(exc, "job_ids", ())),
                    "job_ids_by_role": role_ids,
                    "claim_token": claim_token,
                    "submission_sha256": contract_sha256,
                }
                if os.path.lexists(abort_path):
                    require(read_json(abort_path) == value, "outer abort journal differs")
                elif not os.path.lexists(receipt):
                    exclusive_json(abort_path, value)
            except BaseException:
                pass
        raise


def submit_campaign(
    repo_root: Path,
    submission_root: Path,
    preflight: Mapping[str, Any],
    *,
    audit_runner: AuditRunner = _default_runner,
    scheduler_runner: SchedulerRunner = _default_scheduler_runner,
) -> dict[str, Any]:
    """Submit while holding the single crash-recovery linearization lock."""

    with _TransactionLock(submission_root):
        return _submit_campaign_locked(
            repo_root,
            submission_root,
            preflight,
            audit_runner=audit_runner,
            scheduler_runner=scheduler_runner,
        )


def _receipt_from_ready_record(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "status",
        "campaign_id",
        "submission_root",
        "snapshot_root",
        "submission_sha256",
        "train_array_job_id",
        "report_job_id",
        "array",
        "dependency",
    )
    require(value.get("record") == "ready_to_commit", "recovery journal is not ready-to-commit")
    require(set(value) == RECEIPT_FIELDS | {"record"}, "recovery READY schema differs")
    receipt = {key: value[key] for key in keys}
    require(receipt["schema_version"] == 1 and receipt["status"] == "submitted", "recovery receipt state differs")
    require(JOB_ID.fullmatch(str(receipt["train_array_job_id"])) is not None, "recovery train ID differs")
    require(JOB_ID.fullmatch(str(receipt["report_job_id"])) is not None, "recovery report ID differs")
    require(receipt["train_array_job_id"] != receipt["report_job_id"], "recovery job IDs are not distinct")
    require(receipt["campaign_id"] == CAMPAIGN_ID and receipt["array"] == "0-19%20", "recovery campaign/array differs")
    require(receipt["dependency"] == f"afterok:{receipt['train_array_job_id']}", "recovery dependency differs")
    return receipt


def _validated_abort_role_ids(value: Mapping[str, Any], label: str) -> dict[str, list[str]]:
    raw_roles = value.get("job_ids_by_role")
    require(
        isinstance(raw_roles, Mapping) and set(raw_roles) == {"train", "report"},
        f"{label} role-ID mapping differs",
    )
    result: dict[str, list[str]] = {}
    for role in ("train", "report"):
        raw_values = raw_roles[role]
        require(isinstance(raw_values, list), f"{label} {role} IDs are not a list")
        values = [str(item) for item in raw_values]
        require(
            all(JOB_ID.fullmatch(item) is not None for item in values)
            and values == sorted(set(values), key=int),
            f"{label} {role} IDs differ",
        )
        result[role] = values
    require(
        not set(result["train"]).intersection(result["report"]),
        f"{label} assigns one scheduler ID to both roles",
    )
    raw_flat = value.get("known_job_ids")
    require(isinstance(raw_flat, list), f"{label} flat IDs are not a list")
    flat = [str(item) for item in raw_flat]
    expected = sorted(set(result["train"] + result["report"]), key=int)
    require(flat == expected, f"{label} flat and role-specific IDs differ")
    return result


def _validated_successful_cancellation(
    aborted: Mapping[str, Any] | None,
    scancel: str,
    known_ids: Sequence[str],
) -> set[str]:
    """Authenticate a prior successful exact scancel so recovery is idempotent."""

    if aborted is None:
        return set()
    cancellation = aborted.get("cancellation")
    cancellation_error = aborted.get("cancellation_error")
    require(
        cancellation_error is None or isinstance(cancellation_error, str),
        "abort cancellation error has an invalid type",
    )
    if cancellation is None:
        return set()
    require(cancellation_error is None, "abort claims both cancellation success and failure")
    require(
        isinstance(cancellation, Mapping)
        and set(cancellation) == {"job_ids", "command", "stdout", "stderr"},
        "abort cancellation evidence schema differs",
    )
    raw_ids = cancellation["job_ids"]
    require(isinstance(raw_ids, list), "abort cancellation IDs are not a list")
    ids = [str(item) for item in raw_ids]
    require(
        ids
        and ids == sorted(set(ids), key=int)
        and all(JOB_ID.fullmatch(item) is not None for item in ids),
        "abort cancellation IDs differ",
    )
    require(
        ids == sorted(set(map(str, known_ids)), key=int),
        "abort successful cancellation does not cover every known ID",
    )
    require(cancellation["command"] == [scancel, *ids], "abort cancellation command differs")
    require(
        isinstance(cancellation["stdout"], str) and isinstance(cancellation["stderr"], str),
        "abort cancellation output types differ",
    )
    return set(ids)


def _recover_transaction_locked(
    repo_root: Path,
    submission_root: Path,
    *,
    scheduler_runner: SchedulerRunner = _default_scheduler_runner,
) -> dict[str, Any]:
    """Recover a process crash without ever creating another scheduler job."""

    reject_inherited_environment()
    _directory_nonsymlink(repo_root, "repository root")
    submission_root = _directory_nonsymlink(submission_root, "existing submission root")
    contract_path = submission_root / "SUBMISSION_CONTRACT.json"
    _regular_nonsymlink(contract_path, "recovery submission contract")
    submission_sha256 = file_sha256(contract_path)
    seal_path = submission_root / "journal" / "0002_CONTRACT_SEALED.json"
    _regular_nonsymlink(seal_path, "durable contract-seal journal")
    seal = read_json(seal_path)
    require(
        seal
        == {
            "schema_version": 1,
            "record": "contract_sealed",
            "submission_sha256": submission_sha256,
            "launch_count": 20,
        },
        "recovery contract does not match its durable seal journal",
    )
    claim_path = submission_root / "journal" / "0000_CLAIMED.json"
    _regular_nonsymlink(claim_path, "durable transaction claim")
    claim = read_json(claim_path)
    require(
        set(claim)
        == {
            "schema_version",
            "record",
            "campaign_id",
            "submission_root",
            "claim_token",
            "scientific_output_fingerprint",
        }
        and claim.get("schema_version") == 1
        and claim.get("record") == "claimed"
        and claim.get("campaign_id") == CAMPAIGN_ID
        and claim.get("submission_root") == str(submission_root)
        and SHA256.fullmatch(str(claim.get("claim_token", ""))) is not None
        and SHA256.fullmatch(str(claim.get("scientific_output_fingerprint", ""))) is not None,
        "durable transaction claim differs",
    )
    contract = read_json(contract_path)
    require(contract.get("schema_version") == 1 and contract.get("status") == "sealed_for_submission", "recovery contract is not sealed")
    require(contract.get("submission_root") == str(submission_root), "recovery submission root differs")
    snapshot_root = Path(str(contract.get("snapshot_root", "")))
    snapshot_resolved = _directory_nonsymlink(snapshot_root, "recovery snapshot root")
    require(snapshot_resolved.is_relative_to(submission_root), "recovery snapshot escapes submission root")
    inventory = contract.get("snapshot_inventory")
    require(isinstance(inventory, Mapping) and inventory, "recovery snapshot inventory is absent")
    require(stable_hash(inventory) == contract.get("snapshot_inventory_sha256"), "recovery inventory hash differs")
    verify_snapshot_files(snapshot_root, inventory)
    manifest = read_json(snapshot_root / PACKAGE_RELATIVE / "manifest.json")
    recovered_interpreter = activate_isolated_runtime(manifest)
    require(contract.get("manifest_sha256") == stable_hash(manifest), "recovery manifest binding differs")
    require(contract.get("package_protocol_sha256") == (snapshot_root / PACKAGE_RELATIVE / "protocol.sha256").read_text(encoding="ascii").strip(), "recovery protocol-lock text differs")
    campaign = load_campaign(snapshot_root)
    weight_lock = campaign.read_json(snapshot_root / PACKAGE_RELATIVE / "weight_audit.lock.json")
    campaign.validate_manifest(manifest, weight_lock, snapshot_root)
    source = campaign.source_contract(snapshot_root)
    require(
        contract.get("trainer_code_fingerprint") == source["source_sha256"]
        and contract.get("runtime_sha256") == source["runtime_sha256"],
        "recovery source/runtime contract differs",
    )
    require(
        contract.get("orchestration_interpreter") == recovered_interpreter,
        "recovery interpreter contract differs",
    )
    audit_bindings = {
        "weight_audit_artifact_sha256": manifest["weight_audit"]["artifact_sha256"],
        "prefix_target_artifact_sha256": manifest["prefix_target_contract"]["artifact_sha256"],
        "resolved_config_artifact_sha256": manifest["resolved_config_contract"]["artifact_sha256"],
        "causal_parity_artifact_sha256": manifest["causal_parity_contract"]["artifact_sha256"],
    }
    require(
        all(contract.get(key) == value for key, value in audit_bindings.items()),
        "recovery audit bindings differ",
    )

    receipt_path = submission_root / "SUBMISSION_RECEIPT.json"
    if os.path.lexists(receipt_path):
        _regular_nonsymlink(receipt_path, "committed submission receipt")
        receipt = read_json(receipt_path)
        require(set(receipt) == RECEIPT_FIELDS, "committed receipt schema differs")
        require(receipt.get("schema_version") == 1 and receipt.get("status") == "submitted", "committed receipt status differs")
        require(receipt.get("campaign_id") == CAMPAIGN_ID, "committed receipt campaign differs")
        require(receipt.get("submission_root") == str(submission_root), "committed receipt root differs")
        require(receipt.get("snapshot_root") == str(snapshot_root), "committed receipt snapshot differs")
        require(receipt.get("submission_sha256") == submission_sha256, "committed receipt submission differs")
        train_id = str(receipt.get("train_array_job_id", ""))
        report_id = str(receipt.get("report_job_id", ""))
        require(JOB_ID.fullmatch(train_id) is not None and JOB_ID.fullmatch(report_id) is not None and train_id != report_id, "committed receipt job IDs differ")
        require(receipt.get("array") == "0-19%20" and receipt.get("dependency") == f"afterok:{train_id}", "committed receipt scheduling contract differs")
        for role, ordinal, expected_id in (("train", 3, train_id), ("report", 4, report_id)):
            submitted = read_json(submission_root / "journal" / f"{ordinal:04d}_{role.upper()}_SUBMITTED.json")
            require(submitted.get("record") == f"{role}_submitted" and str(submitted.get("job_id")) == expected_id, f"committed receipt {role} journal differs")
        return {**receipt, "status": "submitted", "recovery": "already_committed", "scheduler_calls": 0}

    ready_path = submission_root / "journal" / "0005_READY_TO_COMMIT.json"
    if os.path.lexists(ready_path):
        require(
            not any(
                os.path.lexists(submission_root / "journal" / name)
                for name in ("9999_ABORTED.json", "9998_OUTER_ABORTED.json", "9000_RECOVERY_CANCELLED.json")
            ),
            "durable READY conflicts with an aborted/cancelled transaction",
        )
        ready = read_json(ready_path)
        receipt = _receipt_from_ready_record(ready)
        require(receipt["submission_sha256"] == submission_sha256, "ready receipt submission differs")
        require(receipt["submission_root"] == str(submission_root), "ready receipt root differs")
        require(receipt["snapshot_root"] == str(snapshot_root), "ready receipt snapshot differs")
        for role, ordinal, receipt_key in (
            ("train", 3, "train_array_job_id"),
            ("report", 4, "report_job_id"),
        ):
            submitted = read_json(
                submission_root / "journal" / f"{ordinal:04d}_{role.upper()}_SUBMITTED.json"
            )
            require(
                submitted.get("record") == f"{role}_submitted"
                and str(submitted.get("job_id")) == str(receipt[receipt_key]),
                f"ready receipt {role} ID differs from durable submission journal",
            )
        require(not os.path.lexists(submission_root / "CANCEL_REQUESTED.json"), "cancelled transaction cannot be committed")
        exclusive_json(receipt_path, receipt)
        return {**receipt, "recovery": "committed_from_durable_ready_record", "scheduler_calls": 0}

    prior_recovery_path = submission_root / "journal" / "9000_RECOVERY_CANCELLED.json"
    if os.path.lexists(prior_recovery_path):
        prior = read_json(prior_recovery_path)
        require(
            prior.get("record") == "recovery_cancelled"
            and prior.get("submission_sha256") == submission_sha256
            and prior.get("status") == "recovered_cancelled_transaction"
            and prior.get("new_jobs_created") == 0,
            "durable recovery-cancellation record differs",
        )
        require(not prior.get("cancellation_error"), "prior exact cancellation failed")
        require(not prior.get("reconciliation_errors"), "prior recovery had unresolved role reconciliation")
        return {**prior, "scheduler_calls": 0}

    execution = manifest["execution"]
    sbatch = str(execution["sbatch"])
    scancel = str(execution["scancel"])
    squeue = str(execution.get("squeue") or (Path(sbatch).parent / "squeue"))
    for path, label in ((scancel, "scancel"), (squeue, "squeue")):
        _regular_nonsymlink(Path(path), f"recovery {label}")
        require(os.access(path, os.X_OK), f"recovery {label} is not executable")
    token = submission_sha256[:16]
    train_name = f"exp23-{token}-train"
    report_name = f"exp23-{token}-report"
    comment = f"treewm-exp23:{submission_sha256}"
    role_names = {"train": train_name, "report": report_name}
    journal_paths = {
        "train": submission_root / "journal" / "0003_TRAIN_SUBMITTED.json",
        "report": submission_root / "journal" / "0004_REPORT_SUBMITTED.json",
    }
    aborted_ids_by_role: dict[str, list[str]] = {"train": [], "report": []}
    aborted: dict[str, Any] | None = None
    aborted_path = submission_root / "journal" / "9999_ABORTED.json"
    if os.path.lexists(aborted_path):
        aborted = read_json(aborted_path)
        require(
            set(aborted)
            == {
                "schema_version",
                "record",
                "error",
                "known_job_ids",
                "job_ids_by_role",
                "submission_sha256",
                "reconciliation_errors",
                "cancellation",
                "cancellation_error",
            }
            and aborted.get("schema_version") == 1
            and aborted.get("record") == "aborted"
            and aborted.get("submission_sha256") == submission_sha256,
            "abort journal is not authenticated to the recovery contract",
        )
        require(
            isinstance(aborted.get("error"), str)
            and isinstance(aborted.get("reconciliation_errors"), list)
            and all(isinstance(item, str) for item in aborted["reconciliation_errors"]),
            "abort error evidence differs",
        )
        aborted_ids_by_role = _validated_abort_role_ids(aborted, "abort")
    outer_path = submission_root / "journal" / "9998_OUTER_ABORTED.json"
    if os.path.lexists(outer_path):
        outer = read_json(outer_path)
        require(
            set(outer)
            == {
                "schema_version",
                "record",
                "error",
                "receipt_committed",
                "known_job_ids",
                "job_ids_by_role",
                "claim_token",
                "submission_sha256",
            }
            and outer.get("schema_version") == 1
            and outer.get("record") == "outer_aborted"
            and outer.get("receipt_committed") is False
            and outer.get("submission_sha256") == submission_sha256
            and outer.get("claim_token") == claim["claim_token"]
            and isinstance(outer.get("error"), str),
            "outer-abort journal is not authenticated to the recovery contract",
        )
        outer_roles = _validated_abort_role_ids(outer, "outer abort")
        for role in ("train", "report"):
            aborted_ids_by_role[role] = sorted(
                set(aborted_ids_by_role[role] + outer_roles[role]), key=int
            )
    ids_by_role: dict[str, list[str]] = {}
    reconciliation_errors: dict[str, str] = {}
    for role, name in role_names.items():
        known: list[str] = list(aborted_ids_by_role[role])
        journal_path = journal_paths[role]
        if os.path.lexists(journal_path):
            record = read_json(journal_path)
            expected_record = f"{role}_submitted"
            require(record.get("record") == expected_record, f"recovery {role} journal record differs")
            journal_id = str(record.get("job_id", ""))
            require(JOB_ID.fullmatch(journal_id) is not None, f"recovery {role} journal ID differs")
            known.append(journal_id)
        try:
            known.extend(_reconcile_job_ids(squeue, name, comment, snapshot_root, scheduler_runner))
        except BaseException as exc:
            reconciliation_errors[role] = repr(exc)
        ids_by_role[role] = sorted(set(known), key=int)
    ids = sorted(set(ids_by_role["train"] + ids_by_role["report"]), key=int)
    prior_cancelled_ids = _validated_successful_cancellation(
        aborted,
        scancel,
        [] if aborted is None else list(aborted["known_job_ids"]),
    )
    require(prior_cancelled_ids.issubset(set(ids)), "prior cancellation IDs escaped recovered IDs")
    ids_to_cancel = [value for value in ids if value not in prior_cancelled_ids]
    latch = {
        "schema_version": 1,
        "status": "cancel_requested",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "train_array_job_id": ids_by_role["train"][0] if len(ids_by_role["train"]) == 1 else None,
        "report_job_id": ids_by_role["report"][0] if len(ids_by_role["report"]) == 1 else None,
        "job_ids_by_role": ids_by_role,
        "recovery": True,
    }
    if not os.path.lexists(submission_root / "CANCEL_REQUESTED.json"):
        exclusive_json(submission_root / "CANCEL_REQUESTED.json", latch)
    cancellation = None
    cancellation_error = None
    if ids_to_cancel:
        try:
            cancellation = _cancel_exact(scancel, ids_to_cancel, snapshot_root, scheduler_runner)
        except BaseException as exc:
            cancellation_error = repr(exc)
    recovery_record = {
        "submission_sha256": submission_sha256,
        "status": "recovered_cancelled_transaction",
        "job_ids": ids,
        "job_ids_by_role": ids_by_role,
        "prior_cancelled_job_ids": sorted(prior_cancelled_ids, key=int),
        "new_cancel_job_ids": ids_to_cancel,
        "cancellation": cancellation,
        "cancellation_error": cancellation_error,
        "reconciliation_errors": reconciliation_errors,
        "new_jobs_created": 0,
    }
    if reconciliation_errors or cancellation_error:
        incomplete_path = submission_root / "journal" / "8999_RECOVERY_INCOMPLETE.json"
        if not os.path.lexists(incomplete_path):
            append_journal(submission_root, 8999, "RECOVERY_INCOMPLETE", recovery_record)
        detail = "; ".join(
            f"{role}: {error}" for role, error in sorted(reconciliation_errors.items())
        )
        if cancellation_error:
            detail += ("; " if detail else "") + f"scancel: {cancellation_error}"
        raise SubmissionError("recovery is incomplete: " + detail)
    recovery_path = submission_root / "journal" / "9000_RECOVERY_CANCELLED.json"
    if os.path.lexists(recovery_path):
        require(read_json(recovery_path) == {"schema_version": 1, "record": "recovery_cancelled", **recovery_record}, "recovery journal differs")
    else:
        append_journal(submission_root, 9000, "RECOVERY_CANCELLED", recovery_record)
    return {**recovery_record, "scheduler_calls": 2 + int(bool(ids_to_cancel))}


def recover_transaction(
    repo_root: Path,
    submission_root: Path,
    *,
    scheduler_runner: SchedulerRunner = _default_scheduler_runner,
) -> dict[str, Any]:
    """Recover only after proving that no live submitter owns the transaction."""

    with _TransactionLock(submission_root):
        return _recover_transaction_locked(
            repo_root,
            submission_root,
            scheduler_runner=scheduler_runner,
        )


def _summary(preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": preflight["status"],
        "campaign_id": preflight["campaign_id"],
        "package_protocol_sha256": preflight["package_protocol_sha256"],
        "source_sha256": preflight["source_contract"]["source_sha256"],
        "runtime_sha256": preflight["source_contract"]["runtime_sha256"],
        "audit_replays": preflight["audit_replays"],
        "direct_hydra_compositions": preflight["direct_hydra_compositions"],
        "scientific_output_fingerprint_before": preflight["scientific_output_fingerprint_before"],
        "scientific_output_fingerprint_after": preflight["scientific_output_fingerprint_after"],
        "cells": len(preflight["launches"]),
        "namespace_fresh": preflight["namespace_fresh"],
        "writes_performed": 0,
        "scheduler_calls": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--test-only", action="store_true", help="read-only verification (default)")
    actions.add_argument("--submit", action="store_true", help="explicitly create the seal and submit two jobs")
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--submission-root",
        type=Path,
        help="transaction root (default: <manifest run_root>/state/submission)",
    )
    return parser


def _internal_snapshot_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--_snapshot-preflight", action="store_true", required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--inventory-json", required=True)
    args = parser.parse_args(argv)
    require(SHA256.fullmatch(args.protocol_sha256) is not None, "snapshot protocol SHA256 is malformed")
    root = _directory_nonsymlink(args.snapshot_root, "snapshot root")
    try:
        inventory = json.loads(args.inventory_json, object_pairs_hook=_pairs)
    except json.JSONDecodeError as exc:
        raise SubmissionError(f"snapshot inventory JSON is invalid: {exc}") from exc
    require(isinstance(inventory, dict) and inventory, "snapshot inventory is absent")
    verify_snapshot_files(root, inventory)
    bootstrap_manifest = read_json(root / PACKAGE_RELATIVE / "manifest.json")
    activate_isolated_runtime(bootstrap_manifest)
    value = _snapshot_preflight_in_process(
        root, args.protocol_sha256, inventory, runner=_default_runner
    )
    print("EXP23_SNAPSHOT_PREFLIGHT=" + canonical_json(value), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if "--_snapshot-preflight" in raw_argv:
            return _internal_snapshot_main(raw_argv)
        args = _parser().parse_args(raw_argv)
        repo_root = _directory_nonsymlink(args.repo_root, "repository root")
        manifest = read_json(repo_root / PACKAGE_RELATIVE / "manifest.json")
        activate_isolated_runtime(manifest)
        submission_root = (
            args.submission_root.absolute()
            if args.submission_root is not None
            else Path(str(manifest["paths"]["run_root"])) / "state" / "submission"
        )
        if args.submit and os.path.lexists(submission_root):
            recovered = recover_transaction(repo_root, submission_root)
            print(json.dumps(recovered, sort_keys=True, indent=2, allow_nan=False))
            return 0 if recovered.get("status") == "submitted" else 2
        preflight = static_preflight(repo_root, submission_root)
        if not args.submit:
            print(json.dumps(_summary(preflight), sort_keys=True, indent=2, allow_nan=False))
            return 0
        receipt = submit_campaign(repo_root, submission_root, preflight)
        print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"Exp23 submission error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
