#!/usr/bin/env python3
"""Read-only preflight, or explicitly submit, the sealed twenty-cell Exp23 pilot.

The default action is the same as ``--test-only``.  Submission is deliberately a
separate, explicit mode.  No preflight path creates a directory, a bytecode file, a
snapshot, or a scheduler process.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import contextlib
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
import tempfile
import time
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence


sys.dont_write_bytecode = True

PACKAGE_RELATIVE = Path("experiments/23-treewm-executable-prefix-repair-pilot-v1")
CAMPAIGN_ID = "treewm-executable-prefix-repair-pilot-v1-launch8"
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
SBATCH_JOB = re.compile(r"^(?P<job_id>[1-9][0-9]*)$")
JOB_ID = re.compile(r"^[1-9][0-9]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA1 = re.compile(r"^[0-9a-f]{40}$")
TRAINER_BOOTSTRAP_SMOKE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "cell_index",
        "python_flags",
        "entry_relative_path",
        "config_package_relative_path",
        "config_package_sha256",
        "snapshot_inventory_sha256",
        "launch_sha256",
        "resolved_config_sha256",
        "stdout_sha256",
        "stdout_bytes",
        "cuda_visible_devices",
        "full_output_fingerprint_before",
        "full_output_fingerprint_after",
        "scientific_output_fingerprint_before",
        "scientific_output_fingerprint_after",
        "persistent_writes_performed",
        "scheduler_calls",
    }
)
RECEIPT_FIELDS = frozenset(
    {
        "schema_version", "status", "campaign_id", "submission_sha256",
        "submission_authorization_sha256", "array", "wave0_array_job_id",
        "wave1_array_job_id", "report_job_id", "wave1_dependency",
        "report_dependency", "kill_on_invalid_dependency",
        "within_wave_requeue", "wave0_submitted_held",
    }
)
AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version", "status", "campaign_id", "submission_sha256",
        "array", "job_ids", "dependencies", "kill_on_invalid_dependency",
        "within_wave_requeue", "wave0_submitted_held",
        "accepted_job_evidence_sha256", "authorized_at_utc",
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
EPHEMERAL_CHILD_DIRECTORIES = {
    "HOME": "home",
    "TMPDIR": "tmp",
    "MPLCONFIGDIR": "matplotlib",
    "XDG_CACHE_HOME": "xdg-cache",
    "XDG_CONFIG_HOME": "xdg-config",
    "XDG_DATA_HOME": "xdg-data",
    "XDG_STATE_HOME": "xdg-state",
    "TORCH_HOME": "torch",
    "HF_HOME": "huggingface",
    "WANDB_CACHE_DIR": "wandb-cache",
    "WANDB_CONFIG_DIR": "wandb-config",
}
WEIGHT_AUDIT_SOURCE_FILES = {
    "audit": PACKAGE_RELATIVE / "weight_audit.py",
    "trainer": Path("scripts/train.py"),
    "executable_loss": Path("treewm/losses/executable_prefix.py"),
    "action_projection": Path("treewm/planning/action_execution.py"),
    "objective_config": Path(
        "configs/experiment/treewm_v2_grounded_executable_prefix_pilot_v1.yaml"
    ),
    "dataset": Path("treewm/data/ogbench_dataset.py"),
    "future_recipe": Path("treewm/data/future_recipe.py"),
    "future_sets": Path("treewm/data/future_sets.py"),
    "total_loss": Path("treewm/losses/total.py"),
}

# Stdlib-only child bootstrap. ``-I -S`` prevents sitecustomize and every ``.pth``
# file from running; the two hash-bound distribution roots are appended directly.
ISOLATED_RUN_CODE = r"""
import hashlib, json, os, stat, sys
root, vsite, bsite, expected_sha, expected_size, expected_python, inventory_json, script = sys.argv[1:9]
if not (sys.flags.isolated and sys.flags.no_site): raise SystemExit('isolation flags absent')
if os.path.normpath(os.path.abspath(sys.executable)) != expected_python: raise SystemExit('lexical interpreter differs')
target = os.path.realpath(sys.executable)
def identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid,
            value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
def directory_mode(value, label):
    mode = stat.S_IMODE(value.st_mode)
    if not stat.S_ISDIR(value.st_mode) or not mode & 0o444 or not mode & 0o111:
        raise SystemExit(label + ' is not a traversable directory')
def execution_directory(value, label, sealed, root_entry=False):
    mode = stat.S_IMODE(value.st_mode)
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.getuid():
        raise SystemExit(label + ' type or owner differs')
    if sealed:
        if mode != 0o555: raise SystemExit(label + ' sealed mode differs')
    elif root_entry:
        if mode != 0o755: raise SystemExit(label + ' live-root mode differs')
    elif mode & 0o500 != 0o500 or mode & 0o022 or mode & 0o7000:
        raise SystemExit(label + ' live mode is unsafe')
def execution_file(value, label, sealed):
    mode = stat.S_IMODE(value.st_mode)
    if (not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid()
            or value.st_nlink != 1):
        raise SystemExit(label + ' type, owner, or link count differs')
    if sealed:
        if mode != 0o444: raise SystemExit(label + ' sealed mode differs')
    elif not mode & 0o400 or mode & 0o022 or mode & 0o7000:
        raise SystemExit(label + ' live mode is unsafe')
def stable_file(fd, before, label):
    if not stat.S_ISREG(before.st_mode) or not stat.S_IMODE(before.st_mode) & 0o444:
        raise SystemExit(label + ' is not a readable regular file')
    digest = hashlib.sha256()
    try:
        while True:
            block = os.read(fd, 16 * 1024 * 1024)
            if not block: break
            digest.update(block)
        after = os.fstat(fd)
    except OSError as exc:
        raise SystemExit(label + ' cannot be read: ' + str(exc)) from exc
    if identity(after) != identity(before): raise SystemExit(label + ' changed while hashing')
    return digest.hexdigest()
nofollow = getattr(os, 'O_NOFOLLOW', 0)
cloexec = getattr(os, 'O_CLOEXEC', 0)
directory_flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | nofollow | cloexec
file_flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | nofollow | cloexec
try: interpreter_fd = os.open(target, file_flags)
except OSError as exc: raise SystemExit('interpreter cannot be opened: ' + str(exc)) from exc
try:
    st = os.fstat(interpreter_fd)
    if not stat.S_ISREG(st.st_mode) or st.st_size != int(expected_size): raise SystemExit('interpreter identity differs')
    h = hashlib.sha256()
    while True:
        block = os.read(interpreter_fd, 16 * 1024 * 1024)
        if not block: break
        h.update(block)
    if identity(os.fstat(interpreter_fd)) != identity(st): raise SystemExit('interpreter changed while hashing')
finally: os.close(interpreter_fd)
if h.hexdigest() != expected_sha: raise SystemExit('interpreter hash differs')
if any('site-packages' in value for value in sys.path): raise SystemExit('site path loaded before bootstrap')
for value in (root, vsite, bsite):
    if not os.path.isdir(value) or os.path.islink(value): raise SystemExit('bootstrap root unavailable')
def open_directory(path, label):
    absolute = os.path.abspath(path)
    parts = absolute.split(os.sep)
    if (not os.path.isabs(path) or os.path.normpath(path) != path
            or any(part in ('.', '..') for part in parts)):
        raise SystemExit(label + ' is not normalized')
    descriptor = os.open(os.sep, directory_flags)
    try:
        for part in parts[1:]:
            if not part: continue
            child = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor); descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor); raise
def scan_tree(directory_fd, parent, before, inventory):
    directory_mode(before, 'snapshot directory ' + (parent or '.'))
    try:
        with os.scandir(directory_fd) as iterator:
            children = [(entry.name, entry.stat(follow_symlinks=False)) for entry in iterator]
    except OSError as exc:
        raise SystemExit('snapshot directory cannot be enumerated: ' + (parent or '.') + ': ' + str(exc)) from exc
    if len({name for name, _ in children}) != len(children): raise SystemExit('duplicate snapshot entry')
    for name, listed in sorted(children):
        if not isinstance(name, str) or name in ('', '.', '..') or '/' in name: raise SystemExit('unsafe snapshot entry name')
        relative = name if not parent else parent + '/' + name
        if stat.S_ISLNK(listed.st_mode): raise SystemExit('snapshot contains symlink: ' + relative)
        if stat.S_ISDIR(listed.st_mode):
            directory_mode(listed, 'snapshot directory ' + relative)
            try: child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
            except OSError as exc: raise SystemExit('snapshot directory cannot be opened: ' + relative + ': ' + str(exc)) from exc
            try:
                opened = os.fstat(child_fd)
                if identity(opened) != identity(listed): raise SystemExit('snapshot directory raced: ' + relative)
                if opened.st_uid != os.getuid() or stat.S_IMODE(opened.st_mode) != 0o555: raise SystemExit('unsafe snapshot directory')
                actual_dirs.add(relative)
                scan_tree(child_fd, relative, opened, inventory)
                if identity(os.fstat(child_fd)) != identity(opened): raise SystemExit('snapshot directory changed: ' + relative)
            finally: os.close(child_fd)
        elif stat.S_ISREG(listed.st_mode):
            if listed.st_mode & 0o222: raise SystemExit('unsafe snapshot file')
            try: child_fd = os.open(name, file_flags, dir_fd=directory_fd)
            except OSError as exc: raise SystemExit('snapshot file cannot be opened: ' + relative + ': ' + str(exc)) from exc
            try:
                opened = os.fstat(child_fd)
                if identity(opened) != identity(listed): raise SystemExit('snapshot file raced: ' + relative)
                if opened.st_uid != os.getuid() or opened.st_nlink != 1 or stat.S_IMODE(opened.st_mode) != 0o444: raise SystemExit('unsafe snapshot file')
                digest = stable_file(child_fd, opened, 'snapshot file ' + relative)
            finally: os.close(child_fd)
            if relative not in inventory: raise SystemExit('unclaimed snapshot file')
            if digest != inventory[relative]: raise SystemExit('snapshot file hash differs')
            actual_files.add(relative)
        else: raise SystemExit('snapshot contains special file: ' + relative)
    if identity(os.fstat(directory_fd)) != identity(before): raise SystemExit('snapshot directory changed: ' + (parent or '.'))
def verify_execution_target(root_fd, parts, expected_directories, expected_file, expected_digest, sealed):
    current_fd = root_fd
    opened_directories = []
    relative_parts = []
    try:
        for index, part in enumerate(parts[:-1]):
            listed = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            child_fd = os.open(part, directory_flags, dir_fd=current_fd)
            try:
                opened = os.fstat(child_fd)
                if identity(opened) != identity(listed): raise SystemExit('pre-import target directory raced: ' + part)
                execution_directory(opened, 'pre-import target directory ' + part, sealed)
                relative_parts.append(part)
                expected = ('/'.join(relative_parts), identity(opened))
                if index >= len(expected_directories) or expected_directories[index] != expected:
                    raise SystemExit('pre-import target directory identity differs: ' + part)
            except BaseException:
                os.close(child_fd); raise
            opened_directories.append((current_fd, part, child_fd, opened))
            current_fd = child_fd
        if len(opened_directories) != len(expected_directories): raise SystemExit('pre-import target directory coverage differs')
        listed_file = os.stat(parts[-1], dir_fd=current_fd, follow_symlinks=False)
        file_fd = os.open(parts[-1], file_flags, dir_fd=current_fd)
        try:
            opened_file = os.fstat(file_fd)
            if identity(opened_file) != identity(listed_file): raise SystemExit('pre-import target script raced')
            execution_file(opened_file, 'pre-import target script', sealed)
            if identity(opened_file) != expected_file: raise SystemExit('pre-import target script identity differs')
            digest = stable_file(file_fd, opened_file, 'pre-import target script')
            named_file = os.stat(parts[-1], dir_fd=current_fd, follow_symlinks=False)
            if identity(named_file) != identity(opened_file): raise SystemExit('pre-import target script name changed')
            if digest != expected_digest: raise SystemExit('pre-import target script hash differs')
        finally: os.close(file_fd)
        for parent_fd, name, child_fd, before in reversed(opened_directories):
            if identity(os.fstat(child_fd)) != identity(before): raise SystemExit('pre-import target directory changed: ' + name)
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if identity(named) != identity(before): raise SystemExit('pre-import target directory name changed: ' + name)
    finally:
        for _parent_fd, _name, child_fd, _before in reversed(opened_directories): os.close(child_fd)
sealed_context = bool(inventory_json)
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
    actual_files, actual_dirs = set(), set()
    named_root = os.lstat(root)
    try: root_fd = open_directory(root, 'snapshot root')
    except OSError as exc: raise SystemExit('snapshot root cannot be opened: ' + str(exc)) from exc
    try:
        opened_root = os.fstat(root_fd)
        if identity(opened_root) != identity(named_root): raise SystemExit('snapshot root raced')
        if opened_root.st_uid != os.getuid() or stat.S_IMODE(opened_root.st_mode) != 0o555: raise SystemExit('snapshot root unsafe')
        scan_tree(root_fd, '', opened_root, inventory)
        if identity(os.fstat(root_fd)) != identity(opened_root): raise SystemExit('snapshot root changed')
    finally: os.close(root_fd)
    if actual_files != expected_files or actual_dirs != expected_dirs: raise SystemExit('snapshot coverage differs')
root_real = os.path.abspath(root)
script_absolute = os.path.abspath(script)
if os.path.commonpath((script_absolute, root_real)) != root_real: raise SystemExit('target script escapes root')
script_relative = os.path.relpath(script_absolute, root_real)
parts = script_relative.split(os.sep)
if not parts or any(part in ('', '.', '..') for part in parts): raise SystemExit('target script path invalid')
try: directory_fd = open_directory(root_real, 'target root')
except OSError as exc: raise SystemExit('target root cannot be reopened: ' + str(exc)) from exc
descriptors = [directory_fd]
directory_records = []
target_directory_identities = []
try:
    target_root_opened = os.fstat(directory_fd)
    execution_directory(target_root_opened, 'target root', sealed_context, True)
    if inventory_json and identity(target_root_opened) != identity(opened_root): raise SystemExit('target root differs after inventory')
    target_relative_parts = []
    for part in parts[:-1]:
        listed_directory = os.stat(part, dir_fd=directory_fd, follow_symlinks=False)
        child_directory_fd = os.open(part, directory_flags, dir_fd=directory_fd)
        try:
            opened_directory = os.fstat(child_directory_fd)
            if identity(opened_directory) != identity(listed_directory): raise SystemExit('target directory raced: ' + part)
            execution_directory(opened_directory, 'target directory ' + part, sealed_context)
        except BaseException:
            os.close(child_directory_fd); raise
        directory_records.append((directory_fd, part, child_directory_fd, opened_directory))
        target_relative_parts.append(part)
        target_directory_identities.append(('/'.join(target_relative_parts), identity(opened_directory)))
        directory_fd = child_directory_fd
        descriptors.append(directory_fd)
    script_listed = os.stat(parts[-1], dir_fd=directory_fd, follow_symlinks=False)
    script_fd = os.open(parts[-1], file_flags, dir_fd=directory_fd)
    try:
        script_before = os.fstat(script_fd)
        if identity(script_before) != identity(script_listed): raise SystemExit('target script raced before read')
        execution_file(script_before, 'target script', sealed_context)
        script_digest = hashlib.sha256()
        script_chunks = []
        while True:
            block = os.read(script_fd, 16 * 1024 * 1024)
            if not block: break
            script_digest.update(block); script_chunks.append(block)
        if identity(os.fstat(script_fd)) != identity(script_before): raise SystemExit('target script changed while reading')
        script_named_after = os.stat(parts[-1], dir_fd=directory_fd, follow_symlinks=False)
        if identity(script_named_after) != identity(script_before): raise SystemExit('target script name changed while reading')
    finally: os.close(script_fd)
    for parent_fd, name, child_fd, directory_before in reversed(directory_records):
        if identity(os.fstat(child_fd)) != identity(directory_before): raise SystemExit('target directory changed while reading: ' + name)
        named_directory = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if identity(named_directory) != identity(directory_before): raise SystemExit('target directory name changed while reading: ' + name)
    if identity(os.fstat(descriptors[0])) != identity(target_root_opened): raise SystemExit('target root changed while reading script')
finally:
    for descriptor in reversed(descriptors): os.close(descriptor)
if inventory_json and inventory.get(script_relative) != script_digest.hexdigest(): raise SystemExit('target script inventory differs')
# Revalidate the complete immutable import surface immediately before exposing it
# to Python.  The same-UID malicious-process case is outside this contract (such a
# process can ptrace this interpreter); the descriptor-pinned sys.path also prevents
# an accidental pathname replacement after this boundary from redirecting imports.
try: import_root_fd = open_directory(root_real, 'pre-import snapshot root')
except OSError as exc: raise SystemExit('pre-import snapshot root cannot be opened: ' + str(exc)) from exc
site_fds = []
try:
    import_root_opened = os.fstat(import_root_fd)
    execution_directory(import_root_opened, 'pre-import snapshot root', sealed_context, True)
    if identity(import_root_opened) != identity(target_root_opened): raise SystemExit('execution root changed before import')
    verify_execution_target(import_root_fd, parts, target_directory_identities,
                            identity(script_before), script_digest.hexdigest(), sealed_context)
    if inventory_json:
        actual_files, actual_dirs = set(), set()
        scan_tree(import_root_fd, '', import_root_opened, inventory)
        if actual_files != expected_files or actual_dirs != expected_dirs: raise SystemExit('pre-import snapshot coverage differs')
        if identity(os.fstat(import_root_fd)) != identity(import_root_opened): raise SystemExit('snapshot root changed during pre-import validation')
    for site_label, site_path in (('venv site-packages', vsite), ('base site-packages', bsite)):
        # The frozen environment paths may use the cluster's trusted /lustre/fsw
        # alias.  Bind that alias once to its canonical target identity, then use
        # only the component-open canonical directory fd for imports.
        try:
            site_named = os.stat(site_path, follow_symlinks=True)
            site_canonical = os.path.realpath(site_path)
        except OSError as exc: raise SystemExit(site_label + ' cannot be stated: ' + str(exc)) from exc
        try: site_fd = open_directory(site_canonical, site_label)
        except OSError as exc: raise SystemExit(site_label + ' cannot be opened: ' + str(exc)) from exc
        site_fds.append([site_fd, site_label, None])
        site_opened = os.fstat(site_fd)
        site_fds[-1][2] = site_opened
        if identity(site_opened) != identity(site_named): raise SystemExit(site_label + ' raced before import')
        if site_opened.st_uid != os.getuid() or stat.S_IMODE(site_opened.st_mode) & 0o022: raise SystemExit(site_label + ' ownership/mode is unsafe')
    pinned_root = '/proc/self/fd/' + str(import_root_fd)
    pinned_script = os.path.join(pinned_root, *parts)
    sys.path.insert(0, pinned_root)
    sys.path.extend('/proc/self/fd/' + str(item[0]) for item in site_fds)
    sys.argv = [script, *sys.argv[9:]]
    namespace = {'__name__': '__main__', '__file__': pinned_script, '__cached__': None,
                 '__loader__': None, '__package__': '',
                 '__treewm_execution_root_fd__': import_root_fd,
                 '__treewm_execution_root_identity__': identity(import_root_opened),
                 '__treewm_execution_root_sealed__': sealed_context,
                 '__treewm_execution_script_relative__': script_relative,
                 '__treewm_execution_script_sha256__': script_digest.hexdigest(),
                 '__treewm_execution_directory_identities__': tuple(target_directory_identities),
                 '__treewm_execution_script_identity__': identity(script_before)}
    execution_error = None
    try: exec(compile(b''.join(script_chunks), pinned_script, 'exec'), namespace, namespace)
    except BaseException as exc: execution_error = exc
    if identity(os.fstat(import_root_fd)) != identity(import_root_opened): raise SystemExit('execution root changed during execution')
    try: named_execution_root_fd = open_directory(root_real, 'post-execution root')
    except OSError as exc: raise SystemExit('post-execution root cannot be opened: ' + str(exc)) from exc
    try:
        if identity(os.fstat(named_execution_root_fd)) != identity(import_root_opened): raise SystemExit('named execution root changed during execution')
    finally: os.close(named_execution_root_fd)
    verify_execution_target(import_root_fd, parts, target_directory_identities,
                            identity(script_before), script_digest.hexdigest(), sealed_context)
    for site_fd, site_label, site_before in site_fds:
        if identity(os.fstat(site_fd)) != identity(site_before): raise SystemExit(site_label + ' changed during execution')
    if execution_error is not None: raise execution_error
finally:
    for site_fd, _site_label, _site_before in reversed(site_fds): os.close(site_fd)
    os.close(import_root_fd)
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


class SchedulerBoundaryError(SubmissionError):
    """A scheduler client returned, but its authenticated boundary did not close."""

    def __init__(
        self,
        message: str,
        *,
        completed: subprocess.CompletedProcess[str] | None,
        observation: Mapping[str, Any] | None,
    ) -> None:
        super().__init__(message)
        self.completed = completed
        self.observation = None if observation is None else dict(observation)


class CommitRecoveryRequired(SchedulerSubmissionError):
    """Both jobs and READY are durable; retry may only commit the receipt."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SubmissionError(message)


def _lstat_if_present(
    path: str | Path, label: str | None = None
) -> os.stat_result | None:
    """Return one lexical entry, treating only ENOENT as absence."""

    source = Path(path)
    try:
        return source.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SubmissionError(
            f"cannot determine whether {label or source} exists: {exc}"
        ) from exc


def _lexical_exists(path: str | Path, label: str | None = None) -> bool:
    return _lstat_if_present(path, label) is not None


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


def exact_json_equal(left: object, right: object) -> bool:
    """Compare JSON values recursively without Python's bool/int coercion."""

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(exact_json_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(exact_json_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest, _payload = _hash_relative_regular(
        source.parent,
        Path(source.name),
        f"SHA256 source {source}",
        capture=False,
    )
    return digest


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json_payload(source: Path, payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SubmissionError(f"non-finite JSON value in {source}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubmissionError(f"cannot read JSON {source}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {source}")
    return value


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    _digest, payload = _hash_relative_regular(
        source.parent,
        Path(source.name),
        f"JSON artifact {source}",
        capture=True,
    )
    assert payload is not None
    return _decode_json_payload(source, payload)


def read_immutable_json(path: str | Path, label: str) -> tuple[dict[str, Any], str]:
    """Read one exact immutable controller artifact and bind its raw encoding."""

    source = Path(path)
    info = _regular_nonsymlink(source, label)
    require(stat.S_IMODE(info.st_mode) == 0o444, f"{label} mode differs")
    digest, payload = _hash_relative_regular(
        source.parent,
        Path(source.name),
        label,
        capture=True,
    )
    assert payload is not None
    value = _decode_json_payload(source, payload)
    canonical_payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    require(
        payload == canonical_payload
        and digest == hashlib.sha256(canonical_payload).hexdigest(),
        f"{label} raw bytes differ from its sealed JSON value",
    )
    return value, digest


def _regular_nonsymlink(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise SubmissionError(f"{label} is unavailable: {exc}") from exc
    require(stat.S_ISREG(info.st_mode), f"{label} is not a regular nonsymlink file")
    return info


def _directory_nonsymlink(path: Path, label: str) -> Path:
    lexical = path.absolute()
    require(
        lexical.is_absolute()
        and all(part not in ("", ".", "..") for part in lexical.parts[1:]),
        f"{label} is not an absolute normalized path",
    )
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            component = current.lstat()
        except OSError as exc:
            raise SubmissionError(f"{label} path component is unavailable: {current}: {exc}") from exc
        require(not stat.S_ISLNK(component.st_mode), f"{label} has a symlink path component: {current}")
    try:
        info = lexical.lstat()
    except OSError as exc:
        raise SubmissionError(f"{label} is unavailable: {exc}") from exc
    require(stat.S_ISDIR(info.st_mode), f"{label} is not a nonsymlink directory")
    # Keep the lexical path.  ``resolve`` would reopen the already-checked
    # components and could follow a symlink installed between the lstat walk and
    # the resolution.  Security-sensitive consumers open this path one component
    # at a time with O_NOFOLLOW below.
    return lexical


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

    safe_relative = _safe_relative(relative, label)
    root_lexical = _directory_nonsymlink(root, f"{label} root")
    current = root_lexical
    for index, part in enumerate(safe_relative.parts):
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise SubmissionError(f"{label} is unavailable: {exc}") from exc
        if index + 1 == len(safe_relative.parts):
            require(stat.S_ISREG(info.st_mode), f"{label} is not a regular nonsymlink file")
        else:
            require(stat.S_ISDIR(info.st_mode), f"{label} has a symlink/non-directory parent: {current}")
    return current


def _manifest_repository_root(
    manifest: Mapping[str, Any], candidate: str | Path | None = None
) -> Path:
    """Bind protected audit reads to the repository named by the manifest.

    A copied source tree is deliberately incomplete: Exp20 checkpoints remain in the
    live repository's protected output namespace.  The live input root is therefore
    distinct from the snapshot execution/import root, but it is not caller-selectable.
    It is derived from the exact absolute run root and its repository-relative form.
    """

    run_root = Path(str(manifest["paths"]["run_root"]))
    require(run_root.is_absolute(), "manifest run root is not absolute")
    prospective = _safe_relative(
        str(manifest["paths"]["prospective_run_root"]),
        "prospective run root",
    )
    repository = run_root
    for _part in prospective.parts:
        repository = repository.parent
    require(
        repository / prospective == run_root,
        "manifest run root does not match its repository-relative binding",
    )
    expected = _directory_nonsymlink(repository, "manifest repository root")
    require(
        expected == repository.absolute(),
        "manifest repository root changed while resolving",
    )
    if candidate is not None:
        supplied_path = Path(candidate)
        require(supplied_path.is_absolute(), "audit input root is not absolute")
        supplied = _directory_nonsymlink(supplied_path, "audit input root")
        require(
            supplied_path.absolute() == repository.absolute() and supplied == expected,
            "audit input root differs from the manifest repository root",
        )
    return expected


def _verify_external_audit_inputs(
    execution_root: Path,
    input_root: Path,
    weight_lock: Mapping[str, Any],
) -> None:
    """Verify live audit inputs without permitting live Python imports.

    The scientific source files are present in both roots.  Requiring both copies to
    match the frozen weight lock prevents the external Exp20 input root from acting as
    an alternate code root.  Checkpoint/data hashes are compared by the replay itself
    and then byte-semantically checked against the four frozen audit locks.
    """

    source_hashes = weight_lock.get("source_sha256")
    require(isinstance(source_hashes, Mapping), "weight audit source hashes are absent")
    for name, relative in WEIGHT_AUDIT_SOURCE_FILES.items():
        expected = str(source_hashes.get(name, ""))
        require(SHA256.fullmatch(expected) is not None, f"weight audit {name} hash is malformed")
        copied = _contained_regular_no_symlinks(
            execution_root, relative, f"snapshot audit source {name}"
        )
        live = _contained_regular_no_symlinks(
            input_root, relative, f"live audit source {name}"
        )
        require(file_sha256(copied) == expected, f"snapshot audit source differs: {name}")
        require(file_sha256(live) == expected, f"live audit source differs: {name}")
    _contained_regular_no_symlinks(
        input_root,
        Path("experiments/20-treewm-grounded-gauge-pilot-v2/manifest.json"),
        "protected Exp20 manifest",
    )
    _directory_nonsymlink(
        input_root / "outputs/treewm-grounded-gauge-pilot-v2-launch2",
        "protected Exp20 output root",
    )


def _open_absolute_directory_components(path: Path, label: str) -> tuple[int, os.stat_result]:
    """Open an absolute lexical directory without ever resolving a component."""

    lexical = path.absolute()
    require(
        lexical.is_absolute()
        and all(part not in ("", ".", "..") for part in lexical.parts[1:]),
        f"{label} is not an absolute normalized path",
    )
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | nofollow
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(lexical.anchor, flags)
        for part in lexical.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        _require_tree_directory_mode(info, label)
        result = descriptor
        descriptor = None
        return result, info
    except OSError as exc:
        raise SubmissionError(f"{label} is unavailable or symlinked: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_relative_regular(root: Path, relative: Path, label: str) -> tuple[int, os.stat_result]:
    """Open a repository file through O_NOFOLLOW directory descriptors."""

    safe_relative = _safe_relative(relative, label)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | nofollow
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors: list[int] = []
    source_fd: int | None = None
    try:
        descriptor, _root_info = _open_absolute_directory_components(root, f"{label} root")
        descriptors.append(descriptor)
        for part in safe_relative.parts[:-1]:
            descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            descriptors.append(descriptor)
            directory_info = os.fstat(descriptor)
            _require_tree_directory_mode(directory_info, f"{label} parent directory")
            require(
                directory_info.st_uid == os.getuid(),
                f"{label} parent directory owner differs",
            )
        source_fd = os.open(
            safe_relative.name,
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | nofollow
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=descriptor,
        )
        info = os.fstat(source_fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o444 == 0:
            raise SubmissionError(f"{label} is not a regular file")
        result = source_fd
        source_fd = None
        return result, info
    except OSError as exc:
        raise SubmissionError(f"{label} is unavailable or symlinked: {exc}") from exc
    finally:
        if source_fd is not None:
            os.close(source_fd)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _hash_relative_regular(
    root: Path,
    relative: Path,
    label: str,
    *,
    capture: bool = False,
) -> tuple[str, bytes | None]:
    """Hash one component-safe open inode and optionally return those exact bytes."""

    _directory_nonsymlink(root, f"{label} root")
    descriptor, before = _open_relative_regular(root, relative, label)
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if capture else None
    try:
        while block := os.read(descriptor, 16 * 1024 * 1024):
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
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
    require(identity(after) == identity(before), f"{label} changed while hashing")
    return digest.hexdigest(), b"".join(chunks) if chunks is not None else None


def _tree_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    """Identity/metadata which must remain stable during one protected traversal."""

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


def _require_tree_directory_mode(info: os.stat_result, label: str) -> None:
    require(stat.S_ISDIR(info.st_mode), f"{label} is not a directory")
    permissions = stat.S_IMODE(info.st_mode)
    require(
        permissions & 0o444 != 0 and permissions & 0o111 != 0,
        f"{label} is not traversable",
    )


def _read_stable_regular_fd(
    descriptor: int,
    before: os.stat_result,
    label: str,
    *,
    hash_content: bool,
) -> str | None:
    require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
    require(stat.S_IMODE(before.st_mode) & 0o444 != 0, f"{label} is not readable")
    digest = hashlib.sha256() if hash_content else None
    try:
        if digest is not None:
            while block := os.read(descriptor, 16 * 1024 * 1024):
                digest.update(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise SubmissionError(f"{label} could not be read or restated: {exc}") from exc
    require(
        _tree_stat_identity(after) == _tree_stat_identity(before),
        f"{label} changed while traversing",
    )
    return None if digest is None else digest.hexdigest()


def _secure_tree_rows(
    root: Path,
    label: str,
    *,
    hash_files: bool = True,
) -> tuple[os.stat_result, list[dict[str, Any]]]:
    """Enumerate a complete nonsymlink tree through directory descriptors.

    Unlike ``Path.rglob``/``os.walk``, an unreadable directory cannot be silently
    omitted.  Each child is opened relative to its already-open parent, every file
    hash is read from that exact inode, and parent/child metadata must remain stable
    across the traversal.
    """

    resolved = _directory_nonsymlink(root, f"{label} root")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow | cloexec
    file_flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | nofollow | cloexec
    try:
        root_named = resolved.lstat()
    except OSError as exc:
        raise SubmissionError(f"{label} root cannot be stated: {exc}") from exc
    root_fd, root_opened = _open_absolute_directory_components(resolved, f"{label} root")
    rows: list[dict[str, Any]] = []

    def walk(directory_fd: int, relative_parent: Path, opened: os.stat_result) -> None:
        _require_tree_directory_mode(opened, f"{label} directory {relative_parent or Path('.')}")
        try:
            with os.scandir(directory_fd) as iterator:
                children = []
                for entry in iterator:
                    try:
                        child_info = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        raise SubmissionError(
                            f"{label} entry cannot be stated: {relative_parent / entry.name}: {exc}"
                        ) from exc
                    children.append((entry.name, child_info))
        except OSError as exc:
            raise SubmissionError(
                f"{label} directory cannot be enumerated: {relative_parent or Path('.')}: {exc}"
            ) from exc
        require(
            len({name for name, _info in children}) == len(children),
            f"{label} directory enumeration contains duplicate names: {relative_parent}",
        )
        for name, listed in sorted(children, key=lambda value: value[0]):
            require(
                isinstance(name, str) and name not in ("", ".", "..") and "/" not in name,
                f"{label} contains an unsafe entry name",
            )
            relative = relative_parent / name
            require(not stat.S_ISLNK(listed.st_mode), f"{label} contains symlink: {relative}")
            if stat.S_ISDIR(listed.st_mode):
                _require_tree_directory_mode(listed, f"{label} directory {relative}")
                try:
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise SubmissionError(f"{label} directory cannot be opened: {relative}: {exc}") from exc
                try:
                    child_opened = os.fstat(child_fd)
                    require(
                        _tree_stat_identity(child_opened) == _tree_stat_identity(listed),
                        f"{label} directory raced before open: {relative}",
                    )
                    require(
                        child_opened.st_uid == os.getuid(),
                        f"{label} directory owner differs: {relative}",
                    )
                    rows.append(
                        {
                            "path": str(relative),
                            "kind": "directory",
                            "mode": stat.S_IMODE(child_opened.st_mode),
                            "size": child_opened.st_size,
                            "mtime_ns": child_opened.st_mtime_ns,
                        }
                    )
                    walk(child_fd, relative, child_opened)
                    child_after = os.fstat(child_fd)
                    require(
                        _tree_stat_identity(child_after) == _tree_stat_identity(child_opened),
                        f"{label} directory changed while traversing: {relative}",
                    )
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(listed.st_mode):
                require(
                    stat.S_IMODE(listed.st_mode) & 0o444 != 0,
                    f"{label} file is not readable: {relative}",
                )
                try:
                    child_fd = os.open(name, file_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise SubmissionError(f"{label} file cannot be opened: {relative}: {exc}") from exc
                try:
                    child_opened = os.fstat(child_fd)
                    require(
                        _tree_stat_identity(child_opened) == _tree_stat_identity(listed),
                        f"{label} file raced before open: {relative}",
                    )
                    require(
                        child_opened.st_uid == os.getuid()
                        and child_opened.st_nlink == 1,
                        f"{label} file ownership/link count differs: {relative}",
                    )
                    digest = _read_stable_regular_fd(
                        child_fd,
                        child_opened,
                        f"{label} file {relative}",
                        hash_content=hash_files,
                    )
                finally:
                    os.close(child_fd)
                row: dict[str, Any] = {
                    "path": str(relative),
                    "kind": "file",
                    "mode": stat.S_IMODE(child_opened.st_mode),
                    "size": child_opened.st_size,
                    "mtime_ns": child_opened.st_mtime_ns,
                }
                if digest is not None:
                    row["sha256"] = digest
                rows.append(row)
            else:
                raise SubmissionError(f"{label} contains special file: {relative}")
        try:
            after = os.fstat(directory_fd)
        except OSError as exc:
            raise SubmissionError(f"{label} directory cannot be restated: {relative_parent}: {exc}") from exc
        require(
            _tree_stat_identity(after) == _tree_stat_identity(opened),
            f"{label} directory changed while enumerating: {relative_parent or Path('.')}",
        )

    try:
        require(
            _tree_stat_identity(root_opened) == _tree_stat_identity(root_named),
            f"{label} root raced before open",
        )
        require(root_opened.st_uid == os.getuid(), f"{label} root owner differs")
        walk(root_fd, Path(), root_opened)
        root_after = os.fstat(root_fd)
        require(
            _tree_stat_identity(root_after) == _tree_stat_identity(root_opened),
            f"{label} root changed while traversing",
        )
        rebound_fd, rebound = _open_absolute_directory_components(
            resolved, f"{label} root revalidation"
        )
        try:
            require(
                _tree_stat_identity(rebound) == _tree_stat_identity(root_opened),
                f"{label} root pathname changed while traversing",
            )
        finally:
            os.close(rebound_fd)
    finally:
        os.close(root_fd)
    rows.sort(key=lambda row: str(row["path"]))
    return root_opened, rows


def _open_relative_directory(root: Path, relative: Path, label: str) -> tuple[int, os.stat_result]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow | getattr(os, "O_CLOEXEC", 0)
    descriptors: list[int] = []
    opened: int | None = None
    try:
        descriptor, _root_info = _open_absolute_directory_components(root, f"{label} root")
        descriptors.append(descriptor)
        for part in relative.parts:
            require(part not in ("", ".", ".."), f"{label} has an unsafe relative path")
            descriptor = os.open(part, flags, dir_fd=descriptor)
            descriptors.append(descriptor)
        opened = os.dup(descriptor)
        info = os.fstat(opened)
        require(stat.S_ISDIR(info.st_mode), f"{label} is not a directory")
        result = opened
        opened = None
        return result, info
    except OSError as exc:
        raise SubmissionError(f"{label} is unavailable or symlinked: {exc}") from exc
    finally:
        if opened is not None:
            os.close(opened)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _seal_tree_read_only(root: Path, label: str) -> None:
    """Seal a private tree through exact fds; never chmod a pathname target."""

    _root_info, rows = _secure_tree_rows(root, label)
    for row in (value for value in rows if value["kind"] == "file"):
        relative = _safe_relative(str(row["path"]), f"{label} file")
        descriptor, opened = _open_relative_regular(root, relative, f"{label} file {relative}")
        try:
            require(opened.st_nlink == 1, f"{label} file has external hard links: {relative}")
            os.fchmod(descriptor, 0o444)
            sealed = os.fstat(descriptor)
            require(
                (sealed.st_dev, sealed.st_ino) == (opened.st_dev, opened.st_ino)
                and stat.S_IMODE(sealed.st_mode) == 0o444,
                f"{label} file seal raced: {relative}",
            )
        except OSError as exc:
            raise SubmissionError(f"{label} file cannot be sealed: {relative}: {exc}") from exc
        finally:
            os.close(descriptor)
    directories = sorted(
        (value for value in rows if value["kind"] == "directory"),
        key=lambda value: len(Path(str(value["path"])).parts),
        reverse=True,
    )
    for row in directories:
        relative = _safe_relative(str(row["path"]), f"{label} directory")
        descriptor, opened = _open_relative_directory(
            root, relative, f"{label} directory {relative}"
        )
        try:
            os.fchmod(descriptor, 0o555)
            sealed = os.fstat(descriptor)
            require(
                (sealed.st_dev, sealed.st_ino) == (opened.st_dev, opened.st_ino)
                and stat.S_IMODE(sealed.st_mode) == 0o555,
                f"{label} directory seal raced: {relative}",
            )
        except OSError as exc:
            raise SubmissionError(f"{label} directory cannot be sealed: {relative}: {exc}") from exc
        finally:
            os.close(descriptor)
    descriptor, opened = _open_relative_directory(root, Path(), f"{label} root")
    try:
        os.fchmod(descriptor, 0o555)
        sealed = os.fstat(descriptor)
        require(
            (sealed.st_dev, sealed.st_ino) == (opened.st_dev, opened.st_ino)
            and stat.S_IMODE(sealed.st_mode) == 0o555,
            f"{label} root seal raced",
        )
    except OSError as exc:
        raise SubmissionError(f"{label} root cannot be sealed: {exc}") from exc
    finally:
        os.close(descriptor)


def _restore_private_tree_modes(root: Path, label: str) -> list[str]:
    """Restore a private temporary tree without ever following a symlink.

    Directories are made owner-traversable before descriptor enumeration, so a
    nested mode-000 directory cannot be skipped.  Unsafe entries are left for safe
    unlink by ``rmtree`` and returned to the caller, which must fail the operation.
    """

    anomalies: list[str] = []
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    path_only = getattr(os, "O_PATH", 0)
    require(path_only != 0, f"{label} requires O_PATH for race-safe cleanup")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | nofollow
        | getattr(os, "O_CLOEXEC", 0)
    )
    path_flags = path_only | nofollow | getattr(os, "O_CLOEXEC", 0)

    def same_inode(left: os.stat_result, right: os.stat_result) -> bool:
        return (left.st_dev, left.st_ino, left.st_uid) == (
            right.st_dev,
            right.st_ino,
            right.st_uid,
        )

    def restore_mode(
        parent_fd: int,
        name: str,
        listed: os.stat_result,
        mode: int,
        relative: Path,
    ) -> os.stat_result:
        """Change the mode of the exact O_PATH-pinned inode.

        Linux does not expose ``fchmodat2(AT_EMPTY_PATH)`` through :mod:`os` on the
        pinned Python.  ``/proc/self/fd/<n>`` is a stable reference to the already
        opened inode; unlike a user-controlled pathname it cannot be rename-swapped.
        Symlinks and foreign/hard-linked files are rejected before the chmod.
        """

        pinned: int | None = None
        try:
            pinned = os.open(name, path_flags, dir_fd=parent_fd)
            opened = os.fstat(pinned)
            require(same_inode(opened, listed), f"{label} entry raced: {relative}")
            require(opened.st_uid == os.getuid(), f"{label} entry owner differs: {relative}")
            require(
                stat.S_ISDIR(opened.st_mode) or stat.S_ISREG(opened.st_mode),
                f"{label} entry is not restorable: {relative}",
            )
            if stat.S_ISREG(opened.st_mode):
                require(opened.st_nlink == 1, f"{label} file has external hard links: {relative}")
            os.chmod(f"/proc/self/fd/{pinned}", mode)
            restored = os.fstat(pinned)
            require(
                same_inode(restored, opened) and stat.S_IMODE(restored.st_mode) == mode,
                f"{label} entry mode restoration raced: {relative}",
            )
            return restored
        except OSError as exc:
            raise SubmissionError(
                f"{label} entry permissions cannot be restored: {relative}: {exc}"
            ) from exc
        finally:
            if pinned is not None:
                os.close(pinned)

    def walk(directory_fd: int, parent: Path) -> None:
        try:
            with os.scandir(directory_fd) as iterator:
                children = []
                for entry in iterator:
                    try:
                        children.append((entry.name, entry.stat(follow_symlinks=False)))
                    except OSError as exc:
                        raise SubmissionError(
                            f"{label} entry cannot be stated: {parent / entry.name}: {exc}"
                        ) from exc
        except OSError as exc:
            raise SubmissionError(f"{label} cannot enumerate {parent}: {exc}") from exc
        for name, listed in sorted(children, key=lambda value: value[0]):
            relative = parent / name
            if listed.st_uid != os.getuid():
                anomalies.append(f"wrong-owner:{relative}")
                continue
            if stat.S_ISLNK(listed.st_mode):
                anomalies.append(f"symlink:{relative}")
                continue
            if stat.S_ISDIR(listed.st_mode):
                try:
                    restored = restore_mode(
                        directory_fd, name, listed, 0o700, relative
                    )
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise SubmissionError(
                        f"{label} directory permissions cannot be restored: {relative}: {exc}"
                    ) from exc
                try:
                    opened = os.fstat(child_fd)
                    require(
                        stat.S_ISDIR(opened.st_mode) and same_inode(opened, restored),
                        f"{label} directory raced during restoration: {relative}",
                    )
                    walk(child_fd, relative)
                    os.fchmod(child_fd, 0o700)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(listed.st_mode):
                if listed.st_nlink != 1:
                    anomalies.append(f"hardlink:{relative}")
                    continue
                restore_mode(directory_fd, name, listed, 0o600, relative)
            else:
                anomalies.append(f"special:{relative}")
        os.fchmod(directory_fd, 0o700)

    parent_fd = _open_relative_directory(root.parent, Path(), f"parent of {label} root")[0]
    root_path_fd: int | None = None
    try:
        try:
            named_root = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
            root_path_fd = os.open(root.name, path_flags, dir_fd=parent_fd)
        except OSError as exc:
            raise SubmissionError(f"{label} root cannot be pinned: {exc}") from exc
        opened_path_root = os.fstat(root_path_fd)
        require(
            stat.S_ISDIR(opened_path_root.st_mode)
            and opened_path_root.st_uid == os.getuid()
            and _tree_stat_identity(opened_path_root) == _tree_stat_identity(named_root),
            f"{label} root is unsafe",
        )
        os.chmod(f"/proc/self/fd/{root_path_fd}", 0o700)
        restored_root = os.fstat(root_path_fd)
        require(
            same_inode(restored_root, opened_path_root)
            and stat.S_IMODE(restored_root.st_mode) == 0o700,
            f"{label} root mode restoration raced",
        )
        try:
            root_fd = os.open(root.name, directory_flags, dir_fd=parent_fd)
        except OSError as exc:
            raise SubmissionError(f"{label} root cannot be opened for restoration: {exc}") from exc
        try:
            opened_root = os.fstat(root_fd)
            require(
                stat.S_ISDIR(opened_root.st_mode) and same_inode(opened_root, restored_root),
                f"{label} root raced during restoration",
            )
            walk(root_fd, Path())
            os.fchmod(root_fd, 0o700)
        finally:
            os.close(root_fd)
    finally:
        if root_path_fd is not None:
            os.close(root_path_fd)
        os.close(parent_fd)
    return anomalies


def _read_inventory_json(
    root: Path,
    relative: Path,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Parse only bytes captured from the exact hash-bound regular-file inode."""

    require(SHA256.fullmatch(expected_sha256) is not None, f"{label} hash is malformed")
    digest, payload = _hash_relative_regular(root, relative, label, capture=True)
    require(digest == expected_sha256, f"{label} differs from frozen snapshot inventory")
    assert payload is not None
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SubmissionError(f"non-finite JSON value in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubmissionError(f"cannot read frozen JSON {label}: {exc}") from exc
    require(isinstance(value, dict), f"frozen JSON root is not an object: {label}")
    return value


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


def load_dag_evidence(repo_root: str | Path) -> ModuleType:
    root = Path(repo_root).resolve(strict=True)
    return _load_module(
        "dag_evidence",
        root / PACKAGE_RELATIVE / "dag_evidence.py",
        containment_root=root,
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
    python_info = _lstat_if_present(python, "pinned Python")
    require(
        python_info is not None
        and (stat.S_ISLNK(python_info.st_mode) or stat.S_ISREG(python_info.st_mode)),
        "pinned Python changed type after interpreter verification",
    )
    venv_root = python.parent.parent
    pyvenv = venv_root / "pyvenv.cfg"
    _regular_nonsymlink(pyvenv, "pinned pyvenv.cfg")
    _pyvenv_digest, pyvenv_payload = _hash_relative_regular(
        pyvenv.parent, Path(pyvenv.name), "pinned pyvenv.cfg", capture=True
    )
    assert pyvenv_payload is not None
    values: dict[str, str] = {}
    try:
        pyvenv_text = pyvenv_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SubmissionError(f"pinned pyvenv.cfg is not UTF-8: {exc}") from exc
    for line in pyvenv_text.splitlines():
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
        "lexical_symlink_target": (
            os.readlink(python) if stat.S_ISLNK(python_info.st_mode) else None
        ),
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
    cache_root: Path,
    extra: Mapping[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    root = _directory_nonsymlink(cache_root, "ephemeral child root")
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
        }
    )
    for variable, relative in EPHEMERAL_CHILD_DIRECTORIES.items():
        directory = root / relative
        resolved = _directory_nonsymlink(directory, f"ephemeral {variable}")
        require(resolved.parent == root, f"ephemeral {variable} escapes its root")
        result[variable] = str(resolved)
    if extra:
        for key, value in extra.items():
            require(key != "TREEWM_STOP_AFTER_UPDATE", "staged stop is forbidden")
            require(
                key not in EPHEMERAL_CHILD_DIRECTORIES,
                f"launch environment may not replace ephemeral {key}",
            )
            result[str(key)] = str(value)
    return result


@contextlib.contextmanager
def _ephemeral_child_environment(
    extra: Mapping[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
):
    """Yield a contained writable cache tree and remove it after one runner call."""

    temporary_parent = _directory_nonsymlink(Path("/tmp"), "temporary parent")
    with tempfile.TemporaryDirectory(
        prefix=f"treewm-exp23-launch8-{os.getuid()}-", dir=temporary_parent
    ) as raw_root:
        root = Path(raw_root)
        root_info = root.lstat()
        require(
            stat.S_ISDIR(root_info.st_mode)
            and root_info.st_uid == os.getuid()
            and root_info.st_mode & 0o077 == 0,
            "ephemeral child root is not private",
        )
        for relative in sorted(set(EPHEMERAL_CHILD_DIRECTORIES.values())):
            (root / relative).mkdir(mode=0o700)
        yield _child_environment(root, extra, environ)


def _output_tree_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Hash every entry and byte beneath the prospective output root."""
    run_root = Path(str(manifest["paths"]["run_root"]))
    if not _lexical_exists(run_root):
        return stable_hash({"exists": False, "entries": []})
    _root_info, rows = _secure_tree_rows(run_root, "output tree")
    return stable_hash(rows)


def _scientific_output_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Hash only cell outputs, allowing intentional submission state creation."""

    run_root = Path(str(manifest["paths"]["run_root"]))
    rows: list[dict[str, Any]] = []
    for setting in manifest["design"]["settings"]:
        setting_relative = _safe_relative(str(setting), "scientific setting")
        require(len(setting_relative.parts) == 1, "scientific setting is not one path component")
        path = run_root / setting_relative
        if not _lexical_exists(path):
            rows.append({"setting": setting, "missing": True})
            continue
        root_info, setting_rows = _secure_tree_rows(
            path, f"scientific setting output {setting}"
        )
        rows.append(
            {
                "path": str(setting_relative),
                "kind": "directory",
                "mode": stat.S_IMODE(root_info.st_mode),
                "size": root_info.st_size,
                "mtime_ns": root_info.st_mtime_ns,
            }
        )
        for row in setting_rows:
            rows.append(
                {
                    **row,
                    "path": str(setting_relative / str(row["path"])),
                }
            )
    return stable_hash(rows)


def _namespace_is_fresh(manifest: Mapping[str, Any], submission_root: Path) -> bool:
    run_root = Path(str(manifest["paths"]["run_root"]))
    if _lexical_exists(submission_root):
        return False
    if not _lexical_exists(run_root):
        return True
    try:
        _root_info, rows = _secure_tree_rows(
            run_root, "prospective output namespace", hash_files=False
        )
    except SubmissionError:
        return False
    # An empty run root is harmless.  Any prior scientific setting, terminal marker,
    # receipt, or cancellation latch makes this prospective namespace non-fresh.
    return not rows


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
    audit_input_root: Path | None = None,
    runner: AuditRunner = _default_runner,
    timeout: float = 7_200,
    snapshot_inventory: Mapping[str, str] | None = None,
    sealed_snapshot: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        isinstance(snapshot_inventory, Mapping) and snapshot_inventory,
        "audit replay requires the static preflight snapshot inventory",
    )
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
    input_root = _manifest_repository_root(
        manifest, repo_root if audit_input_root is None else audit_input_root
    )
    weight_relative = PACKAGE_RELATIVE / AUDIT_LOCKS["weight"]
    weight_lock_sha256 = str(snapshot_inventory.get(str(weight_relative), ""))
    weight_lock = _read_inventory_json(
        repo_root,
        weight_relative,
        weight_lock_sha256,
        "frozen weight-audit lock",
    )
    _verify_external_audit_inputs(repo_root, input_root, weight_lock)
    for label, program, arguments, prefix in AUDITS:
        lock_relative = PACKAGE_RELATIVE / AUDIT_LOCKS[label]
        lock_sha256 = str(snapshot_inventory.get(str(lock_relative), ""))
        lock = _read_inventory_json(
            repo_root,
            lock_relative,
            lock_sha256,
            f"frozen {label} audit lock",
        )
        raw_command = [python, str(package / program), *arguments]
        if label in {"weight", "prefix_target"}:
            raw_command.extend(
                [
                    "--project-root",
                    str(input_root),
                    "--weight-lock-sha256",
                    weight_lock_sha256,
                ]
            )
        command = isolated_python_command(
            raw_command,
            repo_root,
            identity,
            intercept_python_children=True,
            # A live static preflight needs the frozen map for exact lock lookup, but
            # its repository is intentionally writable and contains files outside the
            # snapshot union.  Only copied-tree execution gets sealed-root enforcement.
            snapshot_inventory=snapshot_inventory if sealed_snapshot else None,
        )
        with _ephemeral_child_environment() as environment:
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
    composition_command = isolated_python_command(
        [*argv, "--cfg", "job", "--resolve"],
        root,
        interpreter,
        intercept_python_children=False,
        snapshot_inventory=snapshot_inventory,
    )
    with _ephemeral_child_environment(
        {str(key): str(value) for key, value in launch["environment"].items()}
    ) as environment:
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


def trainer_bootstrap_smoke(
    snapshot_root: Path,
    inventory: Mapping[str, str],
    launch: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    runner: AuditRunner = _default_runner,
    timeout: float = 300,
) -> dict[str, Any]:
    """Exercise the exact trainer entry/import/Hydra path without running the app."""

    root = _directory_nonsymlink(snapshot_root, "trainer smoke snapshot root")
    verify_snapshot_files(root, inventory)
    manifest = read_json(root / PACKAGE_RELATIVE / "manifest.json")
    interpreter = interpreter_contract(manifest)
    python = str(interpreter["lexical_executable"])
    entry = root / PACKAGE_RELATIVE / "train_entry.py"
    _regular_nonsymlink(entry, "trainer composition-smoke entry")
    argv = launch.get("argv")
    cell = launch.get("cell")
    environment = launch.get("environment")
    require(isinstance(argv, list) and len(argv) >= 3, "trainer smoke argv is invalid")
    require(isinstance(cell, Mapping) and type(cell.get("index")) is int, "trainer smoke cell is invalid")
    require(type(cell.get("seed")) is int, "trainer smoke seed is invalid")
    require(isinstance(environment, Mapping), "trainer smoke environment is invalid")
    require(argv[0] == python, "trainer smoke interpreter differs")
    require(Path(str(argv[1])) == root / "scripts/train.py", "trainer smoke script differs")
    command = [
        python,
        "-P",
        "-S",
        "-B",
        str(entry),
        "--_hydra-composition-smoke",
        "--snapshot-root",
        str(root),
        "--snapshot-inventory-sha256",
        stable_hash(inventory),
        "--snapshot-inventory-json",
        canonical_json(inventory),
        "--launch-json",
        canonical_json(launch),
    ]
    smoke_environment = {
        **{str(key): str(value) for key, value in environment.items()},
        "PYTHONHASHSEED": str(cell["seed"]),
        # Importing torch is unavoidable because scripts.train owns the Hydra
        # decorator.  Hiding every device ensures this probe cannot initialize or
        # reserve a campaign GPU.
        "CUDA_VISIBLE_DEVICES": "",
    }
    full_output_before = _output_tree_fingerprint(manifest)
    scientific_output_before = _scientific_output_fingerprint(manifest)
    with _ephemeral_child_environment(smoke_environment) as child_environment:
        completed = runner(command, root, child_environment, timeout)
    require(
        completed.returncode == 0,
        f"cell{cell['index']}: sealed trainer Hydra smoke failed "
        f"({completed.returncode}): "
        + completed.stderr.decode("utf-8", "replace")[-8000:],
    )
    try:
        from omegaconf import OmegaConf

        config = OmegaConf.to_container(
            OmegaConf.create(completed.stdout.decode("utf-8")), resolve=True
        )
    except Exception as exc:
        raise SubmissionError(
            f"cell{cell['index']}: sealed trainer Hydra smoke stdout is invalid: {exc}"
        ) from exc
    require(isinstance(config, dict), "sealed trainer Hydra smoke config is not an object")
    require(
        config == expected.get("resolved_config"),
        f"cell{cell['index']}: sealed trainer Hydra smoke config differs",
    )
    resolved_sha256 = stable_hash(config)
    require(
        resolved_sha256 == expected.get("resolved_config_sha256"),
        f"cell{cell['index']}: sealed trainer Hydra smoke config hash differs",
    )
    verify_snapshot_files(root, inventory)
    full_output_after = _output_tree_fingerprint(manifest)
    scientific_output_after = _scientific_output_fingerprint(manifest)
    require(
        full_output_after == full_output_before,
        "sealed trainer Hydra smoke changed the prospective output tree",
    )
    require(
        scientific_output_after == scientific_output_before,
        "sealed trainer Hydra smoke changed prospective scientific outputs",
    )
    return {
        "schema_version": 1,
        "status": "sealed_trainer_hydra_composition_verified",
        "cell_index": int(cell["index"]),
        "python_flags": ["-P", "-S", "-B"],
        "entry_relative_path": str(PACKAGE_RELATIVE / "train_entry.py"),
        "config_package_relative_path": "configs/__init__.py",
        "config_package_sha256": hashlib.sha256(b"").hexdigest(),
        "snapshot_inventory_sha256": stable_hash(inventory),
        "launch_sha256": str(launch["launch_sha256"]),
        "resolved_config_sha256": resolved_sha256,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stdout_bytes": len(completed.stdout),
        "cuda_visible_devices": "",
        "full_output_fingerprint_before": full_output_before,
        "full_output_fingerprint_after": full_output_after,
        "scientific_output_fingerprint_before": scientific_output_before,
        "scientific_output_fingerprint_after": scientific_output_after,
        "persistent_writes_performed": 0,
        "scheduler_calls": 0,
    }


def _validated_trainer_bootstrap_smoke(
    value: Mapping[str, Any],
    *,
    inventory_sha256: str,
    launch_sha256: str,
    resolved_config_sha256: str,
    full_output_fingerprint: str,
    scientific_output_fingerprint: str,
) -> dict[str, Any]:
    result = dict(value)
    require(set(result) == TRAINER_BOOTSTRAP_SMOKE_FIELDS, "trainer bootstrap smoke fields differ")
    require(
        result.get("schema_version") == 1
        and result.get("status") == "sealed_trainer_hydra_composition_verified"
        and result.get("cell_index") == 0,
        "trainer bootstrap smoke identity differs",
    )
    require(result.get("python_flags") == ["-P", "-S", "-B"], "trainer bootstrap smoke Python flags differ")
    require(
        result.get("entry_relative_path") == str(PACKAGE_RELATIVE / "train_entry.py")
        and result.get("config_package_relative_path") == "configs/__init__.py"
        and result.get("config_package_sha256") == hashlib.sha256(b"").hexdigest(),
        "trainer bootstrap smoke import binding differs",
    )
    require(
        result.get("snapshot_inventory_sha256") == inventory_sha256
        and result.get("launch_sha256") == launch_sha256
        and result.get("resolved_config_sha256") == resolved_config_sha256,
        "trainer bootstrap smoke source/launch/config binding differs",
    )
    require(
        SHA256.fullmatch(str(result.get("stdout_sha256", ""))) is not None
        and type(result.get("stdout_bytes")) is int
        and result["stdout_bytes"] > 0,
        "trainer bootstrap smoke stdout evidence differs",
    )
    require(result.get("cuda_visible_devices") == "", "trainer bootstrap smoke exposed CUDA")
    for flavor in ("full_output", "scientific_output"):
        require(
            SHA256.fullmatch(
                str(result.get(f"{flavor}_fingerprint_before", ""))
            ) is not None
            and SHA256.fullmatch(
                str(result.get(f"{flavor}_fingerprint_after", ""))
            ) is not None
            and result.get(f"{flavor}_fingerprint_before")
            == result.get(f"{flavor}_fingerprint_after"),
            f"trainer bootstrap smoke changed {flavor.replace('_', ' ')}s",
        )
    require(
        result.get("full_output_fingerprint_before") == full_output_fingerprint
        and result.get("scientific_output_fingerprint_before")
        == scientific_output_fingerprint,
        "trainer bootstrap smoke output evidence is detached from snapshot preflight",
    )
    require(
        result.get("persistent_writes_performed") == 0
        and result.get("scheduler_calls") == 0,
        "trainer bootstrap smoke side-effect ledger differs",
    )
    return result


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
    inventory = snapshot_inventory(
        root, campaign, protocol, source_contract=source_before
    )
    output_before = _output_tree_fingerprint(manifest)
    scientific_before = _scientific_output_fingerprint(manifest)
    require(_namespace_is_fresh(manifest, submission_root), "prospective output namespace is not fresh")
    if rerun_audits:
        audit_records, audit_results = rerun_audit_locks(
            root,
            campaign,
            manifest,
            runner=runner,
            snapshot_inventory=inventory,
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
    verify_inventory_sources(root, inventory)
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
        "snapshot_inventory": inventory,
        "snapshot_inventory_sha256": stable_hash(inventory),
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
    repo_root: Path,
    campaign: ModuleType,
    protocol: str,
    *,
    source_contract: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Freeze the exact source union whose protocol was verified in this pass."""

    root = repo_root.resolve(strict=True)
    source = campaign.source_contract(root) if source_contract is None else source_contract
    source_files = source.get("source_files")
    require(isinstance(source_files, Mapping) and source_files, "source file inventory is absent")
    inventory: dict[str, str] = {}
    for raw_relative, claimed_value in source_files.items():
        claimed = str(claimed_value)
        require(SHA256.fullmatch(claimed) is not None, "trainer source SHA256 is malformed")
        relative = _safe_relative(raw_relative, "trainer fingerprint path")
        actual, _bytes = _hash_relative_regular(
            root, relative, f"trainer source {relative}"
        )
        require(actual == claimed, f"trainer fingerprint byte drift: {relative}")
        inventory[str(relative)] = actual
    protocol_names = tuple(campaign.PROTOCOL_FILES)
    require(
        protocol_names and len(protocol_names) == len(set(protocol_names)),
        "protocol file list is empty or duplicated",
    )
    protocol_files: dict[str, str] = {}
    for raw_relative in protocol_names:
        protocol_relative = _safe_relative(raw_relative, "protocol path")
        relative = PACKAGE_RELATIVE / protocol_relative
        digest, _bytes = _hash_relative_regular(
            root, relative, f"protocol file {relative}"
        )
        protocol_files[str(protocol_relative)] = digest
        prior = inventory.get(str(relative))
        require(prior in (None, digest), f"snapshot union has conflicting hashes: {relative}")
        inventory[str(relative)] = digest
    require(
        stable_hash({"schema_version": 1, "files": protocol_files}) == protocol,
        "frozen protocol inventory differs from the verified protocol",
    )
    supplemental = dict(getattr(campaign, "SNAPSHOT_IMPORT_FILES", {}))
    for raw_relative, claimed_value in supplemental.items():
        claimed = str(claimed_value)
        require(SHA256.fullmatch(claimed) is not None, "supplemental snapshot SHA256 is malformed")
        relative = _safe_relative(raw_relative, "supplemental snapshot import path")
        digest, _bytes = _hash_relative_regular(
            root, relative, f"supplemental import file {relative}"
        )
        require(digest == claimed, f"supplemental snapshot byte drift: {relative}")
        prior = inventory.get(str(relative))
        require(prior in (None, digest), f"snapshot union has conflicting hashes: {relative}")
        inventory[str(relative)] = digest
    lock_relative = PACKAGE_RELATIVE / "protocol.sha256"
    lock_digest, lock_bytes = _hash_relative_regular(
        root, lock_relative, "protocol lock", capture=True
    )
    assert lock_bytes is not None
    try:
        locked_protocol = lock_bytes.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise SubmissionError("protocol lock is not ASCII") from exc
    require(locked_protocol == protocol, "protocol lock changed while inventorying snapshot")
    prior = inventory.get(str(lock_relative))
    require(prior in (None, lock_digest), "snapshot union conflicts with protocol lock")
    inventory[str(lock_relative)] = lock_digest
    return dict(sorted(inventory.items()))


def verify_inventory_sources(repo_root: Path, inventory: Mapping[str, str]) -> None:
    """Reopen every frozen source path; extra live repository files are irrelevant."""

    root = _directory_nonsymlink(repo_root, "inventory source root")
    require(isinstance(inventory, Mapping) and inventory, "snapshot inventory is absent")
    for raw_relative, claimed_value in inventory.items():
        claimed = str(claimed_value)
        require(SHA256.fullmatch(claimed) is not None, "snapshot inventory SHA256 is malformed")
        relative = _safe_relative(raw_relative, "snapshot inventory path")
        digest, _bytes = _hash_relative_regular(
            root, relative, f"snapshot inventory source {relative}"
        )
        require(digest == claimed, f"snapshot inventory source changed: {relative}")


def _fsync_directory(path: Path) -> None:
    _directory_nonsymlink(path, f"fsync directory {path}")
    descriptor, _info = _open_relative_directory(
        path, Path(), f"fsync directory {path}"
    )
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


def _transaction_lock_binding(lock: _TransactionLock) -> dict[str, Any]:
    require(lock.descriptor is not None, "transaction lock is not held")
    opened = os.fstat(lock.descriptor)
    named = lock.path.lstat()
    require(
        stat.S_ISREG(opened.st_mode)
        and (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)
        and opened.st_uid == os.getuid()
        and opened.st_nlink == 1,
        "transaction lock identity changed",
    )
    return {
        "path": str(lock.path),
        "device": opened.st_dev,
        "inode": opened.st_ino,
        "uid": opened.st_uid,
        "mode": stat.S_IMODE(opened.st_mode),
    }


def _scheduler_runner_with_lock_lease(
    runner: SchedulerRunner, lock: _TransactionLock
) -> SchedulerRunner:
    """Keep the transaction flock alive until every scheduler child exits."""

    binding = _transaction_lock_binding(lock)
    descriptor = lock.descriptor
    assert descriptor is not None

    def leased(
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
        inherited_fds: Sequence[int] = (),
    ) -> subprocess.CompletedProcess[str]:
        require(
            exact_json_equal(_transaction_lock_binding(lock), binding),
            "scheduler call transaction-lock lease differs",
        )
        descriptors = tuple(dict.fromkeys([*inherited_fds, descriptor]))
        return runner(command, cwd, environment, descriptors)

    setattr(leased, "transaction_lock_binding", binding)
    return leased


def _leased_transaction_lock_binding(runner: SchedulerRunner) -> dict[str, Any]:
    binding = getattr(runner, "transaction_lock_binding", None)
    require(
        isinstance(binding, Mapping)
        and set(binding) == {"path", "device", "inode", "uid", "mode"}
        and type(binding.get("uid")) is int
        and binding.get("uid") == os.getuid()
        and type(binding.get("mode")) is int
        and binding.get("mode") == 0o600
        and all(
            type(binding.get(key)) is int and binding[key] >= 0
            for key in ("device", "inode")
        ),
        "scheduler runner lacks an authenticated transaction-lock lease",
    )
    return dict(binding)


def _mkdir_chain_no_symlinks(path: Path, *, leaf_mode: int = 0o700) -> Path:
    destination = path.absolute()
    require(not _lexical_exists(destination), f"directory claim already exists: {destination}")
    missing: list[Path] = []
    current = destination
    while not _lexical_exists(current):
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
    source_fd, source_opened = _open_relative_regular(
        source_root, relative, f"snapshot source {relative}"
    )
    target_fd: int | None = None
    try:
        target_fd = os.open(
            destination,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        digest = hashlib.sha256()
        while block := os.read(source_fd, 16 * 1024 * 1024):
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(target_fd, view)
                require(written > 0, f"short snapshot write: {destination}")
                view = view[written:]
        source_after = os.fstat(source_fd)
        require(
            _tree_stat_identity(source_after) == _tree_stat_identity(source_opened),
            f"snapshot source changed while copying: {relative}",
        )
        os.fsync(target_fd)
        require(digest.hexdigest() == expected_sha256, f"snapshot source bytes changed: {relative}")
        target_opened = os.fstat(target_fd)
        os.lseek(target_fd, 0, os.SEEK_SET)
        copied_digest = _read_stable_regular_fd(
            target_fd,
            target_opened,
            f"copied snapshot file {relative}",
            hash_content=True,
        )
        require(copied_digest == expected_sha256, f"copied snapshot bytes differ: {relative}")
    finally:
        os.close(source_fd)
        if target_fd is not None:
            os.close(target_fd)


def verify_snapshot_files(snapshot_root: Path, inventory: Mapping[str, str]) -> None:
    root_info, rows = _secure_tree_rows(snapshot_root, "snapshot")
    require(root_info.st_mode & 0o222 == 0, "snapshot root is writable")
    expected = set(inventory)
    expected_directories = {
        str(parent)
        for raw_relative in inventory
        for parent in list(_safe_relative(raw_relative, "snapshot verification path").parents)[:-1]
    }
    actual: set[str] = set()
    actual_directories: set[str] = set()
    for row in rows:
        relative = str(row["path"])
        if row["kind"] == "file":
            actual.add(relative)
            require(relative in inventory, f"snapshot has an unclaimed file: {relative}")
            require(row["sha256"] == inventory[relative], f"snapshot byte drift: {relative}")
            require(int(row["mode"]) & 0o222 == 0, f"snapshot file is writable: {relative}")
        else:
            require(int(row["mode"]) & 0o222 == 0, f"snapshot directory is writable: {relative}")
            actual_directories.add(relative)
    require(actual == expected, "snapshot file coverage differs from exact union")
    require(actual_directories == expected_directories, "snapshot directory coverage differs from exact union")


def create_source_snapshot(
    repo_root: Path, destination: Path, inventory: Mapping[str, str]
) -> Path:
    require(not _lexical_exists(destination), "snapshot destination already exists")
    parent = destination.parent
    _mkdir_chain_no_symlinks(parent)
    parent_before = parent.lstat()
    require(
        stat.S_ISDIR(parent_before.st_mode) and parent_before.st_uid == os.getuid(),
        "snapshot parent is not a private owned directory",
    )
    temporary = parent / f".repo.tmp.{os.getpid()}.{time.time_ns()}"
    temporary.mkdir(mode=0o700)
    _fsync_directory(parent)
    try:
        for raw_relative, digest in inventory.items():
            relative = _safe_relative(raw_relative, "snapshot inventory path")
            _mkdir_parents_nonsymlink(temporary, relative.parent)
            _copy_verified(repo_root, relative, temporary / relative, digest)
        _seal_tree_read_only(temporary, "temporary source snapshot")
        os.replace(temporary, destination)
        _fsync_directory(parent)
        parent_fd, parent_opened = _open_relative_directory(
            parent.parent, Path(parent.name), "source-snapshot parent"
        )
        try:
            require(
                (
                    parent_opened.st_dev,
                    parent_opened.st_ino,
                    parent_opened.st_uid,
                    parent_opened.st_gid,
                    stat.S_IMODE(parent_opened.st_mode),
                )
                == (
                    parent_before.st_dev,
                    parent_before.st_ino,
                    parent_before.st_uid,
                    parent_before.st_gid,
                    0o700,
                ),
                "source-snapshot parent raced before sealing",
            )
            os.fchmod(parent_fd, 0o555)
            parent_sealed = os.fstat(parent_fd)
            require(
                (parent_sealed.st_dev, parent_sealed.st_ino)
                == (parent_opened.st_dev, parent_opened.st_ino)
                and stat.S_IMODE(parent_sealed.st_mode) == 0o555,
                "source-snapshot parent sealing raced",
            )
        finally:
            os.close(parent_fd)
        _fsync_directory(parent.parent)
    except BaseException:
        if _lexical_exists(temporary):
            cleanup_anomalies: list[str] = []
            try:
                cleanup_anomalies = _restore_private_tree_modes(
                    temporary, "failed temporary source snapshot"
                )
                shutil.rmtree(temporary)
            except OSError as cleanup_exc:
                raise SubmissionError(
                    f"failed source-snapshot cleanup could not complete: {cleanup_exc}"
                ) from cleanup_exc
            require(not _lexical_exists(temporary), "failed source-snapshot temporary tree survived cleanup")
            require(
                not cleanup_anomalies,
                "failed source-snapshot cleanup found unsafe entries: "
                + ",".join(cleanup_anomalies),
            )
        raise
    verify_snapshot_files(destination, inventory)
    return destination


def exclusive_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o444) -> str:
    payload = (json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if not _lexical_exists(path.parent):
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


SchedulerRunner = Callable[
    [Sequence[str], Path, Mapping[str, str], Sequence[int]],
    subprocess.CompletedProcess[str],
]

SCHEDULER_CONTROL_PLANE_FIELDS = frozenset(
    {
        "slurm_conf",
        "cluster_name",
        "slurmctld_hosts",
        "slurmctld_port",
        "auth_type",
        "gres_types",
        "cli_filter_plugins",
        "job_submit_plugins",
        "trust_model",
    }
)
SCHEDULER_TRUST_MODEL = (
    "root-admin mutable scheduler control plane; config and Lua policy bytes are "
    "observation-bound from preclaim through submission; root-owned Slurm clients, "
    "plugin binaries, and shared libraries are trusted mutable external runtime"
)
SCHEDULER_POLICY_FILES = (
    "cli_filter.lua",
    "cli_filter_config.lua",
    "cli_filter_config_defaults.lua",
)
SCHEDULER_POLICY_DIRECTORY = "cli_filters"
SCHEDULER_REQUIRED_POLICY_MODULES = frozenset(
    {
        "util.lua",
        "cli_filter_checks_nvl72.lua",
        "cli_filter_checks_qos.lua",
        "cli_filter_checks_stale_data.lua",
    }
)
DEPENDENCY_TEST_REQUIREMENT = {
    "phases": [
        "after_wave0_reconciliation_before_wave1_submission",
        "after_wave1_reconciliation_before_report_submission",
    ],
    "dependencies": [
        "afterok:<accepted_wave0_array_job_id>",
        "afterok:<accepted_wave1_array_job_id>",
    ],
    "kill_on_invalid_dep": "yes",
    "required": True,
}


def _external_stat_identity(value: os.stat_result) -> tuple[int, ...]:
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


def _root_owned_directory_chain(
    path: Path, label: str
) -> tuple[list[int], list[os.stat_result]]:
    """Pin a normalized absolute root-owned path without following any link."""

    lexical = path.absolute()
    require(
        path.is_absolute()
        and lexical == path
        and all(part not in ("", ".", "..") for part in path.parts[1:]),
        f"{label} is not an exact normalized absolute path",
    )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors: list[int] = []
    identities: list[os.stat_result] = []
    try:
        descriptor = os.open(path.anchor, flags)
        descriptors.append(descriptor)
        root_info = os.fstat(descriptor)
        require(
            stat.S_ISDIR(root_info.st_mode)
            and root_info.st_uid == 0
            and stat.S_IMODE(root_info.st_mode) & 0o022 == 0,
            f"{label} filesystem root is not root-owned and non-writable",
        )
        identities.append(root_info)
        for part in path.parts[1:]:
            listed = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            require(stat.S_ISDIR(listed.st_mode), f"{label} component is not a directory: {part}")
            child = os.open(part, flags, dir_fd=descriptor)
            descriptors.append(child)
            opened = os.fstat(child)
            require(
                _external_stat_identity(opened) == _external_stat_identity(listed),
                f"{label} component raced: {part}",
            )
            require(
                opened.st_uid == 0
                and stat.S_IMODE(opened.st_mode) & 0o022 == 0,
                f"{label} component is not root-owned and non-writable: {part}",
            )
            identities.append(opened)
            descriptor = child
        return descriptors, identities
    except OSError as exc:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise SubmissionError(f"{label} cannot be opened without symlinks: {exc}") from exc
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _root_owned_regular_observation(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    """Read one exact root-owned 0644 file and rebind its complete path."""

    require(path.name not in ("", ".", ".."), f"{label} leaf is unsafe")
    descriptors, directory_infos = _root_owned_directory_chain(path.parent, f"{label} parent")
    file_descriptor: int | None = None
    rebound_descriptors: list[int] = []
    try:
        parent_fd = descriptors[-1]
        listed = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        file_descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(file_descriptor)
        require(
            _external_stat_identity(opened) == _external_stat_identity(listed),
            f"{label} raced before open",
        )
        require(
            stat.S_ISREG(opened.st_mode)
            and opened.st_uid == 0
            and opened.st_gid == 0
            and opened.st_nlink == 1
            and stat.S_IMODE(opened.st_mode) == 0o644,
            f"{label} must be root:root regular mode 0644 with one link",
        )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        while block := os.read(file_descriptor, 1024 * 1024):
            total += len(block)
            require(total <= 16 * 1024 * 1024, f"{label} exceeds the 16 MiB bound")
            digest.update(block)
            chunks.append(block)
        after = os.fstat(file_descriptor)
        named_after = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        require(
            _external_stat_identity(after) == _external_stat_identity(opened)
            and _external_stat_identity(named_after) == _external_stat_identity(opened),
            f"{label} changed while reading",
        )
        rebound_descriptors, rebound_infos = _root_owned_directory_chain(
            path.parent, f"{label} parent revalidation"
        )
        require(
            [_external_stat_identity(value) for value in rebound_infos]
            == [_external_stat_identity(value) for value in directory_infos],
            f"{label} parent path changed while reading",
        )
        rebound_named = os.stat(
            path.name, dir_fd=rebound_descriptors[-1], follow_symlinks=False
        )
        require(
            _external_stat_identity(rebound_named) == _external_stat_identity(opened),
            f"{label} lexical path changed while reading",
        )
        payload = b"".join(chunks)
        return payload, {
            "path": str(path),
            "sha256": digest.hexdigest(),
            "identity": {
                "device": opened.st_dev,
                "inode": opened.st_ino,
                "mode": stat.S_IMODE(opened.st_mode),
                "uid": opened.st_uid,
                "gid": opened.st_gid,
                "nlink": opened.st_nlink,
                "size": opened.st_size,
                "mtime_ns": opened.st_mtime_ns,
                "ctime_ns": opened.st_ctime_ns,
            },
        }
    except OSError as exc:
        raise SubmissionError(f"{label} cannot be authenticated: {exc}") from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(rebound_descriptors):
            os.close(descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _scheduler_contract(value: object) -> dict[str, Any]:
    require(isinstance(value, Mapping), "scheduler control-plane contract is absent")
    contract = dict(value)
    require(
        set(contract) == SCHEDULER_CONTROL_PLANE_FIELDS,
        "scheduler control-plane fields differ",
    )
    require(
        exact_json_equal(
            contract,
            {
            "slurm_conf": "/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf",
            "cluster_name": "cs-oci-ord",
            "slurmctld_hosts": ["cs-oci-ord-a", "cs-oci-ord-b"],
            "slurmctld_port": 6817,
            "auth_type": "auth/munge",
            "gres_types": ["gpu"],
            "cli_filter_plugins": ["lua"],
            "job_submit_plugins": ["lua"],
            "trust_model": SCHEDULER_TRUST_MODEL,
            },
        ),
        "scheduler control-plane contract differs",
    )
    return contract


def _parse_slurm_config(payload: bytes, contract: Mapping[str, Any]) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SubmissionError(f"canonical Slurm config is not UTF-8: {exc}") from exc
    require("\x00" not in text, "canonical Slurm config contains NUL")
    directives: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        require(
            re.match(r"(?i)^include(?:\s|=)", line) is None,
            "canonical Slurm config may not include another file",
        )
        require("=" in line, "canonical Slurm config contains a malformed directive")
        raw_key, raw_value = line.split("=", 1)
        key = raw_key.strip().lower()
        value = raw_value.strip()
        require(key and value, "canonical Slurm config contains an empty directive")
        directives.setdefault(key, []).append(value)

    expected = _scheduler_contract(contract)
    require(
        directives.get("clustername") == [expected["cluster_name"]],
        "canonical Slurm ClusterName differs",
    )
    require(
        directives.get("slurmctldhost") == expected["slurmctld_hosts"],
        "canonical Slurm controller hosts differ",
    )
    ports = directives.get("slurmctldport", [str(expected["slurmctld_port"])])
    require(ports == [str(expected["slurmctld_port"])], "canonical Slurm controller port differs")
    require(
        directives.get("authtype") == [expected["auth_type"]],
        "canonical Slurm AuthType differs",
    )
    require(
        directives.get("grestypes") == [",".join(expected["gres_types"])],
        "canonical Slurm GresTypes differs",
    )
    require(
        directives.get("clifilterplugins")
        == [",".join(expected["cli_filter_plugins"])],
        "canonical Slurm CliFilterPlugins differs",
    )
    require(
        directives.get("jobsubmitplugins")
        == [",".join(expected["job_submit_plugins"])],
        "canonical Slurm JobSubmitPlugins differs",
    )
    require(
        directives.get("communicationparameters") == ["NoAddrCache"],
        "canonical Slurm communication parameters differ",
    )
    return {
        "cluster_name": expected["cluster_name"],
        "slurmctld_hosts": list(expected["slurmctld_hosts"]),
        "slurmctld_port": expected["slurmctld_port"],
        "auth_type": expected["auth_type"],
        "gres_types": list(expected["gres_types"]),
        "cli_filter_plugins": list(expected["cli_filter_plugins"]),
        "job_submit_plugins": list(expected["job_submit_plugins"]),
    }


def _scheduler_policy_observation(config_path: Path) -> dict[str, Any]:
    policy_root = config_path.parent
    names = [*SCHEDULER_POLICY_FILES]
    policy_directory = policy_root / SCHEDULER_POLICY_DIRECTORY
    descriptors, directory_infos = _root_owned_directory_chain(
        policy_directory, "Slurm CLI-filter policy directory"
    )
    try:
        with os.scandir(descriptors[-1]) as iterator:
            entries = []
            for entry in iterator:
                entries.append((entry.name, entry.stat(follow_symlinks=False)))
        require(
            len({name for name, _info in entries}) == len(entries),
            "Slurm CLI-filter policy directory has duplicate entries",
        )
        module_names = sorted(name for name, _info in entries)
        require(
            SCHEDULER_REQUIRED_POLICY_MODULES.issubset(module_names),
            "Slurm CLI-filter policy modules are incomplete",
        )
        for name, info in entries:
            require(
                name not in ("", ".", "..")
                and "/" not in name
                and stat.S_ISREG(info.st_mode),
                f"Slurm CLI-filter policy entry is unsafe: {name}",
            )
    except OSError as exc:
        raise SubmissionError(f"Slurm CLI-filter policy cannot be enumerated: {exc}") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)

    files: dict[str, Any] = {}
    for name in [*names, *(f"{SCHEDULER_POLICY_DIRECTORY}/{name}" for name in module_names)]:
        _payload, observation = _root_owned_regular_observation(
            policy_root / name, f"Slurm CLI-filter policy {name}"
        )
        files[name] = observation
    rebound_descriptors, rebound_infos = _root_owned_directory_chain(
        policy_directory, "Slurm CLI-filter policy directory revalidation"
    )
    try:
        require(
            [_external_stat_identity(value) for value in rebound_infos]
            == [_external_stat_identity(value) for value in directory_infos],
            "Slurm CLI-filter policy directory changed",
        )
        with os.scandir(rebound_descriptors[-1]) as iterator:
            rebound_names = sorted(entry.name for entry in iterator)
        require(rebound_names == module_names, "Slurm CLI-filter policy membership changed")
    finally:
        for descriptor in reversed(rebound_descriptors):
            os.close(descriptor)
    return {"files": files, "tree_sha256": stable_hash(files)}


def _scheduler_control_plane_capture(
    contract_value: object,
) -> tuple[bytes, dict[str, Any]]:
    contract = _scheduler_contract(contract_value)
    config_path = Path(str(contract["slurm_conf"]))
    payload_before, config_before = _root_owned_regular_observation(
        config_path, "canonical Slurm config"
    )
    critical = _parse_slurm_config(payload_before, contract)
    policy_before = _scheduler_policy_observation(config_path)
    payload_after, config_after = _root_owned_regular_observation(
        config_path, "canonical Slurm config revalidation"
    )
    policy_after = _scheduler_policy_observation(config_path)
    require(
        payload_after == payload_before
        and exact_json_equal(config_after, config_before)
        and exact_json_equal(policy_after, policy_before),
        "scheduler control plane changed while authenticating",
    )
    return payload_before, {
        "schema_version": 1,
        "trust_model": SCHEDULER_TRUST_MODEL,
        "config": config_before,
        "critical": critical,
        "cli_filter_policy": policy_before,
    }


def _scheduler_control_plane_observation(contract_value: object) -> dict[str, Any]:
    _payload, observation = _scheduler_control_plane_capture(contract_value)
    return observation


def _scheduler_fallback_config(contract_value: object) -> dict[str, Any]:
    """Capture the authenticated original cluster config for accepted-job control.

    The fallback never authorizes sbatch.  Its only consumers are exact-name squeue,
    exact-ID scancel, accepted dependency verification, and held-wave release.  Keeping
    the preclaim bytes in the sealed submission contract prevents later critical
    controller/config drift from redirecting or stranding an accepted job ID.
    """

    payload, observation = _scheduler_control_plane_capture(contract_value)
    return {
        "schema_version": 1,
        "purpose": (
            "accepted-job exact reconciliation, cancellation, dependency verification, "
            "and wave-zero release only; never submission or compute-side execution"
        ),
        "encoding": "base64",
        "payload_base64": base64.b64encode(payload).decode("ascii"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "source_control_plane": observation,
    }


def _validated_scheduler_fallback(
    value: object,
    contract_value: object,
    expected_source_observation: object | None = None,
) -> tuple[dict[str, Any], bytes]:
    require(isinstance(value, Mapping), "scheduler fallback config is absent")
    result = dict(value)
    require(
        set(result)
        == {
            "schema_version",
            "purpose",
            "encoding",
            "payload_base64",
            "sha256",
            "size",
            "source_control_plane",
        },
        "scheduler fallback config fields differ",
    )
    require(
        type(result["schema_version"]) is int
        and result["schema_version"] == 1
        and result["purpose"]
        == (
            "accepted-job exact reconciliation, cancellation, dependency verification, "
            "and wave-zero release only; never submission or compute-side execution"
        )
        and result["encoding"] == "base64"
        and isinstance(result["payload_base64"], str)
        and SHA256.fullmatch(str(result["sha256"])) is not None
        and type(result["size"]) is int
        and 0 < result["size"] <= 16 * 1024 * 1024,
        "scheduler fallback config metadata differs",
    )
    try:
        payload = base64.b64decode(result["payload_base64"], validate=True)
    except (ValueError, UnicodeError) as exc:
        raise SubmissionError(f"scheduler fallback config encoding differs: {exc}") from exc
    require(
        len(payload) == result["size"]
        and hashlib.sha256(payload).hexdigest() == result["sha256"],
        "scheduler fallback config bytes differ",
    )
    critical = _parse_slurm_config(payload, _scheduler_contract(contract_value))
    source = result["source_control_plane"]
    require(
        isinstance(source, Mapping)
        and type(source.get("schema_version")) is int
        and source.get("schema_version") == 1
        and source.get("trust_model") == SCHEDULER_TRUST_MODEL
        and exact_json_equal(source.get("critical"), critical)
        and isinstance(source.get("config"), Mapping)
        and source["config"].get("sha256") == result["sha256"]
        and isinstance(source["config"].get("identity"), Mapping)
        and source["config"]["identity"].get("size") == result["size"],
        "scheduler fallback source observation differs",
    )
    if expected_source_observation is not None:
        require(
            exact_json_equal(source, expected_source_observation),
            "scheduler control plane changed after the exact preclaim",
        )
    return result, payload


@contextlib.contextmanager
def _scheduler_fallback_descriptor(payload: bytes):
    """Expose immutable authenticated config bytes to one inherited Slurm client."""

    require(hasattr(os, "memfd_create"), "scheduler fallback requires memfd_create")
    descriptor = os.memfd_create(
        "treewm-exp23-slurm-conf",
        getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0),
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, "scheduler fallback config write was short")
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.lseek(descriptor, 0, os.SEEK_SET)
        seals = (
            getattr(fcntl, "F_SEAL_SEAL", 0)
            | getattr(fcntl, "F_SEAL_SHRINK", 0)
            | getattr(fcntl, "F_SEAL_GROW", 0)
            | getattr(fcntl, "F_SEAL_WRITE", 0)
        )
        require(seals != 0, "scheduler fallback seals are unavailable")
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        require(
            fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & seals == seals,
            "scheduler fallback config is not immutable",
        )
        yield descriptor
    except OSError as exc:
        raise SubmissionError(f"scheduler fallback descriptor failed: {exc}") from exc
    finally:
        os.close(descriptor)


def _default_scheduler_runner(
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    inherited_fds: Sequence[int] = (),
) -> subprocess.CompletedProcess[str]:
    exact_command = list(command)
    guard_source = (
        "import os,subprocess,sys;"
        "fds=tuple(int(v) for v in sys.argv[1].split(',') if v);"
        "p=subprocess.run(sys.argv[2:],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,"
        "stderr=subprocess.PIPE,pass_fds=fds,close_fds=True);"
        "os.write(1,p.stdout);os.write(2,p.stderr);raise SystemExit(p.returncode)"
    )
    outer_command = exact_command
    if inherited_fds:
        # This authenticated wrapper, rather than the Slurm client, owns the
        # inherited flock lease.  It retains every descriptor while synchronously
        # waiting for the real client, even if that client closes unknown FDs.
        outer_command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            guard_source,
            ",".join(map(str, inherited_fds)),
            *exact_command,
        ]
    completed = subprocess.run(
        outer_command,
        cwd=cwd,
        env=dict(environment),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=tuple(inherited_fds),
    )
    return subprocess.CompletedProcess(
        exact_command,
        completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _scheduler_environment(control_plane: object) -> dict[str, str]:
    contract = _scheduler_contract(control_plane)
    # Never forward library injection, Python, TreeWM, rank, or user payload.  The
    # sole control-plane input is the exact manifest-bound root-admin configuration.
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "SLURM_CONF": str(contract["slurm_conf"]),
    }


def _scheduler_call(
    command: Sequence[str],
    cwd: Path,
    control_plane: object,
    runner: SchedulerRunner,
    expected_observation: Mapping[str, Any] | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Run one scheduler client while its root-admin inputs remain exact."""

    try:
        before = _scheduler_control_plane_observation(control_plane)
    except BaseException as exc:
        raise SchedulerBoundaryError(
            f"scheduler control plane failed before client call: {exc}",
            completed=None,
            observation=None,
        ) from exc
    if expected_observation is not None and not exact_json_equal(
        before, expected_observation
    ):
        raise SchedulerBoundaryError(
            "scheduler control plane differs from the exact preclaim authorization",
            completed=None,
            observation=before,
        )
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        completed = runner(
            command,
            cwd,
            _scheduler_environment(control_plane),
            (),
        )
    except BaseException as exc:
        try:
            after = _scheduler_control_plane_observation(control_plane)
            require(
                exact_json_equal(after, before),
                "scheduler control plane changed during failed client call",
            )
        except BaseException as boundary_exc:
            raise SchedulerBoundaryError(
                f"scheduler client and post-call authentication both failed: {exc}; {boundary_exc}",
                completed=None,
                observation=before,
            ) from exc
        raise
    try:
        after = _scheduler_control_plane_observation(control_plane)
    except BaseException as exc:
        raise SchedulerBoundaryError(
            f"scheduler control plane failed after client call: {exc}",
            completed=completed,
            observation=before,
        ) from exc
    if not exact_json_equal(after, before):
        raise SchedulerBoundaryError(
            "scheduler control plane changed during client call",
            completed=completed,
            observation=before,
        )
    assert completed is not None
    return completed, before


def _fallback_scheduler_call(
    command: Sequence[str],
    cwd: Path,
    control_plane: object,
    fallback: object,
    runner: SchedulerRunner,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Use sealed original config only for exact reconciliation/cancellation."""

    binding, payload = _validated_scheduler_fallback(fallback, control_plane)
    require(
        Path(str(command[0])).name in {"squeue", "scancel"},
        "scheduler fallback is restricted to squeue/scancel",
    )
    with _scheduler_fallback_descriptor(payload) as descriptor:
        environment = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "SLURM_CONF": f"/proc/self/fd/{descriptor}",
        }
        completed = runner(command, cwd, environment, (descriptor,))
    return completed, {
        "schema_version": 1,
        "mode": "sealed_original_config_fallback",
        "sha256": binding["sha256"],
        "size": binding["size"],
        "critical": _parse_slurm_config(payload, control_plane),
    }


def _reconcile_job_ids(
    squeue: str,
    job_name: str,
    comment: str,
    cwd: Path,
    runner: SchedulerRunner,
    control_plane: object,
    *,
    fallback: object | None = None,
    expected_observation: Mapping[str, Any] | None = None,
    observations: list[dict[str, Any]] | None = None,
) -> list[str]:
    command = [squeue, "--noheader", f"--name={job_name}", "--format=%A|%j|%u|%T|%k"]
    boundary_error: str | None = None
    canonical_completed: subprocess.CompletedProcess[str] | None = None
    canonical_observation: Mapping[str, Any] | None = None
    try:
        completed, observation = _scheduler_call(
            command,
            cwd,
            control_plane,
            runner,
            expected_observation,
        )
        canonical_completed = completed
        canonical_observation = observation
    except SchedulerBoundaryError as exc:
        if fallback is None:
            raise
        boundary_error = repr(exc)
        canonical_completed = exc.completed
        canonical_observation = exc.observation
        if observations is not None and canonical_completed is not None:
            observations.append(
                {
                    "command": command,
                    "mode": "canonical_postcondition_failure",
                    "returncode": canonical_completed.returncode,
                    "stdout": canonical_completed.stdout,
                    "stderr": canonical_completed.stderr,
                    "control_plane": canonical_observation,
                    "canonical_boundary_error": boundary_error,
                }
            )
        completed, observation = _fallback_scheduler_call(
            command, cwd, control_plane, fallback, runner
        )
    if observations is not None:
        observations.append(
            {
                "command": command,
                "mode": (
                    "sealed_original_config_fallback"
                    if boundary_error is not None
                    else "authenticated_canonical"
                ),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "control_plane": observation,
                "canonical_boundary_error": boundary_error,
            }
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
    control_plane: object,
    fallback: object | None = None,
    expected_observation: Mapping[str, Any] | None = None,
    observations: list[dict[str, Any]] | None = None,
) -> None:
    require(
        not _reconcile_job_ids(
            squeue,
            job_name,
            comment,
            cwd,
            runner,
            control_plane,
            fallback=fallback,
            expected_observation=expected_observation,
            observations=observations,
        ),
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
    control_plane: object,
    fallback: object,
    expected_observation: Mapping[str, Any] | None = None,
    observations: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    try:
        completed, observation = _scheduler_call(
            command, cwd, control_plane, runner, expected_observation
        )
    except SchedulerBoundaryError as exc:
        completed = exc.completed
        parsed: list[str] = []
        if completed is not None:
            match = SBATCH_JOB.fullmatch(completed.stdout.strip())
            if match is not None:
                parsed.append(match.group("job_id"))
        reconciled: list[str] = []
        reconciliation_error: str | None = None
        try:
            reconciled = _reconcile_job_ids(
                squeue,
                job_name,
                comment,
                cwd,
                runner,
                control_plane,
                fallback=fallback,
                expected_observation=expected_observation,
                observations=observations,
            )
        except BaseException as reconcile_exc:
            reconciliation_error = repr(reconcile_exc)
        raise SchedulerSubmissionError(
            f"scheduler boundary failed for {job_name}: {exc}; "
            f"fallback reconciliation={reconciled}; error={reconciliation_error}",
            sorted(set([*parsed, *reconciled]), key=int),
        ) from exc
    if observations is not None:
        observations.append(
            {"command": list(command), "control_plane": observation}
        )
    response = completed.stdout.strip()
    match = SBATCH_JOB.fullmatch(response)
    reconciled: list[str] = []
    if completed.returncode != 0:
        # Slurm may accept a job and lose the client response.  Never continue the
        # DAG after an error response; the outer transaction reconciles and cancels.
        parsed = [match.group("job_id")] if match is not None else []
        try:
            found = _reconcile_job_ids(
                squeue,
                job_name,
                comment,
                cwd,
                runner,
                control_plane,
                fallback=fallback,
                expected_observation=expected_observation,
                observations=observations,
            )
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
        reconciled = _reconcile_job_ids(
            squeue,
            job_name,
            comment,
            cwd,
            runner,
            control_plane,
            fallback=fallback,
            expected_observation=expected_observation,
            observations=observations,
        )
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
            reconciled = _reconcile_job_ids(
                squeue,
                job_name,
                comment,
                cwd,
                runner,
                control_plane,
                fallback=fallback,
                expected_observation=expected_observation,
                observations=observations,
            )
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
        "scheduler_control_plane": observation,
    }


def _cancel_exact(
    scancel: str,
    job_ids: Sequence[str],
    cwd: Path,
    runner: SchedulerRunner,
    control_plane: object,
    fallback: object,
    expected_observation: Mapping[str, Any],
    observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    exact = sorted(set(job_ids), key=int)
    require(exact and all(JOB_ID.fullmatch(value) for value in exact), "refusing non-exact cancellation target")
    command = [scancel, *exact]
    boundary_error: str | None = None
    attempts: list[dict[str, Any]] = []
    try:
        completed, observation = _scheduler_call(
            command,
            cwd,
            control_plane,
            runner,
            expected_observation,
        )
        attempts.append(
            {
                "command": command,
                "mode": "authenticated_canonical",
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "control_plane": observation,
                "canonical_boundary_error": None,
            }
        )
    except SchedulerBoundaryError as exc:
        boundary_error = repr(exc)
        if exc.completed is not None:
            attempts.append(
                {
                    "command": command,
                    "mode": "canonical_postcondition_failure",
                    "returncode": exc.completed.returncode,
                    "stdout": exc.completed.stdout,
                    "stderr": exc.completed.stderr,
                    "control_plane": exc.observation,
                    "canonical_boundary_error": boundary_error,
                }
            )
        if exc.completed is not None and exc.completed.returncode == 0:
            if observations is not None:
                observations.extend(attempts)
            raise SchedulerBoundaryError(
                "exact cancellation response cannot be authenticated after the "
                "scheduler control plane changed",
                completed=exc.completed,
                observation=exc.observation,
            ) from exc
        else:
            completed, observation = _fallback_scheduler_call(
                command, cwd, control_plane, fallback, runner
            )
            attempts.append(
                {
                    "command": command,
                    "mode": "sealed_original_config_fallback",
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "control_plane": observation,
                    "canonical_boundary_error": boundary_error,
                }
            )
    if observations is not None:
        observations.extend(attempts)
    require(attempts, "exact cancellation has no scheduler-call evidence")
    require(completed.returncode == 0, f"partial-submission cancellation failed: {completed.stderr.strip()}")
    return {
        "job_ids": exact,
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "scheduler_control_plane": observation,
        "canonical_boundary_error": boundary_error,
        "scheduler_attempts": attempts,
    }


def _parse_controller_configuration(
    stdout: str, control_plane: object
) -> dict[str, Any]:
    contract = _scheduler_contract(control_plane)
    values: dict[str, str] = {}
    hosts: dict[int, str] = {}
    for raw_line in stdout.splitlines():
        match = re.match(r"^([^=]+?)\s*=\s*(.*?)\s*$", raw_line)
        if match is None:
            continue
        key, value = match.groups()
        key = key.strip()
        host = re.fullmatch(r"SlurmctldHost\[([0-9]+)\]", key)
        if host is not None:
            index = int(host.group(1))
            require(index not in hosts, "controller configuration duplicates a host index")
            hosts[index] = value
        elif key in {
            "AuthType",
            "CliFilterPlugins",
            "ClusterName",
            "GresTypes",
            "JobSubmitPlugins",
            "SlurmctldPort",
        }:
            require(key not in values, f"controller configuration duplicates {key}")
            values[key] = value
    require(
        values
        == {
            "AuthType": contract["auth_type"],
            "CliFilterPlugins": ",".join(contract["cli_filter_plugins"]),
            "ClusterName": contract["cluster_name"],
            "GresTypes": ",".join(contract["gres_types"]),
            "JobSubmitPlugins": ",".join(contract["job_submit_plugins"]),
            "SlurmctldPort": str(contract["slurmctld_port"]),
        },
        "controller scheduler configuration differs",
    )
    require(
        hosts == dict(enumerate(contract["slurmctld_hosts"])),
        "controller scheduler hosts differ",
    )
    return {
        "cluster_name": values["ClusterName"],
        "slurmctld_hosts": [hosts[index] for index in sorted(hosts)],
        "slurmctld_port": int(values["SlurmctldPort"]),
        "auth_type": values["AuthType"],
        "gres_types": values["GresTypes"].split(","),
        "cli_filter_plugins": values["CliFilterPlugins"].split(","),
        "job_submit_plugins": values["JobSubmitPlugins"].split(","),
    }


def _parse_sbatch_test_only(
    completed: subprocess.CompletedProcess[str],
    *,
    role: str,
    job_name: str,
    comment: str,
    output: str,
    manifest: Mapping[str, Any],
    dependency: str | None = None,
) -> dict[str, Any]:
    require(completed.returncode == 0, f"{role} sbatch --test-only failed: {completed.stderr.strip()}")
    require(not completed.stdout, f"{role} sbatch --test-only unexpectedly wrote stdout")
    lines = completed.stderr.splitlines()
    try:
        start = lines.index("sbatch: defined options")
        end = lines.index("sbatch: end of defined options", start + 1)
    except ValueError as exc:
        raise SubmissionError(f"{role} sbatch --test-only omitted its defined options") from exc
    options: dict[str, str] = {}
    for line in lines[start + 1 : end]:
        match = re.fullmatch(r"sbatch: ([a-z0-9-]+)\s+:\s+(.*)", line)
        if match is None:
            continue
        key, value = match.groups()
        require(key not in options, f"{role} sbatch --test-only duplicated option {key}")
        options[key] = value
    execution = manifest["execution"]
    expected: dict[str, str] = {
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "cpus-per-task": str(execution["cpus_per_task"]),
        "comment": comment,
        "export": "NONE",
        "job-name": job_name,
        "mem": str(execution["memory_per_task"]),
        "nodes": "1",
        "ntasks-per-node": "1",
        "open-mode": "a",
        "output": output,
        "parsable": "set",
        "partition": (
            str(execution["gpu_partitions"])
            if role in {"wave0", "wave1"}
            else str(execution["cpu_partition"])
        ),
        "qos": "normal",
        "test-only": "set",
        "time": str(execution["walltime"]),
        "verbose": "3",
    }
    if role in {"wave0", "wave1"}:
        expected.update(
            {
                "array": "0-19%20",
                "gpus-per-node": str(execution["gpus_per_task"]),
                "no-requeue": "no-requeue",
                "signal": f"B:USR1@{execution['signal_seconds_before_end']}",
            }
        )
        if role == "wave0":
            expected["hold"] = "set"
    if dependency is not None:
        require(
            role in {"wave1", "report"},
            "only dependent DAG roles may carry a scheduler-test dependency",
        )
        expected.update(
            {
                "dependency": dependency,
                "kill-on-invalid-dep": "yes",
            }
        )
    require(options == expected, f"{role} sbatch --test-only options differ")
    decisions = []
    for line in lines:
        match = re.fullmatch(
            r"sbatch: Job ([0-9]+) to start at (\S+) using ([0-9]+) processors "
            r"on nodes (\S+) in partition (\S+)",
            line,
        )
        if match is not None:
            decisions.append(match.groups())
    require(len(decisions) == 1, f"{role} sbatch --test-only decision differs")
    _synthetic_id, _start_time, processors, _nodes, partition = decisions[0]
    require(
        int(processors) == int(execution["cpus_per_task"]),
        f"{role} scheduler-test processor decision differs",
    )
    if role in {"wave0", "wave1"}:
        require(
            partition in str(execution["gpu_partitions"]).split(","),
            f"{role} scheduler-test partition decision differs",
        )
    else:
        require(partition == execution["cpu_partition"], "report scheduler-test partition differs")
    warnings = [
        line
        for line in lines
        if "warning" in line.lower() and not line.startswith("sbatch: debug")
    ]
    return {
        "role": role,
        "defined_options": options,
        "decision": {"processors": int(processors), "partition": partition},
        "warnings": warnings,
    }


def _scontrol_oneliner_field(stdout: str, name: str) -> str:
    lines = [line for line in stdout.splitlines() if line.strip()]
    require(len(lines) == 1, "scontrol returned an ambiguous job record")
    prefix = f"{name}="
    values = [
        token[len(prefix):]
        for token in lines[0].split()
        if token.startswith(prefix)
    ]
    require(len(values) == 1 and values[0], f"scontrol job field differs: {name}")
    return values[0]


def _accepted_dependency_evidence(
    *,
    scontrol: str,
    job_id: str,
    predecessor_id: str,
    job_name: str,
    role: str,
    predecessor_kind: str,
    comment: str,
    cwd: Path,
    runner: SchedulerRunner,
    control_plane: object,
    expected_observation: Mapping[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    require(role in {"wave1", "report"}, "accepted dependency role differs")
    require(
        type(predecessor_kind) is str
        and predecessor_kind in {"scalar", "array"},
        "accepted dependency predecessor kind differs",
    )
    command = [scontrol, "show", "job", job_id, "--oneliner"]
    completed, observation = _scheduler_call(
        command,
        cwd,
        control_plane,
        runner,
        expected_observation,
    )
    observations.append({"command": command, "control_plane": observation})
    require(
        completed.returncode == 0,
        f"cannot verify accepted {role} dependency: {completed.stderr.strip()}",
    )
    require(
        len(completed.stdout) <= 1024 * 1024
        and len(completed.stderr) <= 1024 * 1024,
        f"accepted {role} dependency response is oversized",
    )
    require(
        _scontrol_oneliner_field(completed.stdout, "JobId") == job_id
        and _scontrol_oneliner_field(completed.stdout, "JobName") == job_name
        and _scontrol_oneliner_field(completed.stdout, "Comment") == comment
        and _scontrol_oneliner_field(completed.stdout, "JobState") == "PENDING",
        f"accepted {role} scheduler identity differs",
    )
    dependency = _scontrol_oneliner_field(completed.stdout, "Dependency")
    expected_dependency = (
        f"afterok:{predecessor_id}(unfulfilled)"
        if predecessor_kind == "scalar"
        else f"afterok:{predecessor_id}_*(unfulfilled)"
    )
    require(
        dependency == expected_dependency,
        f"accepted {role} dependency differs",
    )
    kill_on_invalid_dependency = _scontrol_oneliner_field(
        completed.stdout, "KillOInInvalidDependent"
    )
    require(
        kill_on_invalid_dependency == "Yes",
        f"accepted {role} invalid-dependency policy differs",
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "dependency": dependency,
        "role": role,
        "kill_on_invalid_dependency": kill_on_invalid_dependency,
        "scheduler_control_plane": observation,
    }


def _accepted_wave0_hold_evidence(
    *,
    scontrol: str,
    job_id: str,
    job_name: str,
    comment: str,
    cwd: Path,
    runner: SchedulerRunner,
    control_plane: object,
    expected_observation: Mapping[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    command = [scontrol, "show", "job", job_id, "--oneliner"]
    completed, observation = _scheduler_call(
        command, cwd, control_plane, runner, expected_observation
    )
    observations.append({"command": command, "control_plane": observation})
    require(
        completed.returncode == 0
        and len(completed.stdout) <= 1024 * 1024
        and len(completed.stderr) <= 1024 * 1024,
        f"cannot verify accepted held wave zero: {completed.stderr.strip()}",
    )
    require(
        _scontrol_oneliner_field(completed.stdout, "JobId") == job_id
        and _scontrol_oneliner_field(completed.stdout, "JobName") == job_name
        and _scontrol_oneliner_field(completed.stdout, "Comment") == comment
        and _scontrol_oneliner_field(completed.stdout, "JobState") == "PENDING"
        and _scontrol_oneliner_field(completed.stdout, "Reason") == "JobHeldUser",
        "accepted wave zero is not the exact held array",
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "state": "PENDING",
        "reason": "JobHeldUser",
        "scheduler_control_plane": observation,
    }


def _release_authorized_wave0(
    *,
    scontrol: str,
    job_id: str,
    job_name: str,
    comment: str,
    cwd: Path,
    runner: SchedulerRunner,
    control_plane: object,
    expected_observation: Mapping[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    release_command = [scontrol, "release", job_id]
    released, release_observation = _scheduler_call(
        release_command, cwd, control_plane, runner, expected_observation
    )
    observations.append(
        {"command": release_command, "control_plane": release_observation}
    )
    require(
        released.returncode == 0
        and len(released.stdout) <= 1024 * 1024
        and len(released.stderr) <= 1024 * 1024,
        f"cannot release authorized wave zero: {released.stderr.strip()}",
    )
    show_command = [scontrol, "show", "job", job_id, "--oneliner"]
    shown, show_observation = _scheduler_call(
        show_command, cwd, control_plane, runner, expected_observation
    )
    observations.append({"command": show_command, "control_plane": show_observation})
    require(
        shown.returncode == 0
        and len(shown.stdout) <= 1024 * 1024
        and len(shown.stderr) <= 1024 * 1024,
        f"cannot verify released wave zero: {shown.stderr.strip()}",
    )
    state = _scontrol_oneliner_field(shown.stdout, "JobState")
    reason = _scontrol_oneliner_field(shown.stdout, "Reason")
    require(
        _scontrol_oneliner_field(shown.stdout, "JobId") == job_id
        and _scontrol_oneliner_field(shown.stdout, "JobName") == job_name
        and _scontrol_oneliner_field(shown.stdout, "Comment") == comment
        and state in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED"}
        and reason != "JobHeldUser",
        "wave zero remained held or changed identity after release",
    )
    return {
        "release_command": release_command,
        "release_returncode": released.returncode,
        "release_stdout": released.stdout,
        "release_stderr": released.stderr,
        "show_command": show_command,
        "show_returncode": shown.returncode,
        "show_stdout": shown.stdout,
        "show_stderr": shown.stderr,
        "state": state,
        "reason": reason,
        "release_scheduler_control_plane": release_observation,
        "show_scheduler_control_plane": show_observation,
    }


def _validated_wave0_release_evidence(
    value: object,
    *,
    scontrol: str,
    job_id: str,
    job_name: str,
    comment: str,
    expected_observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a durable release record before recovery trusts it."""

    require(isinstance(value, Mapping), "wave-zero release evidence is absent")
    result = dict(value)
    require(
        set(result)
        == {
            "release_command",
            "release_returncode",
            "release_stdout",
            "release_stderr",
            "show_command",
            "show_returncode",
            "show_stdout",
            "show_stderr",
            "state",
            "reason",
            "release_scheduler_control_plane",
            "show_scheduler_control_plane",
        },
        "wave-zero release evidence fields differ",
    )
    release_performed = result["release_command"] is not None
    require(
        (
            (
                result["release_command"] == [scontrol, "release", job_id]
                and type(result["release_returncode"]) is int
                and result["release_returncode"] == 0
                and isinstance(result["release_stdout"], str)
                and isinstance(result["release_stderr"], str)
                and exact_json_equal(
                    result["release_scheduler_control_plane"], expected_observation
                )
            )
            if release_performed
            else (
                result["release_command"] is None
                and result["release_returncode"] is None
                and result["release_stdout"] == ""
                and result["release_stderr"] == ""
                and result["release_scheduler_control_plane"] is None
            )
        )
        and result["show_command"]
        == [scontrol, "show", "job", job_id, "--oneliner"]
        and type(result["show_returncode"]) is int
        and result["show_returncode"] == 0
        and isinstance(result["show_stdout"], str)
        and isinstance(result["show_stderr"], str)
        and len(result["release_stdout"]) <= 1024 * 1024
        and len(result["release_stderr"]) <= 1024 * 1024
        and len(result["show_stdout"]) <= 1024 * 1024
        and len(result["show_stderr"]) <= 1024 * 1024,
        "wave-zero release command/result differs",
    )
    state = _scontrol_oneliner_field(result["show_stdout"], "JobState")
    reason = _scontrol_oneliner_field(result["show_stdout"], "Reason")
    require(
        _scontrol_oneliner_field(result["show_stdout"], "JobId") == job_id
        and _scontrol_oneliner_field(result["show_stdout"], "JobName") == job_name
        and _scontrol_oneliner_field(result["show_stdout"], "Comment") == comment
        and state == result["state"]
        and state in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED"}
        and reason == result["reason"]
        and reason != "JobHeldUser"
        and exact_json_equal(
            result["show_scheduler_control_plane"], expected_observation
        ),
        "wave-zero release identity/state differs",
    )
    return result


def _ensure_authorized_wave0_released(
    *,
    scontrol: str,
    job_id: str,
    job_name: str,
    comment: str,
    cwd: Path,
    runner: SchedulerRunner,
    control_plane: object,
    expected_observation: Mapping[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    show_command = [scontrol, "show", "job", job_id, "--oneliner"]
    shown, observation = _scheduler_call(
        show_command, cwd, control_plane, runner, expected_observation
    )
    observations.append({"command": show_command, "control_plane": observation})
    require(
        shown.returncode == 0
        and len(shown.stdout) <= 1024 * 1024
        and len(shown.stderr) <= 1024 * 1024,
        f"cannot inspect committed wave zero: {shown.stderr.strip()}",
    )
    state = _scontrol_oneliner_field(shown.stdout, "JobState")
    reason = _scontrol_oneliner_field(shown.stdout, "Reason")
    require(
        _scontrol_oneliner_field(shown.stdout, "JobId") == job_id
        and _scontrol_oneliner_field(shown.stdout, "JobName") == job_name
        and _scontrol_oneliner_field(shown.stdout, "Comment") == comment,
        "committed wave-zero identity differs",
    )
    if reason == "JobHeldUser":
        return _release_authorized_wave0(
            scontrol=scontrol,
            job_id=job_id,
            job_name=job_name,
            comment=comment,
            cwd=cwd,
            runner=runner,
            control_plane=control_plane,
            expected_observation=expected_observation,
            observations=observations,
        )
    require(
        state in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED"},
        "committed wave zero is neither held nor runnable",
    )
    return {
        "release_command": None,
        "release_returncode": None,
        "release_stdout": "",
        "release_stderr": "",
        "show_command": show_command,
        "show_returncode": shown.returncode,
        "show_stdout": shown.stdout,
        "show_stderr": shown.stderr,
        "state": state,
        "reason": reason,
        "release_scheduler_control_plane": None,
        "show_scheduler_control_plane": observation,
    }


def scheduler_preclaim_test(
    repo_root: Path,
    manifest: Mapping[str, Any],
    *,
    runner: SchedulerRunner = _default_scheduler_runner,
) -> dict[str, Any]:
    """Contact Slurm read-only and prove exact scripts survive current site policy."""

    root = _directory_nonsymlink(repo_root, "scheduler-test repository root")
    execution = manifest["execution"]
    control_plane = _scheduler_contract(execution.get("scheduler_control_plane"))
    clients = {
        "sbatch": str(execution["sbatch"]),
        "squeue": str(execution.get("squeue") or (Path(str(execution["sbatch"])).parent / "squeue")),
        "scontrol": str(execution["scontrol"]),
    }
    for label, raw_path in clients.items():
        path = Path(raw_path)
        _regular_nonsymlink(path, f"scheduler-test {label}")
        require(path.is_absolute() and os.access(path, os.X_OK), f"scheduler-test {label} is not executable")
    train_script = _contained_regular_no_symlinks(
            root, PACKAGE_RELATIVE / "train.slurm", "scheduler-test training script"
        )
    scripts = {
        "wave0": train_script,
        "wave1": train_script,
        "report": _contained_regular_no_symlinks(
            root, PACKAGE_RELATIVE / "report.slurm", "scheduler-test report script"
        ),
    }
    job_names = {
        "wave0": "exp23-launch8-scheduler-test-wave0",
        "wave1": "exp23-launch8-scheduler-test-wave1",
        "report": "exp23-launch8-scheduler-test-report",
    }
    observations: list[dict[str, Any]] = []
    commands: list[list[str]] = []

    def call(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        completed, observation = _scheduler_call(command, root, control_plane, runner)
        observations.append(observation)
        commands.append(list(command))
        return completed

    controller = call([clients["scontrol"], "show", "config"])
    require(controller.returncode == 0, f"scontrol show config failed: {controller.stderr.strip()}")
    controller_contract = _parse_controller_configuration(controller.stdout, control_plane)
    expected_user = pwd.getpwuid(os.getuid()).pw_name
    for role in ("wave0", "wave1", "report"):
        queried = call(
            [
                clients["squeue"],
                "--noheader",
                f"--name={job_names[role]}",
                f"--user={expected_user}",
                "--format=%A|%j|%u|%T|%k",
            ]
        )
        require(queried.returncode == 0, f"scheduler-test {role} pre-query failed: {queried.stderr.strip()}")
        require(not queried.stdout.strip(), f"scheduler-test name is already present: {job_names[role]}")

    decisions: dict[str, Any] = {}
    dummy_submission = root / "scheduler-test-never-executed"
    dummy_logs = dummy_submission / "logs"
    scheduler_comment = f"treewm-exp23:{'0' * 64}"
    outputs = {
        "wave0": str(dummy_logs / "wave0_%A_%a.out"),
        "wave1": str(dummy_logs / "wave1_%A_%a.out"),
        "report": str(dummy_logs / "report_%j.out"),
    }
    for role in ("wave0", "wave1", "report"):
        command = [
            clients["sbatch"],
            "-vvv",
            "--test-only",
            "--parsable",
            "--export=NONE",
            f"--job-name={job_names[role]}",
            f"--comment={scheduler_comment}",
            f"--output={outputs[role]}",
        ]
        if role in {"wave0", "wave1"}:
            command.append("--array=0-19%20")
        if role == "wave0":
            command.append("--hold")
        command.extend(
            [
                str(scripts[role]),
                str(root),
                str(dummy_submission),
                "0" * 64,
                *(["0", "none"] if role == "wave0" else (["1", "1"] if role == "wave1" else [])),
            ]
        )
        decisions[role] = _parse_sbatch_test_only(
            call(command),
            role=role,
            job_name=job_names[role],
            comment=scheduler_comment,
            output=outputs[role],
            manifest=manifest,
        )

    for role in ("wave0", "wave1", "report"):
        queried = call(
            [
                clients["squeue"],
                "--noheader",
                f"--name={job_names[role]}",
                f"--user={expected_user}",
                "--format=%A|%j|%u|%T|%k",
            ]
        )
        require(queried.returncode == 0, f"scheduler-test {role} post-query failed: {queried.stderr.strip()}")
        require(not queried.stdout.strip(), f"sbatch --test-only created a scheduler job: {job_names[role]}")
    require(
        observations
        and all(exact_json_equal(value, observations[0]) for value in observations),
        "scheduler control plane changed during preclaim test",
    )
    return {
        "schema_version": 1,
        "status": "scheduler_preclaim_verified",
        "campaign_id": manifest["campaign_id"],
        "scheduler_control_plane": observations[0],
        "controller_configuration": controller_contract,
        "sbatch_test_only": decisions,
        "dependency_tests": DEPENDENCY_TEST_REQUIREMENT,
        "scheduler_probe_commands": commands,
        "zero_job_proof": {
            "job_names": job_names,
            "pre_queries": 3,
            "post_queries": 3,
            "matching_jobs_before": 0,
            "matching_jobs_after": 0,
        },
        "scheduler_calls": len(observations),
        "scheduler_mutation_calls": 0,
        "persistent_writes_performed": 0,
    }


def _validated_scheduler_preclaim(
    value: object, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    require(isinstance(value, Mapping), "scheduler preclaim evidence is absent")
    result = dict(value)
    require(
        set(result)
        == {
            "schema_version",
            "status",
            "campaign_id",
            "scheduler_control_plane",
            "controller_configuration",
            "sbatch_test_only",
            "dependency_tests",
            "scheduler_probe_commands",
            "zero_job_proof",
            "scheduler_calls",
            "scheduler_mutation_calls",
            "persistent_writes_performed",
        },
        "scheduler preclaim evidence fields differ",
    )
    require(
        type(result["schema_version"]) is int
        and result["schema_version"] == 1
        and result["status"] == "scheduler_preclaim_verified"
        and result["campaign_id"] == manifest["campaign_id"]
        and type(result["scheduler_calls"]) is int
        and result["scheduler_calls"] == 10
        and type(result["scheduler_mutation_calls"]) is int
        and result["scheduler_mutation_calls"] == 0
        and type(result["persistent_writes_performed"]) is int
        and result["persistent_writes_performed"] == 0,
        "scheduler preclaim status differs",
    )
    control = _scheduler_contract(manifest["execution"].get("scheduler_control_plane"))
    observation = result["scheduler_control_plane"]
    require(
        isinstance(observation, Mapping)
        and type(observation.get("schema_version")) is int
        and observation.get("schema_version") == 1
        and observation.get("trust_model") == SCHEDULER_TRUST_MODEL
        and exact_json_equal(
            observation.get("critical"),
            {
                key: control[key]
                for key in (
                    "cluster_name",
                    "slurmctld_hosts",
                    "slurmctld_port",
                    "auth_type",
                    "gres_types",
                    "cli_filter_plugins",
                    "job_submit_plugins",
                )
            },
        )
        and isinstance(observation.get("config"), Mapping)
        and observation["config"].get("path") == control["slurm_conf"]
        and SHA256.fullmatch(str(observation["config"].get("sha256", ""))) is not None
        and isinstance(observation.get("cli_filter_policy"), Mapping)
        and SHA256.fullmatch(
            str(observation["cli_filter_policy"].get("tree_sha256", ""))
        )
        is not None,
        "scheduler preclaim control-plane observation differs",
    )
    require(
        exact_json_equal(
            result["controller_configuration"], observation["critical"]
        ),
        "scheduler preclaim controller/config identity differs",
    )
    decisions = result["sbatch_test_only"]
    require(
        isinstance(decisions, Mapping)
        and set(decisions) == {"wave0", "wave1", "report"}
        and all(
            isinstance(decisions[role], Mapping)
            and decisions[role].get("role") == role
            and isinstance(decisions[role].get("defined_options"), Mapping)
            and isinstance(decisions[role].get("decision"), Mapping)
            and isinstance(decisions[role].get("warnings"), list)
            for role in ("wave0", "wave1", "report")
        ),
        "scheduler preclaim sbatch evidence differs",
    )
    require(
        exact_json_equal(
            result["dependency_tests"], DEPENDENCY_TEST_REQUIREMENT
        ),
        "scheduler preclaim dependency-test requirement differs",
    )
    commands = result["scheduler_probe_commands"]
    require(
        isinstance(commands, list)
        and len(commands) == 10
        and all(
            isinstance(command, list)
            and command
            and all(isinstance(item, str) for item in command)
            for command in commands
        )
        and [Path(command[0]).name for command in commands]
        == [
            "scontrol",
            "squeue", "squeue", "squeue",
            "sbatch", "sbatch", "sbatch",
            "squeue", "squeue", "squeue",
        ]
        and all(
            Path(command[0]).name != "sbatch" or "--test-only" in command
            for command in commands
        ),
        "scheduler preclaim command ledger differs",
    )
    sbatch_commands = [
        command for command in commands if Path(command[0]).name == "sbatch"
    ]
    require(len(sbatch_commands) == 3, "scheduler preclaim sbatch command count differs")
    expected_comment = f"--comment=treewm-exp23:{'0' * 64}"
    for role, command in zip(("wave0", "wave1", "report"), sbatch_commands, strict=True):
        output_options = [item for item in command if item.startswith("--output=")]
        require(
            "-vvv" in command
            and "--test-only" in command
            and "--parsable" in command
            and "--export=NONE" in command
            and expected_comment in command
            and len(output_options) == 1
            and output_options[0].endswith(
                "scheduler-test-never-executed/logs/"
                + (f"{role}_%A_%a.out" if role != "report" else "report_%j.out")
            )
            and decisions[role]["defined_options"].get("comment")
            == expected_comment.split("=", 1)[1]
            and decisions[role]["defined_options"].get("output")
            == output_options[0].split("=", 1)[1]
            and decisions[role]["defined_options"].get("parsable") == "set"
            and decisions[role]["defined_options"].get("test-only") == "set"
            and decisions[role]["defined_options"].get("export") == "NONE"
            and decisions[role]["defined_options"].get("verbose") == "3",
            f"scheduler preclaim {role} safe-option parity differs",
        )
    require(
        all(
            "--array=0-19%20" in sbatch_commands[index]
            and decisions[role]["defined_options"].get("array") == "0-19%20"
            and decisions[role]["defined_options"].get("no-requeue") == "no-requeue"
            for index, role in enumerate(("wave0", "wave1"))
        )
        and "--hold" in sbatch_commands[0]
        and decisions["wave0"]["defined_options"].get("hold") == "set"
        and "--hold" not in sbatch_commands[1]
        and not any(item.startswith("--array=") for item in sbatch_commands[2]),
        "scheduler preclaim two-wave array parity differs",
    )
    require(
        exact_json_equal(
            result["zero_job_proof"],
            {
            "job_names": {
                "wave0": "exp23-launch8-scheduler-test-wave0",
                "wave1": "exp23-launch8-scheduler-test-wave1",
                "report": "exp23-launch8-scheduler-test-report",
            },
            "pre_queries": 3,
            "post_queries": 3,
            "matching_jobs_before": 0,
            "matching_jobs_after": 0,
            },
        ),
        "scheduler preclaim zero-job evidence differs",
    )
    return result


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
        forbidden.append(submission_root / "journal" / "0007_READY_TO_COMMIT.json")
    require(
        not any(_lexical_exists(path) for path in forbidden),
        "submission transaction acquired a terminal/cancellation marker",
    )


def _initial_exception_reconcile_and_cancel(
    *,
    exception_job_ids: Sequence[str],
    active_role: str,
    prior_claimed_ids_by_role: Mapping[str, Sequence[str]],
    role_names: Mapping[str, str],
    squeue: str,
    scancel: str,
    scheduler_comment: str,
    cancel_directory: Path,
    cancel_calling_prefix: str,
    cancel_result_prefix: str,
    snapshot_root: Path,
    scheduler_runner: SchedulerRunner,
    control_plane: object,
    scheduler_fallback: object,
    expected_observation: Mapping[str, Any],
    scheduler_observations: list[dict[str, Any]],
    cancel_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Cancel only IDs freshly proven by exact scheduler identity.

    Parsed client/error IDs and previously accepted IDs remain useful provenance
    claims, but neither is scheduler-mutation authority after an exception.  Every
    scancel target must appear in the current exact name/comment/owner census.
    """

    roles = ("wave0", "wave1", "report")
    require(
        active_role in roles
        and set(prior_claimed_ids_by_role) == set(roles)
        and set(role_names) == set(roles),
        "initial-abort role mapping differs",
    )
    prior_claims = {
        role: list(prior_claimed_ids_by_role[role]) for role in roles
    }
    require(
        all(
            all(
                isinstance(item, str) and JOB_ID.fullmatch(item) is not None
                for item in values
            )
            and values == sorted(set(values), key=int)
            for values in prior_claims.values()
        )
        and len(
            {
                item
                for values in prior_claims.values()
                for item in values
            }
        )
        == sum(len(values) for values in prior_claims.values()),
        "initial-abort prior claim IDs differ",
    )
    parsed_claims = list(exception_job_ids)
    require(
        all(
            isinstance(item, str) and JOB_ID.fullmatch(item) is not None
            for item in parsed_claims
        ),
        "initial-abort exception ID claim differs",
    )
    authority: dict[str, list[str]] = {role: [] for role in roles}
    reconciliation_errors: list[str] = []
    for role in roles:
        try:
            reconciled = _reconcile_job_ids(
                squeue,
                role_names[role],
                scheduler_comment,
                snapshot_root,
                scheduler_runner,
                control_plane,
                fallback=scheduler_fallback,
                expected_observation=expected_observation,
                observations=scheduler_observations,
            )
            authority[role].extend(reconciled)
        except BaseException as reconcile_exc:
            reconciliation_errors.append(
                f"{role_names[role]}: {reconcile_exc}"
            )
    authority_by_role = {
        role: sorted(set(values), key=int) for role, values in authority.items()
    }
    require(
        all(len(values) <= 1 for values in authority_by_role.values())
        and len(
            {
                item
                for values in authority_by_role.values()
                for item in values
            }
        )
        == sum(len(values) for values in authority_by_role.values()),
        "initial-abort fresh scheduler authority differs",
    )

    # Role attribution is globally injective.  Fresh exact-name census has priority,
    # then previously accepted identities, then untrusted parsed response claims.
    # Thus a stale wave-zero stdout ID observed while submitting wave one remains a
    # single wave-zero provenance claim instead of making the abort self-invalid.
    claimed_by_role: dict[str, list[str]] = {role: [] for role in roles}
    assigned: dict[str, str] = {}
    for role in roles:
        for job_id in authority_by_role[role]:
            assigned[job_id] = role
            claimed_by_role[role].append(job_id)
    for role in roles:
        for job_id in prior_claims[role]:
            if job_id not in assigned:
                assigned[job_id] = role
                claimed_by_role[role].append(job_id)
    for job_id in parsed_claims:
        if job_id not in assigned:
            assigned[job_id] = active_role
            claimed_by_role[active_role].append(job_id)
    claimed_by_role = {
        role: sorted(values, key=int) for role, values in claimed_by_role.items()
    }
    claimed_ids = sorted(
        {item for values in claimed_by_role.values() for item in values}, key=int
    )
    authority_ids = sorted(
        {item for values in authority_by_role.values() for item in values}, key=int
    )
    require(
        set(authority_ids).issubset(claimed_ids),
        "initial-abort cancellation authority escaped its provenance claims",
    )
    cancellation: Mapping[str, Any] | None = None
    cancellation_error: str | None = None
    cancel_history = _validated_recovery_cancel_history(
        cancel_directory,
        calling_prefix=cancel_calling_prefix,
        result_prefix=cancel_result_prefix,
        context=cancel_context,
        scancel=scancel,
        expected_control_plane=expected_observation,
        fallback=scheduler_fallback,
    )
    if authority_ids:
        try:
            cancel_history, cancellation = _append_recovery_cancel_attempt(
                cancel_directory,
                calling_prefix=cancel_calling_prefix,
                result_prefix=cancel_result_prefix,
                context=cancel_context,
                scancel=scancel,
                job_ids=authority_ids,
                cwd=snapshot_root,
                runner=scheduler_runner,
                control_plane=control_plane,
                fallback=scheduler_fallback,
                expected_observation=expected_observation,
                observations=scheduler_observations,
            )
        except BaseException as cancel_exc:
            cancellation_error = repr(cancel_exc)
            cancel_history = _validated_recovery_cancel_history(
                cancel_directory,
                calling_prefix=cancel_calling_prefix,
                result_prefix=cancel_result_prefix,
                context=cancel_context,
                scancel=scancel,
                expected_control_plane=expected_observation,
                fallback=scheduler_fallback,
            )
    return {
        "known_job_ids": claimed_ids,
        "job_ids_by_role": claimed_by_role,
        "cancellation_authority_job_ids": authority_ids,
        "cancellation_authority_job_ids_by_role": authority_by_role,
        "reconciliation_errors": reconciliation_errors,
        "cancellation": cancellation,
        "cancellation_error": cancellation_error,
        "cancel_attempt_history": cancel_history,
    }


def _validated_committed_cancel_latch(
    submission_root: Path, receipt: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Authenticate a committed-receipt cancel latch without ever following it."""

    path = submission_root / "CANCEL_REQUESTED.json"
    if not _lexical_exists(path):
        return None
    value, _latch_sha256 = read_immutable_json(
        path, "committed cancellation latch"
    )
    expected = {
        "schema_version": 1,
        "status": "cancel_requested",
        "campaign_id": receipt["campaign_id"],
        "submission_sha256": receipt["submission_sha256"],
        "wave0_array_job_id": receipt["wave0_array_job_id"],
        "wave1_array_job_id": receipt["wave1_array_job_id"],
        "report_job_id": receipt["report_job_id"],
    }
    require(
        exact_json_equal(value, expected),
        "committed cancellation latch differs",
    )
    return value


def _snapshot_preflight_in_process(
    snapshot_root: Path,
    audit_input_root: Path,
    protocol: str,
    inventory: Mapping[str, str],
    *,
    runner: AuditRunner,
) -> dict[str, Any]:
    package = snapshot_root / PACKAGE_RELATIVE
    manifest = read_json(package / "manifest.json")
    input_root = _manifest_repository_root(manifest, audit_input_root)
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
        audit_input_root=input_root,
        runner=runner,
        snapshot_inventory=inventory,
        sealed_snapshot=True,
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
    expected_rows = resolved.get("matrix")
    require(
        isinstance(expected_rows, list) and len(expected_rows) == 20,
        "snapshot resolved-config matrix differs before trainer smoke",
    )
    smoke = trainer_bootstrap_smoke(
        snapshot_root,
        inventory,
        launches[0],
        expected_rows[0],
        runner=runner,
    )
    smoke = _validated_trainer_bootstrap_smoke(
        smoke,
        inventory_sha256=stable_hash(inventory),
        launch_sha256=launches[0]["launch_sha256"],
        resolved_config_sha256=compositions[0]["resolved_config_sha256"],
        full_output_fingerprint=output_before,
        scientific_output_fingerprint=scientific_before,
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
            "audit_input_root": str(input_root),
            "trainer_bootstrap_smoke": smoke,
        },
    }


def _snapshot_preflight(
    snapshot_root: Path,
    audit_input_root: Path,
    protocol: str,
    inventory: Mapping[str, str],
    *,
    runner: AuditRunner,
) -> dict[str, Any]:
    """Run the complete copied-tree verification in a clean isolated process."""

    package = snapshot_root / PACKAGE_RELATIVE
    verify_snapshot_files(snapshot_root, inventory)
    manifest = read_json(package / "manifest.json")
    input_root = _manifest_repository_root(manifest, audit_input_root)
    identity = interpreter_contract(manifest)
    python = str(identity["lexical_executable"])
    program = package / "submit.py"
    _regular_nonsymlink(program, "snapshot submit verifier")
    command = isolated_python_command(
        [
            python,
            str(program),
            "--_snapshot-preflight",
            "--snapshot-root",
            str(snapshot_root),
            "--audit-input-root",
            str(input_root),
            "--protocol-sha256",
            protocol,
            "--inventory-json",
            canonical_json(inventory),
        ],
        snapshot_root,
        identity,
        intercept_python_children=False,
        snapshot_inventory=inventory,
    )
    with _ephemeral_child_environment() as environment:
        completed = runner(command, snapshot_root, environment, 14_400)
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
    require(
        verification.get("audit_input_root") == str(input_root),
        "isolated snapshot audit input root differs",
    )
    compositions = value.get("compositions")
    launches = value.get("launches")
    require(
        isinstance(compositions, list)
        and len(compositions) == 20
        and isinstance(launches, list)
        and len(launches) == 20,
        "isolated snapshot trainer matrices differ",
    )
    _validated_trainer_bootstrap_smoke(
        verification.get("trainer_bootstrap_smoke") or {},
        inventory_sha256=stable_hash(inventory),
        launch_sha256=str(launches[0].get("launch_sha256", "")),
        resolved_config_sha256=str(compositions[0].get("resolved_config_sha256", "")),
        full_output_fingerprint=str(
            verification.get("full_output_fingerprint_before", "")
        ),
        scientific_output_fingerprint=str(
            verification.get("scientific_output_fingerprint_before", "")
        ),
    )
    return value


def _restore_snapshot_test_permissions(task_root: Path) -> None:
    """Make only the private snapshot-test tree removable by TemporaryDirectory."""

    temporary_parent = _directory_nonsymlink(Path("/tmp"), "temporary parent")
    root = task_root.absolute()
    require(
        root.parent == temporary_parent
        and root.name.startswith(f"treewm-exp23-launch8-snapshot-test-{os.getuid()}-"),
        "refusing to restore permissions outside a snapshot-test temporary tree",
    )
    if not _lexical_exists(root):
        return
    info = root.lstat()
    require(
        stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid(),
        "snapshot-test temporary root is unsafe",
    )
    anomalies = _restore_private_tree_modes(root, "snapshot-test temporary tree")
    require(
        not anomalies,
        "snapshot-test temporary tree contains unsafe entries: " + ",".join(anomalies),
    )


def snapshot_test(
    repo_root: Path,
    *,
    runner: AuditRunner = _default_runner,
) -> dict[str, Any]:
    """Exercise the exact copied-tree preflight without touching Slurm or run state."""

    reject_inherited_environment()
    root = _directory_nonsymlink(repo_root, "repository root")
    bootstrap_manifest = read_json(root / PACKAGE_RELATIVE / "manifest.json")
    runtime_interpreter = activate_isolated_runtime(bootstrap_manifest)
    require(
        not any(name == "treewm" or name.startswith("treewm.") for name in sys.modules),
        "treewm was imported before snapshot-test containment was established",
    )
    campaign = load_campaign(root)
    manifest = campaign.read_json(root / PACKAGE_RELATIVE / "manifest.json")
    require(manifest == bootstrap_manifest, "manifest changed during snapshot-test bootstrap")
    weight_lock = campaign.read_json(root / PACKAGE_RELATIVE / "weight_audit.lock.json")
    campaign.validate_manifest(manifest, weight_lock, root)
    protocol = campaign.verify_protocol_lock(root / PACKAGE_RELATIVE)
    source_before = campaign.source_contract(root)
    require(
        source_before["source_sha256"]
        == manifest["core_binding"]["trainer_code_fingerprint"],
        "snapshot-test source fingerprint differs",
    )
    output_before = _output_tree_fingerprint(manifest)
    scientific_before = _scientific_output_fingerprint(manifest)
    inventory = snapshot_inventory(
        root, campaign, protocol, source_contract=source_before
    )

    temporary_parent = _directory_nonsymlink(Path("/tmp"), "temporary parent")
    task_root: Path | None = None
    copied: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(
        prefix=f"treewm-exp23-launch8-snapshot-test-{os.getuid()}-",
        dir=temporary_parent,
    ) as raw_task_root:
        task_root = Path(raw_task_root)
        try:
            snapshot_root = task_root / "source-snapshot/repo"
            create_source_snapshot(root, snapshot_root, inventory)
            copied = _snapshot_preflight(
                snapshot_root,
                root,
                protocol,
                inventory,
                runner=runner,
            )
            verify_snapshot_files(snapshot_root, inventory)
        finally:
            _restore_snapshot_test_permissions(task_root)
    require(
        task_root is not None and not _lexical_exists(task_root),
        "snapshot-test temporary tree survived cleanup",
    )
    require(copied is not None, "snapshot-test copied preflight is absent")

    source_after = campaign.source_contract(root)
    output_after = _output_tree_fingerprint(manifest)
    scientific_after = _scientific_output_fingerprint(manifest)
    require(source_after == source_before, "snapshot-test changed trainer source/runtime")
    require(campaign.verify_protocol_lock(root / PACKAGE_RELATIVE) == protocol, "snapshot-test changed protocol")
    require(output_after == output_before, "snapshot-test changed the output tree")
    require(scientific_after == scientific_before, "snapshot-test changed scientific outputs")
    require(copied.get("manifest") == manifest, "snapshot-test manifest differs")
    require(len(copied.get("launches") or []) == 20, "snapshot-test launch matrix differs")
    verification = copied.get("verification") or {}
    require(
        set(verification.get("audit_replays") or {})
        == {"weight", "prefix_target", "resolved_config", "causal_parity"},
        "snapshot-test did not replay all four audit locks",
    )
    require(
        verification.get("import_containment")
        == "all_treewm_modules_inside_snapshot",
        "snapshot-test import containment is absent",
    )
    require(
        verification.get("audit_input_root") == str(root),
        "snapshot-test audit input root differs",
    )
    copied_launches = copied.get("launches")
    copied_compositions = copied.get("compositions")
    require(
        isinstance(copied_launches, list)
        and len(copied_launches) == 20
        and isinstance(copied_compositions, list)
        and len(copied_compositions) == 20,
        "snapshot-test trainer matrices differ",
    )
    trainer_smoke = _validated_trainer_bootstrap_smoke(
        verification.get("trainer_bootstrap_smoke") or {},
        inventory_sha256=stable_hash(inventory),
        launch_sha256=str(copied_launches[0].get("launch_sha256", "")),
        resolved_config_sha256=str(
            copied_compositions[0].get("resolved_config_sha256", "")
        ),
        full_output_fingerprint=str(
            verification.get("full_output_fingerprint_before", "")
        ),
        scientific_output_fingerprint=str(
            verification.get("scientific_output_fingerprint_before", "")
        ),
    )
    return {
        "schema_version": 1,
        "status": "snapshot_test_verified",
        "campaign_id": manifest["campaign_id"],
        "package_protocol_sha256": protocol,
        "source_sha256": source_before["source_sha256"],
        "runtime_sha256": source_before["runtime_sha256"],
        "orchestration_interpreter": runtime_interpreter,
        "snapshot_files": len(inventory),
        "audit_replays": verification["audit_replays"],
        "cells": len(copied["launches"]),
        "import_containment": verification["import_containment"],
        "audit_input_root": verification["audit_input_root"],
        "trainer_bootstrap_smoke": trainer_smoke,
        "full_output_fingerprint_before": output_before,
        "full_output_fingerprint_after": output_after,
        "scientific_output_fingerprint_before": scientific_before,
        "scientific_output_fingerprint_after": scientific_after,
        "temporary_tree_removed": True,
        "persistent_writes_performed": 0,
        "scheduler_calls": 0,
    }


def _submission_contract(
    *,
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
    scheduler_preclaim: Mapping[str, Any],
    scheduler_fallback: Mapping[str, Any],
) -> dict[str, Any]:
    audit_bindings = {
        "weight_audit_artifact_sha256": manifest["weight_audit"]["artifact_sha256"],
        "prefix_target_artifact_sha256": manifest["prefix_target_contract"]["artifact_sha256"],
        "resolved_config_artifact_sha256": manifest["resolved_config_contract"]["artifact_sha256"],
        "causal_parity_artifact_sha256": manifest["causal_parity_contract"]["artifact_sha256"],
    }
    verified_scheduler_preclaim = _validated_scheduler_preclaim(
        scheduler_preclaim, manifest
    )
    verified_scheduler_fallback = _validated_scheduler_fallback(
        scheduler_fallback,
        manifest["execution"].get("scheduler_control_plane"),
        verified_scheduler_preclaim["scheduler_control_plane"],
    )[0]
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
        "manifest_sha256": stable_hash(manifest),
        "trainer_code_fingerprint": source["source_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "orchestration_interpreter": dict(preflight["orchestration_interpreter"]),
        "scheduler_control_plane_contract": dict(
            _scheduler_contract(manifest["execution"].get("scheduler_control_plane"))
        ),
        "scheduler_preclaim": verified_scheduler_preclaim,
        "scheduler_fallback_config": verified_scheduler_fallback,
        **audit_bindings,
        "snapshot_inventory": dict(inventory),
        "snapshot_inventory_sha256": stable_hash(inventory),
        "live_audit_replays": preflight["audit_replays"],
        "snapshot_audit_replays": snapshot_preflight["audit_replays"],
        "direct_hydra_compositions": list(compositions),
        "trainer_bootstrap_smoke": _validated_trainer_bootstrap_smoke(
            snapshot_preflight.get("trainer_bootstrap_smoke") or {},
            inventory_sha256=stable_hash(inventory),
            launch_sha256=rows[0]["launch_sha256"],
            resolved_config_sha256=str(compositions[0]["resolved_config_sha256"]),
            full_output_fingerprint=str(
                snapshot_preflight["full_output_fingerprint_before"]
            ),
            scientific_output_fingerprint=str(
                snapshot_preflight["scientific_output_fingerprint_before"]
            ),
        ),
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


def _validated_preflight_inventory(preflight: Mapping[str, Any]) -> dict[str, str]:
    """Return only the exact inventory frozen by the successful static preflight."""

    raw = preflight.get("snapshot_inventory")
    require(isinstance(raw, Mapping) and raw, "preflight snapshot inventory is absent")
    inventory: dict[str, str] = {}
    for raw_relative, raw_digest in raw.items():
        relative = str(_safe_relative(raw_relative, "preflight snapshot path"))
        digest = str(raw_digest)
        require(SHA256.fullmatch(digest) is not None, "preflight snapshot SHA256 is malformed")
        require(relative not in inventory, "preflight snapshot path is duplicated")
        inventory[relative] = digest
    inventory = dict(sorted(inventory.items()))
    require(
        stable_hash(inventory) == preflight.get("snapshot_inventory_sha256"),
        "preflight snapshot inventory hash differs",
    )
    source_contract = preflight.get("source_contract")
    require(isinstance(source_contract, Mapping), "preflight source contract is absent")
    source_files = source_contract.get("source_files")
    require(isinstance(source_files, Mapping) and source_files, "preflight source files are absent")
    for raw_relative, raw_digest in source_files.items():
        relative = str(_safe_relative(raw_relative, "preflight source path"))
        require(
            inventory.get(relative) == str(raw_digest),
            f"preflight source is not bound into snapshot inventory: {relative}",
        )
    for relative in (
        PACKAGE_RELATIVE / "manifest.json",
        PACKAGE_RELATIVE / "submit.py",
        PACKAGE_RELATIVE / "protocol.sha256",
    ):
        require(str(relative) in inventory, f"preflight snapshot omits {relative}")
    return inventory


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
    inventory = _validated_preflight_inventory(preflight)
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
    scheduler_preclaim = _validated_scheduler_preclaim(
        preflight.get("scheduler_preclaim"), manifest
    )
    require(_namespace_is_fresh(manifest, submission_root), "scheduler preclaim changed the submission namespace")
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
        snapshot_root, root, protocol, inventory, runner=audit_runner
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
    scheduler_fallback = _scheduler_fallback_config(
        manifest["execution"].get("scheduler_control_plane")
    )
    contract = _submission_contract(
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
        scheduler_preclaim=scheduler_preclaim,
        scheduler_fallback=scheduler_fallback,
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
    control_plane = _scheduler_contract(execution.get("scheduler_control_plane"))
    submission_authorization = scheduler_preclaim["scheduler_control_plane"]
    scheduler_fallback = _validated_scheduler_fallback(
        contract.get("scheduler_fallback_config"),
        control_plane,
        submission_authorization,
    )[0]
    sbatch = str(execution["sbatch"])
    scancel = str(execution["scancel"])
    scontrol = str(execution["scontrol"])
    squeue = str(execution.get("squeue") or (Path(sbatch).parent / "squeue"))
    for path, label in (
        (sbatch, "sbatch"),
        (scancel, "scancel"),
        (scontrol, "scontrol"),
        (squeue, "squeue"),
    ):
        _regular_nonsymlink(Path(path), label)
        require(os.access(path, os.X_OK), f"{label} is not executable")
    token = submission_sha256[:16]
    wave0_name = f"exp23-launch8-{token}-wave0"
    wave1_name = f"exp23-launch8-{token}-wave1"
    report_name = f"exp23-launch8-{token}-report"
    scheduler_comment = f"treewm-exp23:{submission_sha256}"
    train_script = snapshot_root / PACKAGE_RELATIVE / "train.slurm"
    report_script = snapshot_root / PACKAGE_RELATIVE / "report.slurm"
    _regular_nonsymlink(train_script, "snapshot training Slurm")
    _regular_nonsymlink(report_script, "snapshot report Slurm")
    wave0_command = [
        sbatch,
        "--parsable",
        "--export=NONE",
        "--hold",
        "--array=0-19%20",
        f"--job-name={wave0_name}",
        f"--comment={scheduler_comment}",
        f"--output={logs / 'wave0_%A_%a.out'}",
        str(train_script),
        str(snapshot_root),
        str(submission_root),
        submission_sha256,
        "0",
        "none",
    ]
    known_ids: list[str] = []
    known_ids_by_role: dict[str, list[str]] = {
        "wave0": [],
        "wave1": [],
        "report": [],
    }
    active_role = "wave0"
    ready_to_commit = False
    records: dict[str, Any] = {}
    scheduler_observations: list[dict[str, Any]] = []
    transaction_lock_binding = _leased_transaction_lock_binding(scheduler_runner)
    try:
        _assert_live_transaction(submission_root)
        for name in (wave0_name, wave1_name, report_name):
            _assert_job_absent(
                squeue,
                name,
                scheduler_comment,
                snapshot_root,
                scheduler_runner,
                control_plane,
                scheduler_fallback,
                submission_authorization,
                scheduler_observations,
            )
        wave0_calling_sha256 = exclusive_json(
            submission_root / "journal" / "CALLING_WAVE0.json",
            {
                "schema_version": 1,
                "status": "scheduler_calling",
                "campaign_id": manifest["campaign_id"],
                "submission_sha256": submission_sha256,
                "claim_token": claim_token,
                "role": "wave0",
                "job_name": wave0_name,
                "scheduler_comment": scheduler_comment,
                "command": wave0_command,
                "transaction_lock": transaction_lock_binding,
            },
        )
        wave0_id, wave0_record = _submit_one(
            wave0_command,
            job_name=wave0_name,
            comment=scheduler_comment,
            squeue=squeue,
            cwd=snapshot_root,
            runner=scheduler_runner,
            control_plane=control_plane,
            fallback=scheduler_fallback,
            expected_observation=submission_authorization,
            observations=scheduler_observations,
        )
        known_ids.append(wave0_id)
        known_ids_by_role["wave0"].append(wave0_id)
        wave0_hold = _accepted_wave0_hold_evidence(
            scontrol=scontrol,
            job_id=wave0_id,
            job_name=wave0_name,
            comment=scheduler_comment,
            cwd=snapshot_root,
            runner=scheduler_runner,
            control_plane=control_plane,
            expected_observation=submission_authorization,
            observations=scheduler_observations,
        )
        wave0_record = {
            **wave0_record,
            "calling_sha256": wave0_calling_sha256,
            "accepted_hold": wave0_hold,
        }
        records["wave0"] = wave0_record
        append_journal(
            submission_root, 3, "WAVE0_SUBMITTED", {"job_id": wave0_id, **wave0_record}
        )
        _assert_live_transaction(submission_root)
        wave1_command = [
            sbatch,
            "--parsable",
            "--export=NONE",
            "--array=0-19%20",
            f"--dependency=afterok:{wave0_id}",
            "--kill-on-invalid-dep=yes",
            f"--job-name={wave1_name}",
            f"--comment={scheduler_comment}",
            f"--output={logs / 'wave1_%A_%a.out'}",
            str(train_script),
            str(snapshot_root),
            str(submission_root),
            submission_sha256,
            "1",
            wave0_id,
        ]
        active_role = "wave1"
        wave1_test_command = [
            sbatch,
            "-vvv",
            "--test-only",
            "--parsable",
            "--export=NONE",
            "--array=0-19%20",
            f"--dependency=afterok:{wave0_id}",
            "--kill-on-invalid-dep=yes",
            f"--job-name={wave1_name}",
            f"--comment={scheduler_comment}",
            f"--output={logs / 'wave1_%A_%a.out'}",
            str(train_script),
            str(snapshot_root),
            str(submission_root),
            submission_sha256,
            "1",
            wave0_id,
        ]
        wave1_test_completed, wave1_test_observation = _scheduler_call(
            wave1_test_command,
            snapshot_root,
            control_plane,
            scheduler_runner,
            submission_authorization,
        )
        scheduler_observations.append(
            {
                "command": wave1_test_command,
                "control_plane": wave1_test_observation,
            }
        )
        parsed_wave1_test = _parse_sbatch_test_only(
            wave1_test_completed,
            role="wave1",
            job_name=wave1_name,
            comment=scheduler_comment,
            output=str(logs / "wave1_%A_%a.out"),
            manifest=manifest,
            dependency=f"afterok:{wave0_id}",
        )
        _assert_job_absent(
            squeue,
            wave1_name,
            scheduler_comment,
            snapshot_root,
            scheduler_runner,
            control_plane,
            scheduler_fallback,
            submission_authorization,
            scheduler_observations,
        )
        wave1_test_evidence = {
            "command": wave1_test_command,
            "returncode": wave1_test_completed.returncode,
            "stdout": wave1_test_completed.stdout,
            "stderr": wave1_test_completed.stderr,
            "parsed": parsed_wave1_test,
            "scheduler_control_plane": wave1_test_observation,
            "zero_job_after_test": True,
        }
        wave1_calling_sha256 = exclusive_json(
            submission_root / "journal" / "CALLING_WAVE1.json",
            {
                "schema_version": 1,
                "status": "scheduler_calling",
                "campaign_id": manifest["campaign_id"],
                "submission_sha256": submission_sha256,
                "claim_token": claim_token,
                "role": "wave1",
                "job_name": wave1_name,
                "scheduler_comment": scheduler_comment,
                "command": wave1_command,
                "transaction_lock": transaction_lock_binding,
            },
        )
        wave1_id, wave1_record = _submit_one(
            wave1_command,
            job_name=wave1_name,
            comment=scheduler_comment,
            squeue=squeue,
            cwd=snapshot_root,
            runner=scheduler_runner,
            control_plane=control_plane,
            fallback=scheduler_fallback,
            expected_observation=submission_authorization,
            observations=scheduler_observations,
        )
        known_ids.append(wave1_id)
        known_ids_by_role["wave1"].append(wave1_id)
        wave1_dependency_evidence = _accepted_dependency_evidence(
            scontrol=scontrol,
            job_id=wave1_id,
            predecessor_id=wave0_id,
            job_name=wave1_name,
            role="wave1",
            predecessor_kind="array",
            comment=scheduler_comment,
            cwd=snapshot_root,
            runner=scheduler_runner,
            control_plane=control_plane,
            expected_observation=submission_authorization,
            observations=scheduler_observations,
        )
        wave1_record = {
            **wave1_record,
            "calling_sha256": wave1_calling_sha256,
            "exact_dependency_test_only": wave1_test_evidence,
            "accepted_dependency": wave1_dependency_evidence,
        }
        records["wave1"] = wave1_record
        append_journal(
            submission_root,
            4,
            "WAVE1_SUBMITTED",
            {"job_id": wave1_id, **wave1_record},
        )
        _assert_live_transaction(submission_root)
        report_command = [
            sbatch,
            "--parsable",
            "--export=NONE",
            f"--dependency=afterok:{wave1_id}",
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
        report_test_command = [
            sbatch,
            "-vvv",
            "--test-only",
            "--parsable",
            "--export=NONE",
            f"--dependency=afterok:{wave1_id}",
            "--kill-on-invalid-dep=yes",
            f"--job-name={report_name}",
            f"--comment={scheduler_comment}",
            f"--output={logs / 'report_%j.out'}",
            str(report_script),
            str(snapshot_root),
            str(submission_root),
            submission_sha256,
        ]
        report_test_completed, report_test_observation = _scheduler_call(
            report_test_command,
            snapshot_root,
            control_plane,
            scheduler_runner,
            submission_authorization,
        )
        scheduler_observations.append(
            {"command": report_test_command, "control_plane": report_test_observation}
        )
        parsed_report_test = _parse_sbatch_test_only(
            report_test_completed,
            role="report",
            job_name=report_name,
            comment=scheduler_comment,
            output=str(logs / "report_%j.out"),
            manifest=manifest,
            dependency=f"afterok:{wave1_id}",
        )
        _assert_job_absent(
            squeue,
            report_name,
            scheduler_comment,
            snapshot_root,
            scheduler_runner,
            control_plane,
            scheduler_fallback,
            submission_authorization,
            scheduler_observations,
        )
        report_test_evidence = {
            "command": report_test_command,
            "returncode": report_test_completed.returncode,
            "stdout": report_test_completed.stdout,
            "stderr": report_test_completed.stderr,
            "parsed": parsed_report_test,
            "scheduler_control_plane": report_test_observation,
            "zero_job_after_test": True,
        }
        report_calling_sha256 = exclusive_json(
            submission_root / "journal" / "CALLING_REPORT.json",
            {
                "schema_version": 1,
                "status": "scheduler_calling",
                "campaign_id": manifest["campaign_id"],
                "submission_sha256": submission_sha256,
                "claim_token": claim_token,
                "role": "report",
                "job_name": report_name,
                "scheduler_comment": scheduler_comment,
                "command": report_command,
                "transaction_lock": transaction_lock_binding,
            },
        )
        report_id, report_record = _submit_one(
            report_command,
            job_name=report_name,
            comment=scheduler_comment,
            squeue=squeue,
            cwd=snapshot_root,
            runner=scheduler_runner,
            control_plane=control_plane,
            fallback=scheduler_fallback,
            expected_observation=submission_authorization,
            observations=scheduler_observations,
        )
        known_ids.append(report_id)
        known_ids_by_role["report"].append(report_id)
        report_dependency_evidence = _accepted_dependency_evidence(
            scontrol=scontrol,
            job_id=report_id,
            predecessor_id=wave1_id,
            job_name=report_name,
            role="report",
            predecessor_kind="array",
            comment=scheduler_comment,
            cwd=snapshot_root,
            runner=scheduler_runner,
            control_plane=control_plane,
            expected_observation=submission_authorization,
            observations=scheduler_observations,
        )
        report_record = {
            **report_record,
            "calling_sha256": report_calling_sha256,
            "exact_dependency_test_only": report_test_evidence,
            "accepted_dependency": report_dependency_evidence,
        }
        records["report"] = report_record
        append_journal(
            submission_root, 5, "REPORT_SUBMITTED", {"job_id": report_id, **report_record}
        )
        _assert_live_transaction(submission_root)
        dag_evidence = load_dag_evidence(snapshot_root)
        try:
            accepted_job_evidence_sha256 = dag_evidence.validate_dag_records(
                records,
                {
                    "wave0": read_json(
                        submission_root / "journal" / "CALLING_WAVE0.json"
                    ),
                    "wave1": read_json(
                        submission_root / "journal" / "CALLING_WAVE1.json"
                    ),
                    "report": read_json(
                        submission_root / "journal" / "CALLING_REPORT.json"
                    ),
                },
                {
                    "wave0": file_sha256(
                        submission_root / "journal" / "CALLING_WAVE0.json"
                    ),
                    "wave1": file_sha256(
                        submission_root / "journal" / "CALLING_WAVE1.json"
                    ),
                    "report": file_sha256(
                        submission_root / "journal" / "CALLING_REPORT.json"
                    ),
                },
                manifest=manifest,
                snapshot_root=snapshot_root,
                submission_root=submission_root,
                submission_sha256=submission_sha256,
                claim_token=claim_token,
                job_ids={
                    "wave0": wave0_id,
                    "wave1": wave1_id,
                    "report": report_id,
                },
                expected_control_plane=submission_authorization,
                expected_lock=transaction_lock_binding,
            )
        except BaseException as evidence_exc:
            raise SubmissionError(
                f"accepted DAG scheduler evidence differs: {evidence_exc}"
            ) from evidence_exc
        authorization = {
            "schema_version": 1,
            "status": "authorized_two_wave_dag",
            "campaign_id": manifest["campaign_id"],
            "submission_sha256": submission_sha256,
            "array": "0-19%20",
            "job_ids": {"wave0": wave0_id, "wave1": wave1_id, "report": report_id},
            "dependencies": {
                "wave0": "none",
                "wave1": f"afterok:{wave0_id}",
                "report": f"afterok:{wave1_id}",
            },
            "kill_on_invalid_dependency": {"wave1": "yes", "report": "yes"},
            "within_wave_requeue": False,
            "wave0_submitted_held": True,
            "accepted_job_evidence_sha256": accepted_job_evidence_sha256,
            "authorized_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        require(set(authorization) == AUTHORIZATION_FIELDS, "DAG authorization fields differ")
        authorization_sha256 = exclusive_json(
            submission_root / "SUBMISSION_AUTHORIZATION.json", authorization
        )
        append_journal(
            submission_root,
            6,
            "DAG_AUTHORIZED",
            {
                "submission_authorization_sha256": authorization_sha256,
                "accepted_job_evidence_sha256": accepted_job_evidence_sha256,
                "job_ids": authorization["job_ids"],
                "dependencies": authorization["dependencies"],
            },
        )
        receipt = {
            "schema_version": 1,
            "status": "committed_two_wave_dag",
            "campaign_id": manifest["campaign_id"],
            "submission_sha256": submission_sha256,
            "submission_authorization_sha256": authorization_sha256,
            "array": "0-19%20",
            "wave0_array_job_id": wave0_id,
            "wave1_array_job_id": wave1_id,
            "report_job_id": report_id,
            "wave1_dependency": f"afterok:{wave0_id}",
            "report_dependency": f"afterok:{wave1_id}",
            "kill_on_invalid_dependency": {"wave1": "yes", "report": "yes"},
            "within_wave_requeue": False,
            "wave0_submitted_held": True,
        }
        require(set(receipt) == RECEIPT_FIELDS, "DAG receipt fields differ")
        append_journal(submission_root, 7, "READY_TO_COMMIT", receipt)
        ready_to_commit = True
        _assert_live_transaction(submission_root, ready=True)
        exclusive_json(submission_root / "SUBMISSION_RECEIPT.json", receipt)
        require(
            _validated_committed_cancel_latch(submission_root, receipt) is None,
            "durable cancellation forbids wave-zero release",
        )
        release_calling_sha256 = exclusive_json(
            submission_root / "journal" / "CALLING_WAVE0_RELEASE.json",
            {
                "schema_version": 1,
                "status": "scheduler_calling",
                "campaign_id": manifest["campaign_id"],
                "submission_sha256": submission_sha256,
                "claim_token": claim_token,
                "role": "wave0_release",
                "job_name": wave0_name,
                "scheduler_comment": scheduler_comment,
                "command": [scontrol, "release", wave0_id],
                "transaction_lock": transaction_lock_binding,
            },
        )
        release_evidence = _release_authorized_wave0(
            scontrol=scontrol,
            job_id=wave0_id,
            job_name=wave0_name,
            comment=scheduler_comment,
            cwd=snapshot_root,
            runner=scheduler_runner,
            control_plane=control_plane,
            expected_observation=submission_authorization,
            observations=scheduler_observations,
        )
        _validated_wave0_release_evidence(
            release_evidence,
            scontrol=scontrol,
            job_id=wave0_id,
            job_name=wave0_name,
            comment=scheduler_comment,
            expected_observation=submission_authorization,
        )
        append_journal(
            submission_root,
            8,
            "WAVE0_RELEASED",
            {
                "wave0_array_job_id": wave0_id,
                "submission_authorization_sha256": authorization_sha256,
                "calling_sha256": release_calling_sha256,
                "release_evidence": release_evidence,
            },
        )
        return {
            **receipt,
            "wave0_release": "verified",
            "scheduler_calls": len(scheduler_observations)
            + int(scheduler_preclaim["scheduler_calls"]),
            "scheduler_mutation_calls": 4,
            "snapshot_files": len(inventory),
        }
    except BaseException as exc:
        if ready_to_commit:
            raise CommitRecoveryRequired(
                f"receipt commit failed after durable READY_TO_COMMIT: {exc}",
                sorted(set(known_ids), key=int),
            ) from exc
        cancel_context = {
            "schema_version": 1,
            "status": "scheduler_calling",
            "campaign_id": CAMPAIGN_ID,
            "submission_sha256": submission_sha256,
            "claim_token": claim_token,
            "role": "recovery_cancel",
            "transaction_lock": transaction_lock_binding,
        }
        abort_evidence = _initial_exception_reconcile_and_cancel(
            exception_job_ids=getattr(exc, "job_ids", ()),
            active_role=active_role,
            prior_claimed_ids_by_role=known_ids_by_role,
            role_names={
                "wave0": wave0_name,
                "wave1": wave1_name,
                "report": report_name,
            },
            squeue=squeue,
            scancel=scancel,
            scheduler_comment=scheduler_comment,
            cancel_directory=submission_root / "journal",
            cancel_calling_prefix="CALLING_RECOVERY_CANCEL",
            cancel_result_prefix="RECOVERY_CANCEL_RESULT",
            snapshot_root=snapshot_root,
            scheduler_runner=scheduler_runner,
            control_plane=control_plane,
            scheduler_fallback=scheduler_fallback,
            expected_observation=submission_authorization,
            scheduler_observations=scheduler_observations,
            cancel_context=cancel_context,
        )
        try:
            append_journal(
                submission_root,
                9999,
                "ABORTED",
                {
                    "error": repr(exc),
                    "known_job_ids": abort_evidence["known_job_ids"],
                    "job_ids_by_role": abort_evidence["job_ids_by_role"],
                    "cancellation_authority_job_ids": abort_evidence[
                        "cancellation_authority_job_ids"
                    ],
                    "cancellation_authority_job_ids_by_role": abort_evidence[
                        "cancellation_authority_job_ids_by_role"
                    ],
                    "submission_sha256": submission_sha256,
                    "reconciliation_errors": abort_evidence[
                        "reconciliation_errors"
                    ],
                    "cancellation": abort_evidence["cancellation"],
                    "cancellation_error": abort_evidence["cancellation_error"],
                    "cancel_attempt_history": abort_evidence[
                        "cancel_attempt_history"
                    ],
                    "scheduler_control_plane_observations": scheduler_observations,
                },
            )
        except BaseException:
            pass
        detail = repr(exc)
        if abort_evidence["reconciliation_errors"]:
            detail += "; reconciliation=" + "; ".join(
                abort_evidence["reconciliation_errors"]
            )
        if abort_evidence["cancellation_error"]:
            detail += "; cancellation=" + abort_evidence["cancellation_error"]
        raise SchedulerSubmissionError(
            f"submission transaction aborted: {detail}",
            abort_evidence["known_job_ids"],
            abort_evidence["job_ids_by_role"],
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
        if _lexical_exists(claimed_path):
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
                role_ids: dict[str, list[str]] = {}
                for role in ("wave0", "wave1", "report"):
                    raw_role_ids = list(role_values.get(role, ()))
                    require(
                        all(
                            isinstance(item, str)
                            and JOB_ID.fullmatch(item) is not None
                            for item in raw_role_ids
                        ),
                        f"outer abort {role} ID claims differ",
                    )
                    role_ids[role] = sorted(set(raw_role_ids), key=int)
                contract_path = submission_root / "SUBMISSION_CONTRACT.json"
                contract_sha256 = (
                    file_sha256(contract_path)
                    if _lexical_exists(contract_path)
                    and stat.S_ISREG(contract_path.lstat().st_mode)
                    else None
                )
                value = {
                    "schema_version": 1,
                    "record": "outer_aborted",
                    "error": repr(exc),
                    "receipt_committed": _lexical_exists(receipt),
                    "known_job_ids": list(getattr(exc, "job_ids", ())),
                    "job_ids_by_role": role_ids,
                    "claim_token": claim_token,
                    "submission_sha256": contract_sha256,
                }
                if _lexical_exists(abort_path):
                    require(
                        exact_json_equal(read_json(abort_path), value),
                        "outer abort journal differs",
                    )
                elif not _lexical_exists(receipt):
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

    manifest = preflight.get("manifest")
    require(isinstance(manifest, Mapping), "submission preflight manifest is absent")
    scheduler_preclaim = scheduler_preclaim_test(
        repo_root,
        manifest,
        runner=scheduler_runner,
    )
    require(
        _namespace_is_fresh(manifest, submission_root),
        "scheduler preclaim changed the submission namespace",
    )
    verified_preflight = {**dict(preflight), "scheduler_preclaim": scheduler_preclaim}
    with _TransactionLock(submission_root) as transaction_lock:
        leased_scheduler_runner = _scheduler_runner_with_lock_lease(
            scheduler_runner, transaction_lock
        )
        return _submit_campaign_locked(
            repo_root,
            submission_root,
            verified_preflight,
            audit_runner=audit_runner,
            scheduler_runner=leased_scheduler_runner,
        )


def _receipt_from_ready_record(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "submission_authorization_sha256",
        "array",
        "wave0_array_job_id",
        "wave1_array_job_id",
        "report_job_id",
        "wave1_dependency",
        "report_dependency",
        "kill_on_invalid_dependency",
        "within_wave_requeue",
        "wave0_submitted_held",
    )
    require(value.get("record") == "ready_to_commit", "recovery journal is not ready-to-commit")
    require(set(value) == RECEIPT_FIELDS | {"record"}, "recovery READY schema differs")
    receipt = {key: value[key] for key in keys}
    require(
        type(receipt["schema_version"]) is int
        and receipt["schema_version"] == 1
        and receipt["status"] == "committed_two_wave_dag",
        "recovery receipt state differs",
    )
    ids = [
        receipt[key]
        for key in ("wave0_array_job_id", "wave1_array_job_id", "report_job_id")
    ]
    require(
        all(
            isinstance(value, str) and JOB_ID.fullmatch(value) is not None
            for value in ids
        ),
        "recovery job IDs differ",
    )
    require(len(set(ids)) == 3, "recovery job IDs are not distinct")
    require(receipt["campaign_id"] == CAMPAIGN_ID and receipt["array"] == "0-19%20", "recovery campaign/array differs")
    require(
        SHA256.fullmatch(str(receipt["submission_authorization_sha256"])) is not None
        and receipt["wave1_dependency"] == f"afterok:{ids[0]}"
        and receipt["report_dependency"] == f"afterok:{ids[1]}"
        and exact_json_equal(
            receipt["kill_on_invalid_dependency"],
            {"wave1": "yes", "report": "yes"},
        )
        and receipt["within_wave_requeue"] is False
        and receipt["wave0_submitted_held"] is True,
        "recovery DAG receipt differs",
    )
    return receipt


def _validated_dag_authorization(
    submission_root: Path,
    submission_sha256: str,
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    path = submission_root / "SUBMISSION_AUTHORIZATION.json"
    authorization, authorization_sha256 = read_immutable_json(
        path, "submission DAG authorization"
    )
    require(set(authorization) == AUTHORIZATION_FIELDS, "DAG authorization schema differs")
    job_ids = authorization.get("job_ids")
    dependencies = authorization.get("dependencies")
    require(
        type(authorization.get("schema_version")) is int
        and authorization.get("schema_version") == 1
        and authorization.get("status") == "authorized_two_wave_dag"
        and authorization.get("campaign_id") == CAMPAIGN_ID
        and authorization.get("submission_sha256") == submission_sha256
        and authorization.get("array") == "0-19%20"
        and authorization.get("within_wave_requeue") is False
        and authorization.get("wave0_submitted_held") is True
        and exact_json_equal(
            authorization.get("kill_on_invalid_dependency"),
            {"wave1": "yes", "report": "yes"},
        )
        and SHA256.fullmatch(str(authorization.get("accepted_job_evidence_sha256", "")))
        is not None
        and isinstance(authorization.get("authorized_at_utc"), str)
        and bool(authorization["authorized_at_utc"]),
        "DAG authorization contract differs",
    )
    require(
        isinstance(job_ids, Mapping)
        and set(job_ids) == {"wave0", "wave1", "report"}
        and all(
            isinstance(job_ids[role], str)
            and JOB_ID.fullmatch(job_ids[role]) is not None
            for role in job_ids
        )
        and len(set(job_ids.values())) == 3,
        "DAG authorization job IDs differ",
    )
    require(
        exact_json_equal(
            dependencies,
            {
            "wave0": "none",
            "wave1": f"afterok:{job_ids['wave0']}",
            "report": f"afterok:{job_ids['wave1']}",
            },
        ),
        "DAG authorization dependencies differ",
    )
    receipt_expected = {
        "submission_authorization_sha256": authorization_sha256,
        "wave0_array_job_id": job_ids["wave0"],
        "wave1_array_job_id": job_ids["wave1"],
        "report_job_id": job_ids["report"],
        "wave1_dependency": dependencies["wave1"],
        "report_dependency": dependencies["report"],
    }
    for key, expected in receipt_expected.items():
        require(receipt.get(key) == expected, f"receipt/authorization differs: {key}")
    records: dict[str, Any] = {}
    for role, ordinal in (("wave0", 3), ("wave1", 4), ("report", 5)):
        journal, _journal_sha256 = read_immutable_json(
            submission_root
            / "journal"
            / f"{ordinal:04d}_{role.upper()}_SUBMITTED.json",
            f"DAG {role} submission journal",
        )
        require(
            type(journal.get("schema_version")) is int
            and journal.get("schema_version") == 1
            and journal.get("record") == f"{role}_submitted"
            and isinstance(journal.get("job_id"), str)
            and journal.get("job_id") == job_ids[role],
            f"DAG {role} submission journal differs",
        )
        records[role] = {
            key: value
            for key, value in journal.items()
            if key not in {"schema_version", "record", "job_id"}
        }
    contract, _contract_sha256 = read_immutable_json(
        submission_root / "SUBMISSION_CONTRACT.json", "submission contract"
    )
    snapshot_root = Path(str(contract.get("snapshot_root", "")))
    manifest = read_json(snapshot_root / PACKAGE_RELATIVE / "manifest.json")
    claim, _claim_sha256 = read_immutable_json(
        submission_root / "journal" / "0000_CLAIMED.json",
        "DAG transaction claim",
    )
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
        and type(claim.get("schema_version")) is int
        and claim.get("schema_version") == 1
        and claim.get("record") == "claimed"
        and claim.get("campaign_id") == CAMPAIGN_ID
        and claim.get("submission_root") == str(submission_root)
        and SHA256.fullmatch(str(claim.get("claim_token", ""))) is not None
        and SHA256.fullmatch(
            str(claim.get("scientific_output_fingerprint", ""))
        )
        is not None,
        "DAG transaction claim differs",
    )
    calling_records = {}
    calling_sha256_by_role = {}
    for role in ("wave0", "wave1", "report"):
        calling_records[role], calling_sha256_by_role[role] = read_immutable_json(
            submission_root / "journal" / f"CALLING_{role.upper()}.json",
            f"DAG {role} scheduler calling intent",
        )
    dag_evidence = load_dag_evidence(snapshot_root)
    try:
        semantic_hash = dag_evidence.validate_dag_records(
            records,
            calling_records,
            calling_sha256_by_role,
            manifest=manifest,
            snapshot_root=snapshot_root,
            submission_root=submission_root,
            submission_sha256=submission_sha256,
            claim_token=str(claim.get("claim_token", "")),
            job_ids={role: job_ids[role] for role in job_ids},
            expected_control_plane=contract["scheduler_preclaim"][
                "scheduler_control_plane"
            ],
        )
    except BaseException as evidence_exc:
        raise SubmissionError(
            f"DAG accepted-job semantic evidence differs: {evidence_exc}"
        ) from evidence_exc
    require(
        semantic_hash == authorization["accepted_job_evidence_sha256"],
        "DAG accepted-job evidence hash differs",
    )
    authorized, _authorized_sha256 = read_immutable_json(
        submission_root / "journal" / "0006_DAG_AUTHORIZED.json",
        "durable DAG authorization journal",
    )
    require(
        exact_json_equal(
            authorized,
            {
            "schema_version": 1,
            "record": "dag_authorized",
            "submission_authorization_sha256": authorization_sha256,
            "accepted_job_evidence_sha256": authorization[
                "accepted_job_evidence_sha256"
            ],
            "job_ids": dict(job_ids),
            "dependencies": dict(dependencies),
            },
        ),
        "durable DAG authorization journal differs",
    )
    return dict(authorization), authorization_sha256


def _validated_abort_role_ids(
    value: Mapping[str, Any],
    label: str,
    *,
    role_key: str = "job_ids_by_role",
    flat_key: str = "known_job_ids",
) -> dict[str, list[str]]:
    raw_roles = value.get(role_key)
    require(
        isinstance(raw_roles, Mapping)
        and set(raw_roles) == {"wave0", "wave1", "report"},
        f"{label} role-ID mapping differs",
    )
    result: dict[str, list[str]] = {}
    for role in ("wave0", "wave1", "report"):
        raw_values = raw_roles[role]
        require(isinstance(raw_values, list), f"{label} {role} IDs are not a list")
        require(
            all(
                isinstance(item, str) and JOB_ID.fullmatch(item) is not None
                for item in raw_values
            ),
            f"{label} {role} IDs differ",
        )
        values = list(raw_values)
        require(
            values == sorted(set(values), key=int),
            f"{label} {role} IDs differ",
        )
        result[role] = values
    require(
        not set(result["wave0"]).intersection(result["wave1"])
        and not set(result["wave0"]).intersection(result["report"])
        and not set(result["wave1"]).intersection(result["report"]),
        f"{label} assigns one scheduler ID to both roles",
    )
    raw_flat = value.get(flat_key)
    require(isinstance(raw_flat, list), f"{label} flat IDs are not a list")
    require(
        all(
            isinstance(item, str) and JOB_ID.fullmatch(item) is not None
            for item in raw_flat
        ),
        f"{label} flat IDs differ",
    )
    flat = list(raw_flat)
    expected = sorted(
        set(result["wave0"] + result["wave1"] + result["report"]), key=int
    )
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
        and set(cancellation)
        == {
            "job_ids",
            "command",
            "stdout",
            "stderr",
            "scheduler_control_plane",
            "canonical_boundary_error",
            "scheduler_attempts",
        },
        "abort cancellation evidence schema differs",
    )
    raw_ids = cancellation["job_ids"]
    require(isinstance(raw_ids, list), "abort cancellation IDs are not a list")
    require(
        all(
            isinstance(item, str) and JOB_ID.fullmatch(item) is not None
            for item in raw_ids
        ),
        "abort cancellation IDs differ",
    )
    ids = list(raw_ids)
    require(
        ids
        and ids == sorted(set(ids), key=int),
        "abort cancellation IDs differ",
    )
    require(
        all(
            isinstance(item, str) and JOB_ID.fullmatch(item) is not None
            for item in known_ids
        )
        and ids == sorted(set(known_ids), key=int),
        "abort successful cancellation does not cover every known ID",
    )
    require(cancellation["command"] == [scancel, *ids], "abort cancellation command differs")
    require(
        isinstance(cancellation["stdout"], str) and isinstance(cancellation["stderr"], str),
        "abort cancellation output types differ",
    )
    require(
        isinstance(cancellation["scheduler_control_plane"], Mapping)
        and isinstance(cancellation["scheduler_attempts"], list)
        and bool(cancellation["scheduler_attempts"])
        and (
            cancellation["canonical_boundary_error"] is None
            or isinstance(cancellation["canonical_boundary_error"], str)
        ),
        "abort cancellation scheduler evidence differs",
    )
    return set(ids)


def _recovery_census_rounds(
    *,
    squeue: str,
    role_names: Mapping[str, str],
    comment: str,
    cwd: Path,
    runner: SchedulerRunner,
    control_plane: object,
    fallback: object,
    expected_observation: Mapping[str, Any],
    observations: list[dict[str, Any]],
    rounds: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Require two consecutive identical exact-name censuses after client quiescence."""

    require(rounds >= 3, "recovery census round count is too small")
    evidence: list[dict[str, Any]] = []
    for round_index in range(rounds):
        ids_by_role: dict[str, list[str]] = {}
        spans_by_role: dict[str, dict[str, int]] = {}
        for role in ("wave0", "wave1", "report"):
            attempt_start = len(observations)
            values = _reconcile_job_ids(
                squeue,
                role_names[role],
                comment,
                cwd,
                runner,
                control_plane,
                fallback=fallback,
                expected_observation=expected_observation,
                observations=observations,
            )
            require(
                len(values) <= 1,
                f"recovery found multiple exact scheduler identities for {role}",
            )
            attempt_stop = len(observations)
            require(
                attempt_stop > attempt_start,
                f"recovery {role} census did not preserve its scheduler attempt",
            )
            ids_by_role[role] = values
            spans_by_role[role] = {
                "start": attempt_start,
                "stop": attempt_stop,
            }
        require(
            len({item for values in ids_by_role.values() for item in values})
            == sum(len(values) for values in ids_by_role.values()),
            "recovery census assigns one scheduler ID to multiple roles",
        )
        evidence.append(
            {
                "round": round_index,
                "job_ids_by_role": ids_by_role,
                "scheduler_attempt_spans_by_role": spans_by_role,
            }
        )
        if round_index + 1 < rounds:
            time.sleep(0.25)
    require(
        evidence[-2]["job_ids_by_role"] == evidence[-1]["job_ids_by_role"],
        "recovery scheduler identity did not settle across the final censuses",
    )
    return evidence, dict(evidence[-1]["job_ids_by_role"])


def _expected_fallback_attempt_control_plane(
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    source = fallback.get("source_control_plane")
    require(
        type(fallback.get("schema_version")) is int
        and fallback.get("schema_version") == 1
        and SHA256.fullmatch(str(fallback.get("sha256", ""))) is not None
        and type(fallback.get("size")) is int
        and fallback["size"] > 0
        and isinstance(source, Mapping)
        and isinstance(source.get("critical"), Mapping),
        "scheduler fallback attempt binding differs",
    )
    return {
        "schema_version": 1,
        "mode": "sealed_original_config_fallback",
        "sha256": fallback["sha256"],
        "size": fallback["size"],
        "critical": dict(source["critical"]),
    }


def _validated_scheduler_attempt_ledger(
    value: object,
    *,
    expected_control_plane: Mapping[str, Any],
    fallback: Mapping[str, Any],
) -> list[dict[str, Any]]:
    require(isinstance(value, list), "recovery scheduler-attempt ledger is absent")
    require(
        isinstance(expected_control_plane, Mapping),
        "scheduler attempt canonical control-plane binding is absent",
    )
    fallback_control_plane: dict[str, Any] | None = None
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        require(
            isinstance(raw, Mapping)
            and set(raw)
            == {
                "command",
                "mode",
                "returncode",
                "stdout",
                "stderr",
                "control_plane",
                "canonical_boundary_error",
            }
            and isinstance(raw.get("command"), list)
            and bool(raw["command"])
            and all(isinstance(item, str) for item in raw["command"])
            and raw.get("mode")
            in {
                "authenticated_canonical",
                "canonical_postcondition_failure",
                "sealed_original_config_fallback",
            }
            and type(raw.get("returncode")) is int
            and isinstance(raw.get("stdout"), str)
            and isinstance(raw.get("stderr"), str)
            and isinstance(raw.get("control_plane"), Mapping)
            and (
                raw.get("canonical_boundary_error") is None
                or isinstance(raw["canonical_boundary_error"], str)
            ),
            f"recovery scheduler attempt {index} differs",
        )
        mode = str(raw["mode"])
        if mode == "authenticated_canonical":
            require(
                exact_json_equal(raw["control_plane"], expected_control_plane)
                and raw["canonical_boundary_error"] is None,
                f"recovery scheduler attempt {index} canonical binding differs",
            )
        elif mode == "canonical_postcondition_failure":
            require(
                exact_json_equal(raw["control_plane"], expected_control_plane)
                and isinstance(raw["canonical_boundary_error"], str)
                and bool(raw["canonical_boundary_error"]),
                f"recovery scheduler attempt {index} failed-call binding differs",
            )
        else:
            if fallback_control_plane is None:
                fallback_control_plane = _expected_fallback_attempt_control_plane(
                    fallback
                )
            require(
                exact_json_equal(raw["control_plane"], fallback_control_plane)
                and isinstance(raw["canonical_boundary_error"], str)
                and bool(raw["canonical_boundary_error"]),
                f"recovery scheduler attempt {index} fallback binding differs",
            )
        result.append(dict(raw))
    return result


def _validated_recovery_cancel_history(
    directory: Path,
    *,
    calling_prefix: str,
    result_prefix: str,
    context: Mapping[str, Any],
    scancel: str,
    expected_control_plane: Mapping[str, Any],
    fallback: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate an append-only sequence of exact cancellation call/result pairs."""

    _directory_nonsymlink(directory, "recovery cancellation journal directory")
    calling_pattern = re.compile(rf"{re.escape(calling_prefix)}_([0-9]{{4}})\.json")
    result_pattern = re.compile(rf"{re.escape(result_prefix)}_([0-9]{{4}})\.json")
    calling_indices: set[int] = set()
    result_indices: set[int] = set()
    for entry in os.scandir(directory):
        match = calling_pattern.fullmatch(entry.name)
        if match is not None:
            calling_indices.add(int(match.group(1)))
        match = result_pattern.fullmatch(entry.name)
        if match is not None:
            result_indices.add(int(match.group(1)))
    require(
        calling_indices == set(range(len(calling_indices)))
        and result_indices.issubset(calling_indices),
        "recovery cancellation attempts are not an append-only prefix",
    )
    history: list[dict[str, Any]] = []
    for index in range(len(calling_indices)):
        calling_name = f"{calling_prefix}_{index:04d}.json"
        calling_path = directory / calling_name
        _regular_nonsymlink(calling_path, f"recovery cancellation intent {index}")
        require(
            stat.S_IMODE(calling_path.lstat().st_mode) == 0o444,
            f"recovery cancellation intent {index} mode differs",
        )
        calling = read_json(calling_path)
        raw_ids = calling.get("job_ids")
        require(
            isinstance(raw_ids, list)
            and raw_ids
            and all(
                isinstance(item, str) and JOB_ID.fullmatch(item) is not None
                for item in raw_ids
            )
            and raw_ids == sorted(set(raw_ids), key=int),
            f"recovery cancellation intent {index} IDs differ",
        )
        job_ids = list(raw_ids)
        require(
            exact_json_equal(
                calling,
                {
                **dict(context),
                "attempt_index": index,
                "job_ids": job_ids,
                "command": [scancel, *job_ids],
                },
            ),
            f"recovery cancellation intent {index} differs",
        )
        calling_sha256 = file_sha256(calling_path)
        result_name: str | None = None
        result_sha256: str | None = None
        cancellation: dict[str, Any] | None = None
        if index in result_indices:
            result_name = f"{result_prefix}_{index:04d}.json"
            result_path = directory / result_name
            _regular_nonsymlink(result_path, f"recovery cancellation result {index}")
            require(
                stat.S_IMODE(result_path.lstat().st_mode) == 0o444,
                f"recovery cancellation result {index} mode differs",
            )
            result = read_json(result_path)
            require(
                set(result)
                == {
                    "schema_version",
                    "status",
                    "attempt_index",
                    "calling_sha256",
                    "cancellation",
                }
                and type(result.get("schema_version")) is int
                and result.get("schema_version") == 1
                and result.get("status") == "scheduler_cancel_completed"
                and type(result.get("attempt_index")) is int
                and result.get("attempt_index") == index
                and result.get("calling_sha256") == calling_sha256
                and isinstance(result.get("cancellation"), Mapping),
                f"recovery cancellation result {index} differs",
            )
            raw_cancellation = result["cancellation"]
            require(
                set(raw_cancellation)
                == {
                    "job_ids",
                    "command",
                    "stdout",
                    "stderr",
                    "scheduler_control_plane",
                    "canonical_boundary_error",
                    "scheduler_attempts",
                }
                and raw_cancellation.get("job_ids") == job_ids
                and raw_cancellation.get("command") == [scancel, *job_ids]
                and isinstance(raw_cancellation.get("stdout"), str)
                and isinstance(raw_cancellation.get("stderr"), str)
                and isinstance(raw_cancellation.get("scheduler_control_plane"), Mapping)
                and (
                    raw_cancellation.get("canonical_boundary_error") is None
                    or isinstance(raw_cancellation["canonical_boundary_error"], str)
                ),
                f"recovery cancellation result {index} evidence differs",
            )
            scheduler_attempts = _validated_scheduler_attempt_ledger(
                raw_cancellation.get("scheduler_attempts"),
                expected_control_plane=expected_control_plane,
                fallback=fallback,
            )
            require(
                scheduler_attempts
                and all(
                    attempt["command"] == [scancel, *job_ids]
                    for attempt in scheduler_attempts
                )
                and scheduler_attempts[-1]["returncode"] == 0,
                f"recovery cancellation result {index} scheduler attempts differ",
            )
            require(
                (
                    len(scheduler_attempts) == 1
                    and scheduler_attempts[0]["mode"]
                    in {"authenticated_canonical", "sealed_original_config_fallback"}
                )
                or (
                    len(scheduler_attempts) == 2
                    and scheduler_attempts[0]["mode"]
                    == "canonical_postcondition_failure"
                    and scheduler_attempts[0]["returncode"] != 0
                    and scheduler_attempts[1]["mode"]
                    == "sealed_original_config_fallback"
                ),
                f"recovery cancellation result {index} attempt lineage differs",
            )
            terminal_attempt = scheduler_attempts[-1]
            require(
                raw_cancellation["stdout"] == terminal_attempt["stdout"]
                and raw_cancellation["stderr"] == terminal_attempt["stderr"]
                and exact_json_equal(
                    raw_cancellation["scheduler_control_plane"],
                    terminal_attempt["control_plane"],
                )
                and raw_cancellation["canonical_boundary_error"]
                == terminal_attempt["canonical_boundary_error"],
                f"recovery cancellation result {index} terminal summary differs",
            )
            cancellation = dict(raw_cancellation)
            result_sha256 = file_sha256(result_path)
        history.append(
            {
                "attempt_index": index,
                "calling_name": calling_name,
                "calling_sha256": calling_sha256,
                "job_ids": job_ids,
                "result_name": result_name,
                "result_sha256": result_sha256,
                "cancellation": cancellation,
            }
        )
    return history


def _append_recovery_cancel_attempt(
    directory: Path,
    *,
    calling_prefix: str,
    result_prefix: str,
    context: Mapping[str, Any],
    scancel: str,
    job_ids: Sequence[str],
    cwd: Path,
    runner: SchedulerRunner,
    control_plane: object,
    fallback: object,
    expected_observation: Mapping[str, Any],
    observations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    history = _validated_recovery_cancel_history(
        directory,
        calling_prefix=calling_prefix,
        result_prefix=result_prefix,
        context=context,
        scancel=scancel,
        expected_control_plane=expected_observation,
        fallback=fallback,
    )
    index = len(history)
    require(
        all(
            isinstance(item, str) and JOB_ID.fullmatch(item) is not None
            for item in job_ids
        ),
        "recovery cancellation IDs differ",
    )
    exact = sorted(set(job_ids), key=int)
    require(exact, "recovery cancellation attempt has no exact target")
    calling_path = directory / f"{calling_prefix}_{index:04d}.json"
    calling_sha256 = exclusive_json(
        calling_path,
        {
            **dict(context),
            "attempt_index": index,
            "job_ids": exact,
            "command": [scancel, *exact],
        },
    )
    cancellation = _cancel_exact(
        scancel,
        exact,
        cwd,
        runner,
        control_plane,
        fallback,
        expected_observation,
        observations,
    )
    exclusive_json(
        directory / f"{result_prefix}_{index:04d}.json",
        {
            "schema_version": 1,
            "status": "scheduler_cancel_completed",
            "attempt_index": index,
            "calling_sha256": calling_sha256,
            "cancellation": cancellation,
        },
    )
    validated = _validated_recovery_cancel_history(
        directory,
        calling_prefix=calling_prefix,
        result_prefix=result_prefix,
        context=context,
        scancel=scancel,
        expected_control_plane=expected_observation,
        fallback=fallback,
    )
    require(
        len(validated) == index + 1
        and validated[-1]["cancellation"] == cancellation,
        "recovery cancellation attempt did not seal exactly",
    )
    return validated, cancellation


def _validated_recovery_census_rounds(
    value: object,
    *,
    attempts: Sequence[Mapping[str, Any]],
    role_names: Mapping[str, str],
    squeue: str,
    comment: str,
    label: str,
    expected_start: int,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], int]:
    """Reconstruct every stored census value from its exact successful squeue row."""

    require(isinstance(value, list) and len(value) == 3, f"{label} census differs")
    expected_user = pwd.getpwuid(os.getuid()).pw_name
    cursor = expected_start
    normalized_rounds: list[dict[str, Any]] = []
    for round_index, raw_round in enumerate(value):
        require(
            isinstance(raw_round, Mapping)
            and set(raw_round)
            == {
                "round",
                "job_ids_by_role",
                "scheduler_attempt_spans_by_role",
            }
            and type(raw_round.get("round")) is int
            and raw_round.get("round") == round_index,
            f"{label} census round {round_index} differs",
        )
        raw_ids = raw_round["job_ids_by_role"]
        raw_spans = raw_round["scheduler_attempt_spans_by_role"]
        require(
            isinstance(raw_ids, Mapping)
            and isinstance(raw_spans, Mapping)
            and set(raw_ids) == set(raw_spans) == {"wave0", "wave1", "report"},
            f"{label} census role maps differ",
        )
        ids_by_role: dict[str, list[str]] = {}
        spans_by_role: dict[str, dict[str, int]] = {}
        for role in ("wave0", "wave1", "report"):
            span = raw_spans[role]
            require(
                isinstance(span, Mapping)
                and set(span) == {"start", "stop"}
                and type(span.get("start")) is int
                and type(span.get("stop")) is int
                and span["start"] == cursor
                and span["start"] < span["stop"] <= len(attempts),
                f"{label} {role} scheduler-attempt span differs",
            )
            selected = list(attempts[span["start"] : span["stop"]])
            require(
                len(selected) in {1, 2},
                f"{label} {role} scheduler-attempt cardinality differs",
            )
            expected_command = [
                squeue,
                "--noheader",
                f"--name={role_names[role]}",
                "--format=%A|%j|%u|%T|%k",
            ]
            require(
                all(attempt.get("command") == expected_command for attempt in selected)
                and (
                    (
                        len(selected) == 1
                        and selected[0].get("mode")
                        in {"authenticated_canonical", "sealed_original_config_fallback"}
                    )
                    or (
                        len(selected) == 2
                        and selected[0].get("mode")
                        == "canonical_postcondition_failure"
                        and selected[1].get("mode")
                        == "sealed_original_config_fallback"
                    )
                ),
                f"{label} {role} scheduler-attempt lineage differs",
            )
            terminal = selected[-1]
            require(
                terminal.get("returncode") == 0,
                f"{label} {role} scheduler census did not succeed",
            )
            parsed: set[str] = set()
            for raw_line in str(terminal.get("stdout", "")).splitlines():
                if not raw_line.strip():
                    continue
                fields = [field.strip() for field in raw_line.strip().split("|", 4)]
                require(
                    len(fields) == 5,
                    f"{label} {role} scheduler census output is malformed",
                )
                job_id, actual_name, user, state, actual_comment = fields
                require(
                    JOB_ID.fullmatch(job_id) is not None
                    and actual_name == role_names[role]
                    and user == expected_user
                    and bool(state)
                    and actual_comment == comment,
                    f"{label} {role} scheduler census identity differs",
                )
                parsed.add(job_id)
            normalized_ids = sorted(parsed, key=int)
            require(
                len(normalized_ids) <= 1 and raw_ids[role] == normalized_ids,
                f"{label} {role} census map was not derived from scheduler stdout",
            )
            ids_by_role[role] = normalized_ids
            spans_by_role[role] = {"start": span["start"], "stop": span["stop"]}
            cursor = span["stop"]
        require(
            len({item for values in ids_by_role.values() for item in values})
            == sum(len(values) for values in ids_by_role.values()),
            f"{label} census assigns one scheduler ID to multiple roles",
        )
        normalized_rounds.append(
            {
                "round": round_index,
                "job_ids_by_role": ids_by_role,
                "scheduler_attempt_spans_by_role": spans_by_role,
            }
        )
    require(
        normalized_rounds[-2]["job_ids_by_role"]
        == normalized_rounds[-1]["job_ids_by_role"],
        f"{label} scheduler identity did not settle",
    )
    return normalized_rounds, dict(normalized_rounds[-1]["job_ids_by_role"]), cursor


def _validated_recovery_cancel_latch(
    value: object,
    *,
    submission_sha256: str,
    claim_token: str,
    transaction_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the historical recovery latch before any scheduler call."""

    require(isinstance(value, Mapping), "recovery cancellation latch is absent")
    result = dict(value)
    require(
        set(result)
        == {
            "schema_version",
            "status",
            "campaign_id",
            "submission_sha256",
            "claim_token",
            "wave0_array_job_id",
            "wave1_array_job_id",
            "report_job_id",
            "job_ids_by_role",
            "transaction_lock",
            "recovery",
            "scheduler_id_authority",
        }
        and type(result.get("schema_version")) is int
        and result.get("schema_version") == 1
        and result.get("status") == "cancel_requested"
        and result.get("campaign_id") == CAMPAIGN_ID
        and result.get("submission_sha256") == submission_sha256
        and result.get("claim_token") == claim_token
        and exact_json_equal(result.get("transaction_lock"), transaction_lock)
        and result.get("recovery") is True
        and result.get("scheduler_id_authority")
        == "fresh_settled_exact_name_census_only",
        "recovery cancellation latch identity differs",
    )
    raw_roles = result.get("job_ids_by_role")
    require(
        isinstance(raw_roles, Mapping)
        and set(raw_roles) == {"wave0", "wave1", "report"},
        "recovery cancellation latch role map differs",
    )
    roles: dict[str, list[str]] = {}
    for role in ("wave0", "wave1", "report"):
        raw = raw_roles[role]
        require(isinstance(raw, list), f"recovery cancellation latch {role} differs")
        require(
            all(
                isinstance(item, str) and JOB_ID.fullmatch(item) is not None
                for item in raw
            ),
            f"recovery cancellation latch {role} differs",
        )
        normalized = list(raw)
        require(
            normalized == sorted(set(normalized), key=int)
            and len(normalized) <= 1,
            f"recovery cancellation latch {role} differs",
        )
        roles[role] = normalized
    require(
        len({item for values in roles.values() for item in values})
        == sum(len(values) for values in roles.values()),
        "recovery cancellation latch assigns one scheduler ID to multiple roles",
    )
    for role, scalar_key in (
        ("wave0", "wave0_array_job_id"),
        ("wave1", "wave1_array_job_id"),
        ("report", "report_job_id"),
    ):
        require(
            result.get(scalar_key) == (roles[role][0] if roles[role] else None),
            f"recovery cancellation latch historical {role} fields differ",
        )
    return result


def _validated_recovery_terminal_record(
    value: Mapping[str, Any],
    *,
    submission_root: Path,
    submission_sha256: str,
    claim_token: str,
    package_protocol_sha256: str,
    role_names: Mapping[str, str],
    comment: str,
    squeue: str,
    scancel: str,
    transaction_lock: Mapping[str, Any],
    expected_control_plane: Mapping[str, Any],
    scheduler_fallback: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "record",
        "submission_sha256",
        "submission_root",
        "claim_token",
        "package_protocol_sha256",
        "status",
        "durable_claimed_job_ids",
        "durable_claimed_job_ids_by_role",
        "live_verified_job_ids",
        "live_verified_job_ids_by_role",
        "calling_intent_sha256_by_role",
        "pre_cancel_census_rounds",
        "cancelled_live_job_ids",
        "cancel_latch_sha256",
        "cancel_calling_sha256",
        "cancellation",
        "cancel_attempt_history",
        "post_cancel_census_rounds",
        "post_cancel_active_job_ids_by_role",
        "scheduler_control_plane_observations",
        "scheduler_calls",
        "transaction_lock",
        "new_jobs_created",
    }
    require(set(value) == expected_fields, "recovery terminal record fields differ")
    require(
        type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("record") == "recovery_cancelled"
        and value.get("submission_sha256") == submission_sha256
        and value.get("submission_root") == str(submission_root)
        and value.get("claim_token") == claim_token
        and value.get("package_protocol_sha256") == package_protocol_sha256
        and type(value.get("new_jobs_created")) is int
        and value.get("new_jobs_created") == 0
        and exact_json_equal(value.get("transaction_lock"), transaction_lock),
        "recovery terminal identity differs",
    )

    def role_map(
        raw: object,
        label: str,
        *,
        require_single_current: bool,
    ) -> dict[str, list[str]]:
        require(
            isinstance(raw, Mapping)
            and set(raw) == {"wave0", "wave1", "report"},
            f"{label} role map differs",
        )
        normalized: dict[str, list[str]] = {}
        for role in ("wave0", "wave1", "report"):
            values = raw[role]
            require(isinstance(values, list), f"{label} {role} IDs differ")
            require(
                all(
                    isinstance(item, str) and JOB_ID.fullmatch(item) is not None
                    for item in values
                ),
                f"{label} {role} IDs differ",
            )
            converted = list(values)
            require(
                converted == sorted(set(converted), key=int)
                and (not require_single_current or len(converted) <= 1),
                f"{label} {role} IDs differ",
            )
            normalized[role] = converted
        require(
            len({item for values in normalized.values() for item in values})
            == sum(len(values) for values in normalized.values()),
            f"{label} assigns one scheduler ID to multiple roles",
        )
        return normalized

    durable = role_map(
        value.get("durable_claimed_job_ids_by_role"),
        "durable claimed",
        require_single_current=False,
    )
    live = role_map(
        value.get("live_verified_job_ids_by_role"),
        "live verified",
        require_single_current=True,
    )
    post = role_map(
        value.get("post_cancel_active_job_ids_by_role"),
        "post-cancel active",
        require_single_current=True,
    )
    durable_flat = sorted({item for ids in durable.values() for item in ids}, key=int)
    live_flat = sorted({item for ids in live.values() for item in ids}, key=int)
    require(
        value.get("durable_claimed_job_ids") == durable_flat
        and value.get("live_verified_job_ids") == live_flat,
        "recovery flat/role job-ID evidence differs",
    )
    calling = value.get("calling_intent_sha256_by_role")
    require(
        isinstance(calling, Mapping)
        and set(calling).issubset({"wave0", "wave1", "report"})
        and all(
            isinstance(digest, str) and SHA256.fullmatch(digest) is not None
            for digest in calling.values()
        ),
        "recovery calling-intent hashes differ",
    )
    attempts = _validated_scheduler_attempt_ledger(
        value.get("scheduler_control_plane_observations"),
        expected_control_plane=expected_control_plane,
        fallback=scheduler_fallback,
    )
    pre_rounds, reconstructed_live, attempt_cursor = (
        _validated_recovery_census_rounds(
            value.get("pre_cancel_census_rounds"),
            attempts=attempts,
            role_names=role_names,
            squeue=squeue,
            comment=comment,
            label="recovery pre-cancel",
            expected_start=0,
        )
    )
    require(
        exact_json_equal(reconstructed_live, live),
        "recovery pre-cancel census differs",
    )
    latch_path = submission_root / "CANCEL_REQUESTED.json"
    _regular_nonsymlink(latch_path, "recovery cancellation latch")
    require(
        stat.S_IMODE(latch_path.lstat().st_mode) == 0o444
        and value.get("cancel_latch_sha256") == file_sha256(latch_path),
        "recovery cancellation latch bytes differ",
    )
    _validated_recovery_cancel_latch(
        read_json(latch_path),
        submission_sha256=submission_sha256,
        claim_token=claim_token,
        transaction_lock=transaction_lock,
    )
    # The latch linearizes cancellation precedence.  Its IDs are a historical
    # observation only: delayed scheduler visibility may make a later settled
    # exact-name census nonempty or different.  No scheduler mutation is ever
    # authorized from these latch fields.
    cancel_context = {
        "schema_version": 1,
        "status": "scheduler_calling",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "claim_token": claim_token,
        "role": "recovery_cancel",
        "transaction_lock": dict(transaction_lock),
    }
    disk_history = _validated_recovery_cancel_history(
        submission_root / "journal",
        calling_prefix="CALLING_RECOVERY_CANCEL",
        result_prefix="RECOVERY_CANCEL_RESULT",
        context=cancel_context,
        scancel=scancel,
        expected_control_plane=expected_control_plane,
        fallback=scheduler_fallback,
    )
    stored_history = value.get("cancel_attempt_history")
    require(
        isinstance(stored_history, list)
        and len(stored_history) <= len(disk_history)
        and exact_json_equal(stored_history, disk_history[: len(stored_history)]),
        "recovery cancellation attempt history differs",
    )
    last_attempt = stored_history[-1] if stored_history else None
    require(
        value.get("cancel_calling_sha256")
        == (last_attempt["calling_sha256"] if last_attempt else None)
        and exact_json_equal(
            value.get("cancellation"),
            last_attempt["cancellation"] if last_attempt else None,
        ),
        "recovery terminal cancellation mirrors differ",
    )
    cancelled = value.get("cancelled_live_job_ids")
    require(cancelled == live_flat, "recovery cancellation targets differ")
    cancellation = value.get("cancellation")
    if live_flat:
        require(
            last_attempt is not None
            and last_attempt["job_ids"] == live_flat
            and last_attempt["cancellation"] is not None,
            "recovery cancellation calling record differs",
        )
        require(
            value.get("status") == "recovered_terminal_after_cancel_attempts"
            and isinstance(value.get("cancel_calling_sha256"), str)
            and SHA256.fullmatch(value["cancel_calling_sha256"]) is not None
            and isinstance(cancellation, Mapping)
            and cancellation.get("job_ids") == live_flat
            and cancellation.get("command") == [scancel, *live_flat]
            and exact_json_equal(
                cancellation.get("scheduler_attempts"),
                [
                    row
                    for row in value["scheduler_control_plane_observations"]
                    if row.get("command") == [scancel, *live_flat]
                ],
            ),
            "recovery exact cancellation evidence differs",
        )
        cancellation_attempts = _validated_scheduler_attempt_ledger(
            cancellation.get("scheduler_attempts"),
            expected_control_plane=expected_control_plane,
            fallback=scheduler_fallback,
        )
        require(
            exact_json_equal(
                attempts[
                    attempt_cursor : attempt_cursor + len(cancellation_attempts)
                ],
                cancellation_attempts,
            ),
            "recovery cancellation attempts are not in the scheduler ledger",
        )
        attempt_cursor += len(cancellation_attempts)
        post_rounds, reconstructed_post, attempt_cursor = (
            _validated_recovery_census_rounds(
                value.get("post_cancel_census_rounds"),
                attempts=attempts,
                role_names=role_names,
                squeue=squeue,
                comment=comment,
                label="recovery post-cancel",
                expected_start=attempt_cursor,
            )
        )
        require(
            exact_json_equal(reconstructed_post, post)
            and exact_json_equal(
                post, {"wave0": [], "wave1": [], "report": []}
            ),
            "recovery post-cancel terminal census differs",
        )
    else:
        require(
            value.get("status")
            == (
                "recovered_terminal_after_cancel_attempts"
                if stored_history
                else "recovered_terminal_no_active_jobs"
            )
            and value.get("post_cancel_census_rounds") == []
            and exact_json_equal(
                post, {"wave0": [], "wave1": [], "report": []}
            ),
            "recovery no-active terminal evidence differs",
        )
    require(
        attempt_cursor == len(attempts)
        and type(value.get("scheduler_calls")) is int
        and value.get("scheduler_calls") == len(attempts),
        "recovery scheduler-call count differs",
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
            for attempt in attempts
        ),
        "recovery scheduler-attempt command escaped exact identities",
    )
    return dict(value)


def _validated_residual_recovery_chain(
    directory: Path,
    *,
    filename_prefix: str,
    initial_terminal_path: Path,
    initial_cancel_history_length: int,
    identity: Mapping[str, Any],
    status_prefix: str,
    cancel_directory: Path,
    cancel_calling_prefix: str,
    cancel_result_prefix: str,
    cancel_context: Mapping[str, Any],
    role_names: Mapping[str, str],
    comment: str,
    squeue: str,
    scancel: str,
    expected_control_plane: Mapping[str, Any],
    scheduler_fallback: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate gap-free immutable residual recovery generations.

    The original terminal record remains generation zero.  A later exact-name job
    can reappear only through an append-only generation that binds the previous
    terminal's raw bytes, a fresh census, the exact cancellation history delta,
    and a terminal empty census.  This avoids both overwriting old evidence and a
    permanent dead end on delayed scheduler visibility.
    """

    _directory_nonsymlink(directory, "residual recovery journal directory")
    _regular_nonsymlink(initial_terminal_path, "initial recovery terminal")
    require(
        stat.S_IMODE(initial_terminal_path.lstat().st_mode) == 0o444
        and type(initial_cancel_history_length) is int
        and initial_cancel_history_length >= 0,
        "initial recovery terminal binding differs",
    )
    pattern = re.compile(
        rf"{re.escape(filename_prefix)}_([0-9]{{4}})\.json"
    )
    indices: set[int] = set()
    for entry in os.scandir(directory):
        match = pattern.fullmatch(entry.name)
        if match is not None:
            indices.add(int(match.group(1)))
    require(
        indices == set(range(len(indices))),
        "residual recovery generations are not an append-only prefix",
    )
    disk_history = _validated_recovery_cancel_history(
        cancel_directory,
        calling_prefix=cancel_calling_prefix,
        result_prefix=cancel_result_prefix,
        context=cancel_context,
        scancel=scancel,
        expected_control_plane=expected_control_plane,
        fallback=scheduler_fallback,
    )
    previous_name = initial_terminal_path.name
    previous_sha256 = file_sha256(initial_terminal_path)
    previous_history_length = initial_cancel_history_length
    expected_fields = {
        *identity,
        "schema_version",
        "record",
        "generation",
        "previous_terminal_name",
        "previous_terminal_sha256",
        "status",
        "live_verified_job_ids",
        "live_verified_job_ids_by_role",
        "pre_cancel_census_rounds",
        "cancelled_live_job_ids",
        "cancel_history_length_before",
        "cancel_attempt_history",
        "cancel_calling_sha256",
        "cancellation",
        "post_cancel_census_rounds",
        "post_cancel_active_job_ids_by_role",
        "scheduler_control_plane_observations",
        "scheduler_calls",
        "new_jobs_created",
    }

    def role_map(raw: object, label: str) -> dict[str, list[str]]:
        require(
            isinstance(raw, Mapping)
            and set(raw) == {"wave0", "wave1", "report"},
            f"{label} role map differs",
        )
        result: dict[str, list[str]] = {}
        for role in ("wave0", "wave1", "report"):
            values = raw[role]
            require(isinstance(values, list), f"{label} {role} IDs differ")
            require(
                all(
                    isinstance(item, str) and JOB_ID.fullmatch(item) is not None
                    for item in values
                ),
                f"{label} {role} IDs differ",
            )
            normalized = list(values)
            require(
                len(normalized) <= 1
                and normalized == sorted(set(normalized), key=int),
                f"{label} {role} IDs differ",
            )
            result[role] = normalized
        require(
            len({item for values in result.values() for item in values})
            == sum(len(values) for values in result.values()),
            f"{label} assigns one scheduler ID to multiple roles",
        )
        return result

    chain: list[dict[str, Any]] = []
    for generation in range(len(indices)):
        name = f"{filename_prefix}_{generation:04d}.json"
        path = directory / name
        _regular_nonsymlink(path, f"residual recovery generation {generation}")
        require(
            stat.S_IMODE(path.lstat().st_mode) == 0o444,
            f"residual recovery generation {generation} mode differs",
        )
        value = read_json(path)
        require(
            set(value) == expected_fields
            and type(value.get("schema_version")) is int
            and value.get("schema_version") == 1
            and value.get("record") == "residual_reconciliation"
            and type(value.get("generation")) is int
            and value.get("generation") == generation
            and all(
                exact_json_equal(value.get(key), expected)
                for key, expected in identity.items()
            )
            and value.get("previous_terminal_name") == previous_name
            and value.get("previous_terminal_sha256") == previous_sha256
            and type(value.get("new_jobs_created")) is int
            and value.get("new_jobs_created") == 0,
            f"residual recovery generation {generation} identity differs",
        )
        attempts = _validated_scheduler_attempt_ledger(
            value.get("scheduler_control_plane_observations"),
            expected_control_plane=expected_control_plane,
            fallback=scheduler_fallback,
        )
        live = role_map(
            value.get("live_verified_job_ids_by_role"),
            f"residual recovery generation {generation} live",
        )
        live_flat = sorted(
            {item for values in live.values() for item in values}, key=int
        )
        require(
            value.get("live_verified_job_ids") == live_flat
            and value.get("cancelled_live_job_ids") == live_flat,
            f"residual recovery generation {generation} flat IDs differ",
        )
        _pre, reconstructed_live, cursor = _validated_recovery_census_rounds(
            value.get("pre_cancel_census_rounds"),
            attempts=attempts,
            role_names=role_names,
            squeue=squeue,
            comment=comment,
            label=f"residual recovery generation {generation} pre-cancel",
            expected_start=0,
        )
        require(
            exact_json_equal(reconstructed_live, live),
            f"residual recovery generation {generation} census differs",
        )
        stored_history = value.get("cancel_attempt_history")
        require(
            type(value.get("cancel_history_length_before")) is int
            and value["cancel_history_length_before"] == previous_history_length
            and isinstance(stored_history, list)
            and previous_history_length <= len(stored_history) <= len(disk_history)
            and exact_json_equal(
                stored_history, disk_history[: len(stored_history)]
            ),
            f"residual recovery generation {generation} cancellation history differs",
        )
        post = role_map(
            value.get("post_cancel_active_job_ids_by_role"),
            f"residual recovery generation {generation} post-cancel",
        )
        if live_flat:
            require(
                len(stored_history) > previous_history_length,
                f"residual recovery generation {generation} lacks a new cancel attempt",
            )
            last_attempt = stored_history[-1]
            cancellation = last_attempt.get("cancellation")
            require(
                last_attempt.get("job_ids") == live_flat
                and isinstance(cancellation, Mapping)
                and value.get("cancel_calling_sha256")
                == last_attempt.get("calling_sha256")
                and exact_json_equal(value.get("cancellation"), cancellation)
                and value.get("status")
                == f"{status_prefix}_terminal_after_residual_cancel",
                f"residual recovery generation {generation} cancellation differs",
            )
            cancellation_attempts = _validated_scheduler_attempt_ledger(
                cancellation.get("scheduler_attempts"),
                expected_control_plane=expected_control_plane,
                fallback=scheduler_fallback,
            )
            require(
                exact_json_equal(
                    attempts[cursor : cursor + len(cancellation_attempts)],
                    cancellation_attempts,
                ),
                f"residual recovery generation {generation} cancel ledger differs",
            )
            cursor += len(cancellation_attempts)
            _post_rounds, reconstructed_post, cursor = (
                _validated_recovery_census_rounds(
                    value.get("post_cancel_census_rounds"),
                    attempts=attempts,
                    role_names=role_names,
                    squeue=squeue,
                    comment=comment,
                    label=f"residual recovery generation {generation} post-cancel",
                    expected_start=cursor,
                )
            )
            require(
                exact_json_equal(reconstructed_post, post)
                and exact_json_equal(
                    post, {"wave0": [], "wave1": [], "report": []}
                ),
                f"residual recovery generation {generation} is not terminal",
            )
        else:
            require(
                value.get("status")
                == f"{status_prefix}_terminal_no_active_jobs"
                and value.get("cancel_calling_sha256") is None
                and value.get("cancellation") is None
                and value.get("post_cancel_census_rounds") == []
                and exact_json_equal(
                    post, {"wave0": [], "wave1": [], "report": []}
                ),
                f"residual recovery generation {generation} no-active evidence differs",
            )
        require(
            cursor == len(attempts)
            and type(value.get("scheduler_calls")) is int
            and value["scheduler_calls"] == len(attempts),
            f"residual recovery generation {generation} scheduler ledger differs",
        )
        chain.append(dict(value))
        previous_name = name
        previous_sha256 = file_sha256(path)
        previous_history_length = len(stored_history)
    return chain


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
    contract, submission_sha256 = read_immutable_json(
        contract_path, "recovery submission contract"
    )
    seal_path = submission_root / "journal" / "0002_CONTRACT_SEALED.json"
    seal, _seal_sha256 = read_immutable_json(
        seal_path, "durable contract-seal journal"
    )
    require(
        exact_json_equal(
            seal,
            {
            "schema_version": 1,
            "record": "contract_sealed",
            "submission_sha256": submission_sha256,
            "launch_count": 20,
            },
        ),
        "recovery contract does not match its durable seal journal",
    )
    claim_path = submission_root / "journal" / "0000_CLAIMED.json"
    claim, _claim_sha256 = read_immutable_json(
        claim_path, "durable transaction claim"
    )
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
        and type(claim.get("schema_version")) is int
        and claim.get("schema_version") == 1
        and claim.get("record") == "claimed"
        and claim.get("campaign_id") == CAMPAIGN_ID
        and claim.get("submission_root") == str(submission_root)
        and SHA256.fullmatch(str(claim.get("claim_token", ""))) is not None
        and SHA256.fullmatch(str(claim.get("scientific_output_fingerprint", ""))) is not None,
        "durable transaction claim differs",
    )
    require(
        type(contract.get("schema_version")) is int
        and contract.get("schema_version") == 1
        and contract.get("status") == "sealed_for_submission",
        "recovery contract is not sealed",
    )
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
    _protocol_digest, protocol_payload = _hash_relative_regular(
        snapshot_root,
        PACKAGE_RELATIVE / "protocol.sha256",
        "recovery protocol lock",
        capture=True,
    )
    assert protocol_payload is not None
    try:
        recovered_protocol = protocol_payload.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise SubmissionError(f"recovery protocol lock is not ASCII: {exc}") from exc
    require(
        contract.get("package_protocol_sha256") == recovered_protocol,
        "recovery protocol-lock text differs",
    )
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
        exact_json_equal(
            contract.get("orchestration_interpreter"), recovered_interpreter
        ),
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
    control_plane = _scheduler_contract(
        manifest["execution"].get("scheduler_control_plane")
    )
    require(
        exact_json_equal(
            contract.get("scheduler_control_plane_contract"), control_plane
        ),
        "recovery scheduler control-plane contract differs",
    )
    verified_scheduler_preclaim = _validated_scheduler_preclaim(
        contract.get("scheduler_preclaim"), manifest
    )
    scheduler_fallback = _validated_scheduler_fallback(
        contract.get("scheduler_fallback_config"),
        control_plane,
        verified_scheduler_preclaim["scheduler_control_plane"],
    )[0]

    execution = manifest["execution"]
    scontrol = str(execution["scontrol"])
    _regular_nonsymlink(Path(scontrol), "recovery scontrol")
    require(os.access(scontrol, os.X_OK), "recovery scontrol is not executable")
    token = submission_sha256[:16]
    wave0_name = f"exp23-launch8-{token}-wave0"
    comment = f"treewm-exp23:{submission_sha256}"

    def finish_committed_receipt(
        receipt: dict[str, Any], recovery_status: str
    ) -> dict[str, Any]:
        durable_ready, _durable_ready_sha256 = read_immutable_json(
            submission_root / "journal" / "0007_READY_TO_COMMIT.json",
            "durable ready-to-commit journal",
        )
        require(
            exact_json_equal(_receipt_from_ready_record(durable_ready), receipt),
            "durable READY differs from committed submission receipt",
        )
        authorization, authorization_sha256 = _validated_dag_authorization(
            submission_root, submission_sha256, receipt
        )
        cancel_latch = _validated_committed_cancel_latch(submission_root, receipt)
        committed_receipt_path = submission_root / "SUBMISSION_RECEIPT.json"
        if _lexical_exists(committed_receipt_path):
            committed_receipt, _committed_receipt_sha256 = read_immutable_json(
                committed_receipt_path,
                "committed submission receipt",
            )
            require(
                exact_json_equal(committed_receipt, receipt),
                "committed submission receipt bytes differ from recovery authority",
            )
        else:
            require(
                cancel_latch is not None,
                "durable READY lacks its committed submission receipt",
            )
        release_path = submission_root / "journal" / "0008_WAVE0_RELEASED.json"
        release_calling_path = (
            submission_root / "journal" / "CALLING_WAVE0_RELEASE.json"
        )
        release_calling_value = {
            "schema_version": 1,
            "status": "scheduler_calling",
            "campaign_id": CAMPAIGN_ID,
            "submission_sha256": submission_sha256,
            "claim_token": claim["claim_token"],
            "role": "wave0_release",
            "job_name": wave0_name,
            "scheduler_comment": comment,
            "command": [scontrol, "release", receipt["wave0_array_job_id"]],
            "transaction_lock": _leased_transaction_lock_binding(
                scheduler_runner
            ),
        }
        scheduler_observations: list[dict[str, Any]] = []
        scheduler_calls = 0
        if cancel_latch is not None and not _lexical_exists(release_path):
            return {
                **receipt,
                "status": "committed_two_wave_dag_cancel_requested",
                "recovery": recovery_status,
                "wave0_release": "forbidden_by_durable_cancel",
                "scheduler_calls": 0,
                "authorization": authorization,
                "cancellation": cancel_latch,
            }
        if _lexical_exists(release_path):
            require(
                _lexical_exists(release_calling_path),
                "durable wave-zero release calling record is absent",
            )
            release_calling, release_calling_sha256 = read_immutable_json(
                release_calling_path, "durable wave-zero release calling record"
            )
            require(
                exact_json_equal(release_calling, release_calling_value),
                "durable wave-zero release calling record differs",
            )
            released, _released_sha256 = read_immutable_json(
                release_path, "durable wave-zero release journal"
            )
            require(
                set(released)
                == {
                    "schema_version",
                    "record",
                    "wave0_array_job_id",
                    "submission_authorization_sha256",
                    "calling_sha256",
                    "release_evidence",
                }
                and type(released.get("schema_version")) is int
                and released.get("schema_version") == 1
                and released.get("record") == "wave0_released"
                and released.get("wave0_array_job_id")
                == receipt["wave0_array_job_id"]
                and released.get("submission_authorization_sha256")
                == authorization_sha256
                and released.get("calling_sha256") == release_calling_sha256
                and isinstance(released.get("release_evidence"), Mapping),
                "durable wave-zero release journal differs",
            )
            _validated_wave0_release_evidence(
                released["release_evidence"],
                scontrol=scontrol,
                job_id=receipt["wave0_array_job_id"],
                job_name=wave0_name,
                comment=comment,
                expected_observation=verified_scheduler_preclaim[
                    "scheduler_control_plane"
                ],
            )
        else:
            require(
                cancel_latch is None,
                "durable cancellation forbids recovery wave-zero release",
            )
            if _lexical_exists(release_calling_path):
                release_calling, release_calling_sha256 = read_immutable_json(
                    release_calling_path,
                    "recovery wave-zero release calling record",
                )
                require(
                    exact_json_equal(release_calling, release_calling_value),
                    "recovery wave-zero release calling record differs",
                )
            else:
                release_calling_sha256 = exclusive_json(
                    release_calling_path, release_calling_value
                )
            release_evidence = _ensure_authorized_wave0_released(
                scontrol=scontrol,
                job_id=receipt["wave0_array_job_id"],
                job_name=wave0_name,
                comment=comment,
                cwd=snapshot_root,
                runner=scheduler_runner,
                control_plane=control_plane,
                expected_observation=verified_scheduler_preclaim[
                    "scheduler_control_plane"
                ],
                observations=scheduler_observations,
            )
            _validated_wave0_release_evidence(
                release_evidence,
                scontrol=scontrol,
                job_id=receipt["wave0_array_job_id"],
                job_name=wave0_name,
                comment=comment,
                expected_observation=verified_scheduler_preclaim[
                    "scheduler_control_plane"
                ],
            )
            append_journal(
                submission_root,
                8,
                "WAVE0_RELEASED",
                {
                    "wave0_array_job_id": receipt["wave0_array_job_id"],
                    "submission_authorization_sha256": authorization_sha256,
                    "calling_sha256": release_calling_sha256,
                    "release_evidence": release_evidence,
                },
            )
            scheduler_calls = len(scheduler_observations)
        return {
            **receipt,
            "status": (
                "committed_two_wave_dag_cancel_requested_after_release"
                if cancel_latch is not None
                else "committed_two_wave_dag"
            ),
            "recovery": recovery_status,
            "wave0_release": "verified",
            "scheduler_calls": scheduler_calls,
            "authorization": authorization,
            **({"cancellation": cancel_latch} if cancel_latch is not None else {}),
        }

    receipt_path = submission_root / "SUBMISSION_RECEIPT.json"
    if _lexical_exists(receipt_path):
        receipt, _receipt_sha256 = read_immutable_json(
            receipt_path, "committed submission receipt"
        )
        validated = _receipt_from_ready_record({"record": "ready_to_commit", **receipt})
        require(
            validated["submission_sha256"] == submission_sha256,
            "committed receipt submission differs",
        )
        return finish_committed_receipt(validated, "already_committed")

    ready_path = submission_root / "journal" / "0007_READY_TO_COMMIT.json"
    if _lexical_exists(ready_path):
        require(
            not any(
                _lexical_exists(submission_root / "journal" / name)
                for name in ("9999_ABORTED.json", "9998_OUTER_ABORTED.json", "9000_RECOVERY_CANCELLED.json")
            ),
            "durable READY conflicts with an aborted/cancelled transaction",
        )
        ready, _ready_sha256 = read_immutable_json(
            ready_path, "durable ready-to-commit journal"
        )
        receipt = _receipt_from_ready_record(ready)
        require(receipt["submission_sha256"] == submission_sha256, "ready receipt submission differs")
        if _validated_committed_cancel_latch(submission_root, receipt) is not None:
            return finish_committed_receipt(
                receipt, "durable_ready_cancel_precedence"
            )
        _validated_dag_authorization(submission_root, submission_sha256, receipt)
        exclusive_json(receipt_path, receipt)
        return finish_committed_receipt(
            receipt, "committed_from_durable_ready_record"
        )

    execution = manifest["execution"]
    sbatch = str(execution["sbatch"])
    scancel = str(execution["scancel"])
    squeue = str(execution.get("squeue") or (Path(sbatch).parent / "squeue"))
    for path, label in ((scancel, "scancel"), (squeue, "squeue")):
        _regular_nonsymlink(Path(path), f"recovery {label}")
        require(os.access(path, os.X_OK), f"recovery {label} is not executable")
    token = submission_sha256[:16]
    wave0_name = f"exp23-launch8-{token}-wave0"
    wave1_name = f"exp23-launch8-{token}-wave1"
    report_name = f"exp23-launch8-{token}-report"
    comment = f"treewm-exp23:{submission_sha256}"
    role_names = {"wave0": wave0_name, "wave1": wave1_name, "report": report_name}
    journal_paths = {
        "wave0": submission_root / "journal" / "0003_WAVE0_SUBMITTED.json",
        "wave1": submission_root / "journal" / "0004_WAVE1_SUBMITTED.json",
        "report": submission_root / "journal" / "0005_REPORT_SUBMITTED.json",
    }
    transaction_lock_binding = _leased_transaction_lock_binding(scheduler_runner)
    cancel_context = {
        "schema_version": 1,
        "status": "scheduler_calling",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "claim_token": claim["claim_token"],
        "role": "recovery_cancel",
        "transaction_lock": transaction_lock_binding,
    }
    existing_cancel_history = _validated_recovery_cancel_history(
        submission_root / "journal",
        calling_prefix="CALLING_RECOVERY_CANCEL",
        result_prefix="RECOVERY_CANCEL_RESULT",
        context=cancel_context,
        scancel=scancel,
        expected_control_plane=verified_scheduler_preclaim[
            "scheduler_control_plane"
        ],
        fallback=scheduler_fallback,
    )
    prior_recovery_path = submission_root / "journal" / "9000_RECOVERY_CANCELLED.json"
    prior_recovery: dict[str, Any] | None = None
    if _lexical_exists(prior_recovery_path):
        _regular_nonsymlink(prior_recovery_path, "terminal recovery journal")
        require(
            stat.S_IMODE(prior_recovery_path.lstat().st_mode) == 0o444,
            "terminal recovery journal mode differs",
        )
        prior_recovery = _validated_recovery_terminal_record(
            read_json(prior_recovery_path),
            submission_root=submission_root,
            submission_sha256=submission_sha256,
            claim_token=str(claim["claim_token"]),
            package_protocol_sha256=recovered_protocol,
            role_names=role_names,
            comment=comment,
            squeue=squeue,
            scancel=scancel,
            transaction_lock=transaction_lock_binding,
            expected_control_plane=verified_scheduler_preclaim[
                "scheduler_control_plane"
            ],
            scheduler_fallback=scheduler_fallback,
        )

    aborted_ids_by_role: dict[str, list[str]] = {
        "wave0": [], "wave1": [], "report": []
    }
    aborted: dict[str, Any] | None = None
    aborted_path = submission_root / "journal" / "9999_ABORTED.json"
    if _lexical_exists(aborted_path):
        _regular_nonsymlink(aborted_path, "initial abort journal")
        require(
            stat.S_IMODE(aborted_path.lstat().st_mode) == 0o444,
            "initial abort journal mode differs",
        )
        aborted = read_json(aborted_path)
        require(
            set(aborted)
            == {
                "schema_version",
                "record",
                "error",
                "known_job_ids",
                "job_ids_by_role",
                "cancellation_authority_job_ids",
                "cancellation_authority_job_ids_by_role",
                "submission_sha256",
                "reconciliation_errors",
                "cancellation",
                "cancellation_error",
                "cancel_attempt_history",
                "scheduler_control_plane_observations",
            }
            and type(aborted.get("schema_version")) is int
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
        abort_authority_by_role = _validated_abort_role_ids(
            aborted,
            "abort cancellation authority",
            role_key="cancellation_authority_job_ids_by_role",
            flat_key="cancellation_authority_job_ids",
        )
        require(
            all(
                set(abort_authority_by_role[role]).issubset(
                    aborted_ids_by_role[role]
                )
                for role in ("wave0", "wave1", "report")
            ),
            "abort cancellation authority escaped the claimed role identity",
        )
        aborted_history = aborted.get("cancel_attempt_history")
        require(
            isinstance(aborted_history, list)
            and len(aborted_history) <= len(existing_cancel_history)
            and exact_json_equal(
                aborted_history,
                existing_cancel_history[: len(aborted_history)],
            )
            and exact_json_equal(
                aborted.get("cancellation"),
                aborted_history[-1]["cancellation"] if aborted_history else None,
            ),
            "abort cancellation attempt history differs",
        )
    outer_path = submission_root / "journal" / "9998_OUTER_ABORTED.json"
    if _lexical_exists(outer_path):
        _regular_nonsymlink(outer_path, "outer abort journal")
        require(
            stat.S_IMODE(outer_path.lstat().st_mode) == 0o444,
            "outer abort journal mode differs",
        )
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
            and type(outer.get("schema_version")) is int
            and outer.get("schema_version") == 1
            and outer.get("record") == "outer_aborted"
            and outer.get("receipt_committed") is False
            and outer.get("submission_sha256") == submission_sha256
            and outer.get("claim_token") == claim["claim_token"]
            and isinstance(outer.get("error"), str),
            "outer-abort journal is not authenticated to the recovery contract",
        )
        outer_roles = _validated_abort_role_ids(outer, "outer abort")
        for role in ("wave0", "wave1", "report"):
            aborted_ids_by_role[role] = sorted(
                set(aborted_ids_by_role[role] + outer_roles[role]), key=int
            )

    dag_evidence = load_dag_evidence(snapshot_root)
    records: dict[str, Any] = {}
    calling_records: dict[str, Any] = {}
    journal_ids: dict[str, str] = {}
    found_gap = False
    for role in ("wave0", "wave1", "report"):
        journal_path = journal_paths[role]
        if not _lexical_exists(journal_path):
            found_gap = True
            continue
        require(not found_gap, "recovery submission journals are not a contiguous prefix")
        _regular_nonsymlink(journal_path, f"recovery {role} submitted journal")
        require(
            stat.S_IMODE(journal_path.lstat().st_mode) == 0o444,
            f"recovery {role} submitted journal mode differs",
        )
        journal = read_json(journal_path)
        require(
            type(journal.get("schema_version")) is int
            and journal.get("schema_version") == 1
            and journal.get("record") == f"{role}_submitted",
            f"recovery {role} journal record differs",
        )
        journal_id = journal.get("job_id")
        require(
            isinstance(journal_id, str)
            and JOB_ID.fullmatch(journal_id) is not None,
            f"recovery {role} journal ID differs",
        )
        journal_ids[role] = journal_id
        records[role] = {
            key: item
            for key, item in journal.items()
            if key not in {"schema_version", "record", "job_id"}
        }
        calling_path = (
            submission_root / "journal" / f"CALLING_{role.upper()}.json"
        )
        require(
            _lexical_exists(calling_path),
            f"recovery {role} journal lacks its pre-call intent",
        )
        calling_records[role] = read_json(calling_path)

    calling_hashes: dict[str, str] = {}
    calling_seen_gap = False
    for role in ("wave0", "wave1", "report"):
        path = submission_root / "journal" / f"CALLING_{role.upper()}.json"
        if not _lexical_exists(path):
            calling_seen_gap = True
            continue
        require(
            not calling_seen_gap,
            "recovery scheduler calling intents are not a contiguous prefix",
        )
        _regular_nonsymlink(path, f"recovery {role} scheduler calling intent")
        require(
            stat.S_IMODE(path.lstat().st_mode) == 0o444,
            f"recovery {role} scheduler calling intent mode differs",
        )
        value = read_json(path)
        try:
            digest = dag_evidence.validate_calling_intent(
                value,
                role=role,
                manifest=manifest,
                snapshot_root=snapshot_root,
                submission_root=submission_root,
                submission_sha256=submission_sha256,
                claim_token=str(claim["claim_token"]),
                predecessor_job_ids=journal_ids,
                expected_lock=transaction_lock_binding,
            )
        except BaseException as exc:
            raise SubmissionError(
                f"recovery {role} scheduler calling intent differs: {exc}"
            ) from exc
        require(digest == file_sha256(path), f"recovery {role} calling hash differs")
        calling_hashes[role] = digest
        calling_records.setdefault(role, value)
    require(
        set(records).issubset(set(calling_hashes))
        and len(calling_hashes) <= len(records) + 1,
        "recovery calling/submitted prefix differs",
    )
    if records:
        try:
            dag_evidence.validate_dag_records(
                records,
                {role: calling_records[role] for role in records},
                {role: calling_hashes[role] for role in records},
                manifest=manifest,
                snapshot_root=snapshot_root,
                submission_root=submission_root,
                submission_sha256=submission_sha256,
                claim_token=str(claim["claim_token"]),
                job_ids=journal_ids,
                expected_control_plane=verified_scheduler_preclaim[
                    "scheduler_control_plane"
                ],
                expected_lock=transaction_lock_binding,
                allow_prefix=True,
            )
        except BaseException as exc:
            raise SubmissionError(
                f"recovery durable submitted scheduler evidence differs: {exc}"
            ) from exc

    durable_ids_by_role = {
        role: sorted(
            set(aborted_ids_by_role[role])
            | ({journal_ids[role]} if role in journal_ids else set()),
            key=int,
        )
        for role in ("wave0", "wave1", "report")
    }
    require(
        len(
            {
                item
                for values in durable_ids_by_role.values()
                for item in values
            }
        )
        == sum(len(values) for values in durable_ids_by_role.values()),
        "recovery durable evidence assigns one scheduler ID to multiple roles",
    )
    durable_ids = sorted(
        {item for values in durable_ids_by_role.values() for item in values}, key=int
    )
    if aborted is not None:
        _validated_successful_cancellation(
            aborted, scancel, list(aborted["cancellation_authority_job_ids"])
        )

    if prior_recovery is not None:
        require(
            prior_recovery.get("durable_claimed_job_ids_by_role")
            == durable_ids_by_role
            and prior_recovery.get("durable_claimed_job_ids") == durable_ids
            and prior_recovery.get("calling_intent_sha256_by_role")
            == calling_hashes,
            "terminal recovery record differs from the reconstructed durable prefix",
        )
        residual_identity = {
            "campaign_id": CAMPAIGN_ID,
            "submission_root": str(submission_root),
            "submission_sha256": submission_sha256,
            "claim_token": claim["claim_token"],
            "package_protocol_sha256": recovered_protocol,
            "transaction_lock": transaction_lock_binding,
        }
        residual_chain = _validated_residual_recovery_chain(
            submission_root / "journal",
            filename_prefix="RECOVERY_RECONCILED",
            initial_terminal_path=prior_recovery_path,
            initial_cancel_history_length=len(
                prior_recovery["cancel_attempt_history"]
            ),
            identity=residual_identity,
            status_prefix="recovered_residual",
            cancel_directory=submission_root / "journal",
            cancel_calling_prefix="CALLING_RECOVERY_CANCEL",
            cancel_result_prefix="RECOVERY_CANCEL_RESULT",
            cancel_context=cancel_context,
            role_names=role_names,
            comment=comment,
            squeue=squeue,
            scancel=scancel,
            expected_control_plane=verified_scheduler_preclaim[
                "scheduler_control_plane"
            ],
            scheduler_fallback=scheduler_fallback,
        )
        bound_history_length = len(
            (
                residual_chain[-1]["cancel_attempt_history"]
                if residual_chain
                else prior_recovery["cancel_attempt_history"]
            )
        )
        revalidation_observations: list[dict[str, Any]] = []
        revalidation_rounds, revalidation_active = _recovery_census_rounds(
            squeue=squeue,
            role_names=role_names,
            comment=comment,
            cwd=snapshot_root,
            runner=scheduler_runner,
            control_plane=control_plane,
            fallback=scheduler_fallback,
            expected_observation=verified_scheduler_preclaim[
                "scheduler_control_plane"
            ],
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
        for role in ("wave0", "wave1", "report"):
            if role in journal_ids and revalidation_active[role]:
                require(
                    revalidation_active[role] == [journal_ids[role]],
                    f"residual recovery live {role} identity differs from semantic journal",
                )
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
                    _append_recovery_cancel_attempt(
                        submission_root / "journal",
                        calling_prefix="CALLING_RECOVERY_CANCEL",
                        result_prefix="RECOVERY_CANCEL_RESULT",
                        context=cancel_context,
                        scancel=scancel,
                        job_ids=revalidation_ids,
                        cwd=snapshot_root,
                        runner=scheduler_runner,
                        control_plane=control_plane,
                        fallback=scheduler_fallback,
                        expected_observation=verified_scheduler_preclaim[
                            "scheduler_control_plane"
                        ],
                        observations=revalidation_observations,
                    )
                )
                residual_calling_sha256 = residual_history[-1][
                    "calling_sha256"
                ]
                post_rounds, post_active = _recovery_census_rounds(
                    squeue=squeue,
                    role_names=role_names,
                    comment=comment,
                    cwd=snapshot_root,
                    runner=scheduler_runner,
                    control_plane=control_plane,
                    fallback=scheduler_fallback,
                    expected_observation=verified_scheduler_preclaim[
                        "scheduler_control_plane"
                    ],
                    observations=revalidation_observations,
                )
                require(
                    post_active == {"wave0": [], "wave1": [], "report": []},
                    "residual recovery exact jobs remain active after cancellation",
                )
            generation = len(residual_chain)
            previous_path = (
                submission_root
                / "journal"
                / f"RECOVERY_RECONCILED_{generation - 1:04d}.json"
                if generation
                else prior_recovery_path
            )
            exclusive_json(
                submission_root
                / "journal"
                / f"RECOVERY_RECONCILED_{generation:04d}.json",
                {
                    "schema_version": 1,
                    "record": "residual_reconciliation",
                    **residual_identity,
                    "generation": generation,
                    "previous_terminal_name": previous_path.name,
                    "previous_terminal_sha256": file_sha256(previous_path),
                    "status": (
                        "recovered_residual_terminal_after_residual_cancel"
                        if revalidation_ids
                        else "recovered_residual_terminal_no_active_jobs"
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
            residual_chain = _validated_residual_recovery_chain(
                submission_root / "journal",
                filename_prefix="RECOVERY_RECONCILED",
                initial_terminal_path=prior_recovery_path,
                initial_cancel_history_length=len(
                    prior_recovery["cancel_attempt_history"]
                ),
                identity=residual_identity,
                status_prefix="recovered_residual",
                cancel_directory=submission_root / "journal",
                cancel_calling_prefix="CALLING_RECOVERY_CANCEL",
                cancel_result_prefix="RECOVERY_CANCEL_RESULT",
                cancel_context=cancel_context,
                role_names=role_names,
                comment=comment,
                squeue=squeue,
                scancel=scancel,
                expected_control_plane=verified_scheduler_preclaim[
                    "scheduler_control_plane"
                ],
                scheduler_fallback=scheduler_fallback,
            )
        return {
            **prior_recovery,
            "reused_recovery": True,
            "residual_reconciliation_chain": residual_chain,
            "revalidation_census_rounds": revalidation_rounds,
            "revalidation_scheduler_control_plane_observations": revalidation_observations,
            "recovery_invocation_scheduler_calls": len(revalidation_observations),
        }

    latch_path = submission_root / "CANCEL_REQUESTED.json"
    preexisting_latch: dict[str, Any] | None = None
    cancel_latch_sha256: str | None = None
    if _lexical_exists(latch_path):
        raw_latch, cancel_latch_sha256 = read_immutable_json(
            latch_path, "recovery cancellation latch"
        )
        preexisting_latch = _validated_recovery_cancel_latch(
            raw_latch,
            submission_sha256=submission_sha256,
            claim_token=str(claim["claim_token"]),
            transaction_lock=transaction_lock_binding,
        )

    scheduler_observations: list[dict[str, Any]] = []
    pre_cancel_rounds, live_ids_by_role = _recovery_census_rounds(
        squeue=squeue,
        role_names=role_names,
        comment=comment,
        cwd=snapshot_root,
        runner=scheduler_runner,
        control_plane=control_plane,
        fallback=scheduler_fallback,
        expected_observation=verified_scheduler_preclaim[
            "scheduler_control_plane"
        ],
        observations=scheduler_observations,
    )
    live_ids = sorted(
        {item for values in live_ids_by_role.values() for item in values}, key=int
    )
    for role in ("wave0", "wave1", "report"):
        if role in journal_ids and live_ids_by_role[role]:
            require(
                live_ids_by_role[role] == [journal_ids[role]],
                f"recovery live {role} identity differs from semantic journal",
            )
    latch = {
        "schema_version": 1,
        "status": "cancel_requested",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "claim_token": claim["claim_token"],
        "wave0_array_job_id": live_ids_by_role["wave0"][0] if live_ids_by_role["wave0"] else None,
        "wave1_array_job_id": live_ids_by_role["wave1"][0] if live_ids_by_role["wave1"] else None,
        "report_job_id": live_ids_by_role["report"][0] if live_ids_by_role["report"] else None,
        "job_ids_by_role": live_ids_by_role,
        "transaction_lock": transaction_lock_binding,
        "recovery": True,
        "scheduler_id_authority": "fresh_settled_exact_name_census_only",
    }
    if preexisting_latch is not None:
        assert cancel_latch_sha256 is not None
    else:
        cancel_latch_sha256 = exclusive_json(latch_path, latch)
    cancel_history = existing_cancel_history
    cancellation = None
    cancel_calling_sha256 = None
    post_cancel_rounds: list[dict[str, Any]] = []
    post_cancel_active = {"wave0": [], "wave1": [], "report": []}
    if live_ids:
        cancel_history, cancellation = _append_recovery_cancel_attempt(
            submission_root / "journal",
            calling_prefix="CALLING_RECOVERY_CANCEL",
            result_prefix="RECOVERY_CANCEL_RESULT",
            context=cancel_context,
            scancel=scancel,
            job_ids=live_ids,
            cwd=snapshot_root,
            runner=scheduler_runner,
            control_plane=control_plane,
            fallback=scheduler_fallback,
            expected_observation=verified_scheduler_preclaim[
                "scheduler_control_plane"
            ],
            observations=scheduler_observations,
        )
        post_cancel_rounds, post_cancel_active = _recovery_census_rounds(
            squeue=squeue,
            role_names=role_names,
            comment=comment,
            cwd=snapshot_root,
            runner=scheduler_runner,
            control_plane=control_plane,
            fallback=scheduler_fallback,
            expected_observation=verified_scheduler_preclaim[
                "scheduler_control_plane"
            ],
            observations=scheduler_observations,
        )
        require(
            post_cancel_active == {"wave0": [], "wave1": [], "report": []},
            "recovery exact jobs remain active after cancellation",
        )
    if cancel_history:
        cancel_calling_sha256 = cancel_history[-1]["calling_sha256"]
        cancellation = cancel_history[-1]["cancellation"]
    recovery_record = {
        "submission_sha256": submission_sha256,
        "submission_root": str(submission_root),
        "claim_token": claim["claim_token"],
        "package_protocol_sha256": recovered_protocol,
        "status": (
            "recovered_terminal_after_cancel_attempts"
            if cancel_history
            else "recovered_terminal_no_active_jobs"
        ),
        "durable_claimed_job_ids": durable_ids,
        "durable_claimed_job_ids_by_role": durable_ids_by_role,
        "live_verified_job_ids": live_ids,
        "live_verified_job_ids_by_role": live_ids_by_role,
        "calling_intent_sha256_by_role": calling_hashes,
        "pre_cancel_census_rounds": pre_cancel_rounds,
        "cancelled_live_job_ids": live_ids,
        "cancel_latch_sha256": cancel_latch_sha256,
        "cancel_calling_sha256": cancel_calling_sha256,
        "cancellation": cancellation,
        "cancel_attempt_history": cancel_history,
        "post_cancel_census_rounds": post_cancel_rounds,
        "post_cancel_active_job_ids_by_role": post_cancel_active,
        "scheduler_control_plane_observations": scheduler_observations,
        "scheduler_calls": len(scheduler_observations),
        "transaction_lock": transaction_lock_binding,
        "new_jobs_created": 0,
    }
    recovery_path = submission_root / "journal" / "9000_RECOVERY_CANCELLED.json"
    if _lexical_exists(recovery_path):
        require(
            exact_json_equal(
                read_json(recovery_path),
                {
                    "schema_version": 1,
                    "record": "recovery_cancelled",
                    **recovery_record,
                },
            ),
            "recovery journal differs",
        )
    else:
        append_journal(submission_root, 9000, "RECOVERY_CANCELLED", recovery_record)
    validated_recovery = _validated_recovery_terminal_record(
        read_json(recovery_path),
        submission_root=submission_root,
        submission_sha256=submission_sha256,
        claim_token=str(claim["claim_token"]),
        package_protocol_sha256=recovered_protocol,
        role_names=role_names,
        comment=comment,
        squeue=squeue,
        scancel=scancel,
        transaction_lock=transaction_lock_binding,
        expected_control_plane=verified_scheduler_preclaim[
            "scheduler_control_plane"
        ],
        scheduler_fallback=scheduler_fallback,
    )
    return {**validated_recovery, "reused_recovery": False}


def recover_transaction(
    repo_root: Path,
    submission_root: Path,
    *,
    scheduler_runner: SchedulerRunner = _default_scheduler_runner,
) -> dict[str, Any]:
    """Recover only after proving that no live submitter owns the transaction."""

    with _TransactionLock(submission_root) as transaction_lock:
        leased_scheduler_runner = _scheduler_runner_with_lock_lease(
            scheduler_runner, transaction_lock
        )
        return _recover_transaction_locked(
            repo_root,
            submission_root,
            scheduler_runner=leased_scheduler_runner,
        )


def _summary(preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": preflight["status"],
        "campaign_id": preflight["campaign_id"],
        "package_protocol_sha256": preflight["package_protocol_sha256"],
        "source_sha256": preflight["source_contract"]["source_sha256"],
        "runtime_sha256": preflight["source_contract"]["runtime_sha256"],
        "snapshot_files": len(preflight["snapshot_inventory"]),
        "snapshot_inventory_sha256": preflight["snapshot_inventory_sha256"],
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
    actions.add_argument(
        "--snapshot-test",
        action="store_true",
        help="seal and fully verify a temporary copied tree without contacting Slurm",
    )
    actions.add_argument(
        "--scheduler-test",
        action="store_true",
        help="run only authenticated read-only Slurm controller and sbatch --test-only probes",
    )
    actions.add_argument(
        "--submit",
        action="store_true",
        help="explicitly seal and submit the held-wave0, wave1, and report DAG",
    )
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
    parser.add_argument("--audit-input-root", type=Path, required=True)
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
    input_root = _manifest_repository_root(bootstrap_manifest, args.audit_input_root)
    activate_isolated_runtime(bootstrap_manifest)
    value = _snapshot_preflight_in_process(
        root,
        input_root,
        args.protocol_sha256,
        inventory,
        runner=_default_runner,
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
        if args.snapshot_test:
            require(
                args.submission_root is None,
                "--snapshot-test does not accept a submission root",
            )
            verified = snapshot_test(repo_root)
            print(json.dumps(verified, sort_keys=True, indent=2, allow_nan=False))
            return 0
        activate_isolated_runtime(manifest)
        submission_root = (
            args.submission_root.absolute()
            if args.submission_root is not None
            else Path(str(manifest["paths"]["run_root"])) / "state" / "submission"
        )
        if args.scheduler_test:
            require(
                args.submission_root is None,
                "--scheduler-test does not accept a submission root",
            )
            preflight = static_preflight(repo_root, submission_root)
            verified = scheduler_preclaim_test(repo_root, preflight["manifest"])
            print(json.dumps(verified, sort_keys=True, indent=2, allow_nan=False))
            return 0
        if args.submit and _lexical_exists(submission_root):
            recovered = recover_transaction(repo_root, submission_root)
            print(json.dumps(recovered, sort_keys=True, indent=2, allow_nan=False))
            return 0 if recovered.get("status") == "committed_two_wave_dag" else 2
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
