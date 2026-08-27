"""Focused fail-closed tests for Exp23 submission, cancellation, and reporting."""

from __future__ import annotations

import importlib.util
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]


def load(name: str):
    path = PACKAGE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"exp23_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def submit():
    return load("submit")


@pytest.fixture(scope="module")
def report():
    return load("report")


@pytest.fixture(scope="module")
def cancel():
    return load("cancel")


@pytest.fixture(scope="module")
def weight_audit():
    return load("weight_audit")


@pytest.fixture(scope="module")
def prefix_target_audit():
    return load("prefix_target_audit")


@pytest.fixture(scope="module")
def causal_parity_audit():
    return load("causal_parity_audit")


@pytest.fixture(scope="module")
def worker():
    return load("worker")


def bind_causal_execution_context(
    causal_parity_audit,
    monkeypatch,
    execution_fd: int,
    expected_root_fd: int,
    sealed: bool,
    source: Path,
):
    monkeypatch.setattr(
        causal_parity_audit, "BOOTSTRAP_EXECUTION_ROOT_FD", execution_fd
    )
    monkeypatch.setattr(
        causal_parity_audit,
        "BOOTSTRAP_EXECUTION_ROOT_IDENTITY",
        causal_parity_audit._tree_stat_identity(os.fstat(expected_root_fd)),
    )
    monkeypatch.setattr(
        causal_parity_audit, "BOOTSTRAP_EXECUTION_ROOT_SEALED", sealed
    )
    monkeypatch.setattr(
        causal_parity_audit,
        "BOOTSTRAP_EXECUTION_SCRIPT_RELATIVE",
        causal_parity_audit.AUDIT_SOURCE_RELATIVE.as_posix(),
    )
    monkeypatch.setattr(
        causal_parity_audit,
        "BOOTSTRAP_EXECUTION_SCRIPT_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    relative = causal_parity_audit.AUDIT_SOURCE_RELATIVE
    source_root = source.parents[len(relative.parts) - 1]
    directory_identities = tuple(
        (
            Path(*relative.parts[:index]).as_posix(),
            causal_parity_audit._tree_stat_identity(
                (source_root / Path(*relative.parts[:index])).lstat()
            ),
        )
        for index in range(1, len(relative.parts))
    )
    monkeypatch.setattr(
        causal_parity_audit,
        "BOOTSTRAP_EXECUTION_DIRECTORY_IDENTITIES",
        directory_identities,
    )
    monkeypatch.setattr(
        causal_parity_audit,
        "BOOTSTRAP_EXECUTION_SCRIPT_IDENTITY",
        causal_parity_audit._tree_stat_identity(source.lstat()),
    )


def interpreter_identity(submit):
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    python = Path(manifest["paths"]["python"])
    target = python.resolve(strict=True)
    venv = python.parent.parent
    values = {
        key.strip(): value.strip()
        for line in (venv / "pyvenv.cfg").read_text().splitlines()
        if "=" in line
        for key, value in (line.split("=", 1),)
    }
    base = Path(values["home"].strip()).parent
    major, minor, *_rest = values["version"].split(".")
    version = f"python{major}.{minor}"
    return {
        "lexical_executable": str(python),
        "venv_site_packages": str(venv / "lib" / version / "site-packages"),
        "base_site_packages": str(base / "lib" / version / "site-packages"),
        "resolved_executable_sha256": submit.file_sha256(target),
        "resolved_executable_size": target.stat().st_size,
    }


def test_launch3_transaction_lock_path_is_exact(submit):
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    run_root = Path(manifest["paths"]["run_root"])
    submission_root = run_root / "state" / "submission"
    expected = run_root.parents[1] / manifest["paths"]["transaction_lock"]
    assert submit._transaction_lock_path(submission_root) == expected
    assert expected.name == ".exp23-6e55bb3083712144.transaction.lock"


def test_default_cli_is_read_only_and_rejects_wrong_interpreter(tmp_path):
    prospective = tmp_path / "must-not-exist"
    completed = subprocess.run(
        [sys.executable, str(PACKAGE / "submit.py"), "--test-only", "--submission-root", str(prospective)],
        cwd=REPO,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 2
    assert not prospective.exists()


def test_rejects_unknown_treewm_and_distributed_environment(submit):
    with pytest.raises(submit.SubmissionError):
        submit.reject_inherited_environment({"TREEWM_SURPRISE": "1"})
    with pytest.raises(submit.SubmissionError):
        submit.reject_inherited_environment({"WORLD_SIZE": "2"})


def test_isolated_bootstrap_blocks_sitecustomize_and_nested_python(submit, tmp_path):
    identity = interpreter_identity(submit)
    root = tmp_path / "root"
    hostile = tmp_path / "hostile"
    root.mkdir()
    hostile.mkdir()
    marker = tmp_path / "sitecustomize-ran"
    (hostile / "sitecustomize.py").write_text(
        f"from pathlib import Path; Path({str(marker)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    child = root / "child.py"
    child.write_text("import sys; print(sys.flags.isolated, sys.flags.no_site)\n", encoding="utf-8")
    outer = root / "outer.py"
    outer.write_text(
        "import subprocess, sys\n"
        f"r=subprocess.run([{identity['lexical_executable']!r},{str(child)!r}],capture_output=True,text=True)\n"
        "print(r.returncode, r.stdout.strip())\n",
        encoding="utf-8",
    )
    command = submit.isolated_python_command(
        [identity["lexical_executable"], str(outer)],
        root,
        identity,
        intercept_python_children=True,
    )
    completed = subprocess.run(
        command,
        env={"PYTHONPATH": str(hostile), "PATH": "/usr/bin:/bin"},
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "0 1 1"
    assert not marker.exists()


def test_isolated_child_reverifies_exact_read_only_snapshot(submit, tmp_path):
    identity = interpreter_identity(submit)
    root = tmp_path / "snapshot"
    root.mkdir()
    script = root / "probe.py"
    script.write_text("print('verified')\n", encoding="utf-8")
    inventory = {"probe.py": submit.file_sha256(script)}
    script.chmod(0o444)
    root.chmod(0o555)
    command = submit.isolated_python_command(
        [identity["lexical_executable"], str(script)],
        root,
        identity,
        intercept_python_children=False,
        snapshot_inventory=inventory,
    )
    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert completed.returncode == 0 and completed.stdout.strip() == "verified"
    root.chmod(0o755)
    rejected = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert rejected.returncode != 0


def test_isolated_bootstrap_causal_self_hash_uses_bound_snapshot_fd(
    submit, causal_parity_audit, tmp_path
):
    identity = interpreter_identity(submit)
    root = tmp_path / "snapshot"
    source = root / causal_parity_audit.AUDIT_SOURCE_RELATIVE
    source.parent.mkdir(parents=True)
    source_text = (PACKAGE / "causal_parity_audit.py").read_text(encoding="utf-8")
    source.write_text(
        source_text.replace(
            'if __name__ == "__main__":\n    raise SystemExit(main())\n',
            'if __name__ == "__main__":\n    print(file_sha256(Path(__file__)))\n',
        ),
        encoding="utf-8",
    )
    inventory = {str(source.relative_to(root)): submit.file_sha256(source)}
    directories = [root, source.parent, *source.parent.parents]
    directories = [path for path in directories if path == root or path.is_relative_to(root)]
    try:
        source.chmod(0o444)
        for directory in sorted(set(directories), key=lambda path: len(path.parts), reverse=True):
            directory.chmod(0o555)
        command = submit.isolated_python_command(
            [identity["lexical_executable"], str(source)],
            root,
            identity,
            intercept_python_children=False,
            snapshot_inventory=inventory,
        )
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == inventory[
            str(causal_parity_audit.AUDIT_SOURCE_RELATIVE)
        ]
    finally:
        for directory in directories:
            directory.chmod(0o755)


def test_isolated_bootstrap_causal_self_hash_accepts_exact_live_root(
    submit, causal_parity_audit, tmp_path
):
    identity = interpreter_identity(submit)
    root = tmp_path / "live-repository"
    source = root / causal_parity_audit.AUDIT_SOURCE_RELATIVE
    source.parent.mkdir(parents=True)
    source_text = (PACKAGE / "causal_parity_audit.py").read_text(encoding="utf-8")
    source.write_text(
        source_text.replace(
            'if __name__ == "__main__":\n    raise SystemExit(main())\n',
            'if __name__ == "__main__":\n    print(file_sha256(Path(__file__)))\n',
        ),
        encoding="utf-8",
    )
    expected = submit.file_sha256(source)
    command = submit.isolated_python_command(
        [identity["lexical_executable"], str(source)],
        root,
        identity,
        intercept_python_children=False,
        snapshot_inventory=None,
    )
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected


@pytest.mark.parametrize(
    ("surface", "unsafe_mode"),
    (
        ("root", 0o775),
        ("root", 0o757),
        ("directory", 0o775),
        ("directory", 0o757),
        ("file", 0o664),
        ("file", 0o646),
    ),
)
def test_isolated_bootstrap_rejects_unsafe_live_target_modes(
    submit, tmp_path, surface, unsafe_mode
):
    identity = interpreter_identity(submit)
    root = tmp_path / "live-root"
    target = root / "nested" / "main.py"
    target.parent.mkdir(parents=True)
    target.write_text("print('UNSAFE_TARGET_EXECUTED')\n", encoding="utf-8")
    changed = {"root": root, "directory": target.parent, "file": target}[surface]
    safe_mode = stat.S_IMODE(changed.stat().st_mode)
    changed.chmod(unsafe_mode)
    try:
        command = submit.isolated_python_command(
            [identity["lexical_executable"], str(target)],
            root,
            identity,
            intercept_python_children=False,
            snapshot_inventory=None,
        )
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert completed.returncode != 0
        assert "UNSAFE_TARGET_EXECUTED" not in completed.stdout
    finally:
        changed.chmod(safe_mode)


@pytest.mark.parametrize(
    ("sealed", "surface", "changed_mode"),
    (
        (False, "directory", 0o700),
        (False, "file", 0o600),
        (True, "file", 0o644),
    ),
)
def test_isolated_bootstrap_revalidates_target_after_system_exit(
    submit, tmp_path, sealed, surface, changed_mode
):
    identity = interpreter_identity(submit)
    root = tmp_path / "execution-root"
    target = root / "nested" / "main.py"
    target.parent.mkdir(parents=True)
    expression = "__file__" if surface == "file" else "os.path.dirname(__file__)"
    target.write_text(
        "import os\n"
        f"os.chmod({expression}, {changed_mode})\n"
        "print('POST_EXEC_MUTATION_RAN', flush=True)\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    inventory = None
    directories = [
        path
        for path in (root, target.parent, *target.parent.parents)
        if path == root or path.is_relative_to(root)
    ]
    if sealed:
        inventory = {str(target.relative_to(root)): submit.file_sha256(target)}
        target.chmod(0o444)
        for directory in sorted(
            set(directories), key=lambda path: len(path.parts), reverse=True
        ):
            directory.chmod(0o555)
    try:
        command = submit.isolated_python_command(
            [identity["lexical_executable"], str(target)],
            root,
            identity,
            intercept_python_children=False,
            snapshot_inventory=inventory,
        )
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert completed.stdout.strip() == "POST_EXEC_MUTATION_RAN"
        assert completed.returncode != 0, completed.stderr
        assert "target" in completed.stderr
    finally:
        target.chmod(0o644)
        for directory in directories:
            directory.chmod(0o755)


def test_isolated_bootstrap_rejects_live_root_swap_before_import(
    submit, tmp_path, monkeypatch
):
    identity = interpreter_identity(submit)
    root = tmp_path / "live-root"
    replacement = tmp_path / "replacement"
    root.mkdir()
    replacement.mkdir()
    source = "print('SWAPPED_ROOT_EXECUTED')\n"
    (root / "main.py").write_text(source, encoding="utf-8")
    (replacement / "main.py").write_text(source, encoding="utf-8")
    marker = tmp_path / "script-read"
    gate = tmp_path / "continue"
    needle = "try: import_root_fd = open_directory(root_real, 'pre-import snapshot root')"
    assert submit.ISOLATED_RUN_CODE.count(needle) == 1
    raced_bootstrap = submit.ISOLATED_RUN_CODE.replace(
        needle,
        "import time\n"
        "with open(sys.argv[9], 'w', encoding='utf-8') as stream: stream.write('ready')\n"
        "while not os.path.exists(sys.argv[10]): time.sleep(0.01)\n"
        + needle,
    )
    monkeypatch.setattr(submit, "ISOLATED_RUN_CODE", raced_bootstrap)
    command = submit.isolated_python_command(
        [
            identity["lexical_executable"],
            str(root / "main.py"),
            str(marker),
            str(gate),
        ],
        root,
        identity,
        intercept_python_children=False,
        snapshot_inventory=None,
    )
    process = subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    authenticated = tmp_path / "authenticated-root"
    try:
        deadline = time.monotonic() + 5.0
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), process.communicate(timeout=1)[1]
        root.rename(authenticated)
        replacement.rename(root)
        gate.write_text("continue", encoding="ascii")
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode != 0, (stdout, stderr)
        assert "execution root changed before import" in stderr
        assert "SWAPPED_ROOT_EXECUTED" not in stdout
    finally:
        if process.poll() is None:
            gate.write_text("continue", encoding="ascii")
            process.kill()
            process.communicate()


def test_isolated_bootstrap_rejects_live_nested_target_swap_before_import(
    submit, tmp_path, monkeypatch
):
    identity = interpreter_identity(submit)
    root = tmp_path / "live-root"
    target_directory = root / "outer" / "nested"
    replacement = tmp_path / "replacement-nested"
    target_directory.mkdir(parents=True)
    replacement.mkdir()
    source = "print('SWAPPED_NESTED_TARGET_EXECUTED')\n"
    (target_directory / "main.py").write_text(source, encoding="utf-8")
    (replacement / "main.py").write_text(source, encoding="utf-8")
    marker = tmp_path / "script-read"
    gate = tmp_path / "continue"
    needle = "try: import_root_fd = open_directory(root_real, 'pre-import snapshot root')"
    assert submit.ISOLATED_RUN_CODE.count(needle) == 1
    raced_bootstrap = submit.ISOLATED_RUN_CODE.replace(
        needle,
        "import time\n"
        "with open(sys.argv[9], 'w', encoding='utf-8') as stream: stream.write('ready')\n"
        "while not os.path.exists(sys.argv[10]): time.sleep(0.01)\n"
        + needle,
    )
    monkeypatch.setattr(submit, "ISOLATED_RUN_CODE", raced_bootstrap)
    command = submit.isolated_python_command(
        [
            identity["lexical_executable"],
            str(target_directory / "main.py"),
            str(marker),
            str(gate),
        ],
        root,
        identity,
        intercept_python_children=False,
        snapshot_inventory=None,
    )
    process = subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    authenticated = tmp_path / "authenticated-nested"
    try:
        deadline = time.monotonic() + 5.0
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), process.communicate(timeout=1)[1]
        target_directory.rename(authenticated)
        replacement.rename(target_directory)
        gate.write_text("continue", encoding="ascii")
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode != 0, (stdout, stderr)
        assert "pre-import target directory" in stderr
        assert "SWAPPED_NESTED_TARGET_EXECUTED" not in stdout
    finally:
        if process.poll() is None:
            gate.write_text("continue", encoding="ascii")
            process.kill()
            process.communicate()


def test_causal_source_hash_rejects_unsafe_snapshot_fd_forms_and_races(
    causal_parity_audit, tmp_path, monkeypatch
):
    relative = causal_parity_audit.AUDIT_SOURCE_RELATIVE
    roots_to_restore: list[Path] = []

    def sealed_root(name: str, payload: bytes = b"causal-source\n"):
        root = tmp_path / name
        target = root / relative
        target.parent.mkdir(parents=True)
        target.write_bytes(payload)
        target.chmod(0o444)
        directories = [
            path
            for path in (root, target.parent, *target.parent.parents)
            if path == root or path.is_relative_to(root)
        ]
        for directory in sorted(set(directories), key=lambda path: len(path.parts), reverse=True):
            directory.chmod(0o555)
        roots_to_restore.extend(directories)
        return root, target

    root, target = sealed_root("bound")
    wrong_root, _wrong_target = sealed_root("wrong")
    symlink_root = tmp_path / "symlink-root"
    symlink_root.mkdir()
    (symlink_root / relative.parts[0]).symlink_to(
        root / relative.parts[0], target_is_directory=True
    )
    symlink_root.chmod(0o555)
    roots_to_restore.append(symlink_root)
    root_fd = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    relative_text = relative.as_posix()
    bound_source = f"/proc/self/fd/{root_fd}/{relative_text}"
    try:
        monkeypatch.setattr(causal_parity_audit, "PROJECT_ROOT", root)
        bind_causal_execution_context(
            causal_parity_audit, monkeypatch, root_fd, root_fd, True, target
        )
        assert causal_parity_audit.file_sha256(bound_source) == hashlib.sha256(
            target.read_bytes()
        ).hexdigest()

        root.chmod(0o755)
        try:
            with pytest.raises(causal_parity_audit.ParityAuditError, match="mode differs"):
                causal_parity_audit.file_sha256(bound_source)
        finally:
            root.chmod(0o555)
        bind_causal_execution_context(
            causal_parity_audit, monkeypatch, root_fd, root_fd, True, target
        )

        malformed = (
            f"/proc/self/fd/{root_fd}",
            f"/proc/self/fd/notdecimal/{relative_text}",
            f"/proc/self/fd/0{root_fd}/{relative_text}",
            f"/proc/self/fd/{root_fd}/../{relative_text}",
            f"/proc/self/fd/{root_fd}//{relative_text}",
            f"/proc/self/fd/{root_fd}/wrong.py",
            f"/proc/{os.getpid()}/fd/{root_fd}/{relative_text}",
        )
        for value in malformed:
            with pytest.raises(causal_parity_audit.ParityAuditError):
                causal_parity_audit.file_sha256(value)

        closed_fd = os.dup(root_fd)
        bind_causal_execution_context(
            causal_parity_audit, monkeypatch, closed_fd, root_fd, True, target
        )
        os.close(closed_fd)
        with pytest.raises(causal_parity_audit.ParityAuditError, match="unavailable"):
            causal_parity_audit.file_sha256(
                f"/proc/self/fd/{closed_fd}/{relative_text}"
            )

        file_fd = os.open(target, os.O_RDONLY)
        try:
            bind_causal_execution_context(
                causal_parity_audit, monkeypatch, file_fd, root_fd, True, target
            )
            with pytest.raises(causal_parity_audit.ParityAuditError, match="type, owner, or mode"):
                causal_parity_audit.file_sha256(
                    f"/proc/self/fd/{file_fd}/{relative_text}"
                )
        finally:
            os.close(file_fd)

        wrong_fd = os.open(wrong_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            bind_causal_execution_context(
                causal_parity_audit, monkeypatch, wrong_fd, root_fd, True, target
            )
            with pytest.raises(causal_parity_audit.ParityAuditError, match="identity differs"):
                causal_parity_audit.file_sha256(
                    f"/proc/self/fd/{wrong_fd}/{relative_text}"
                )
        finally:
            os.close(wrong_fd)

        reused_token = os.dup(root_fd)
        os.close(reused_token)
        replacement_fd = os.open(
            wrong_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        reused_is_replacement = replacement_fd == reused_token
        if not reused_is_replacement:
            os.dup2(replacement_fd, reused_token, inheritable=False)
        try:
            bind_causal_execution_context(
                causal_parity_audit,
                monkeypatch,
                reused_token,
                root_fd,
                True,
                target,
            )
            with pytest.raises(causal_parity_audit.ParityAuditError, match="identity differs"):
                causal_parity_audit.file_sha256(
                    f"/proc/self/fd/{reused_token}/{relative_text}"
                )
        finally:
            os.close(reused_token)
            if not reused_is_replacement:
                os.close(replacement_fd)

        monkeypatch.setattr(causal_parity_audit, "PROJECT_ROOT", symlink_root)
        symlink_fd = os.open(
            symlink_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            bind_causal_execution_context(
                causal_parity_audit,
                monkeypatch,
                symlink_fd,
                symlink_fd,
                True,
                target,
            )
            with pytest.raises(causal_parity_audit.ParityAuditError, match="cannot open"):
                causal_parity_audit.file_sha256(
                    f"/proc/self/fd/{symlink_fd}/{relative_text}"
                )
        finally:
            os.close(symlink_fd)

        monkeypatch.setattr(causal_parity_audit, "PROJECT_ROOT", root)
        bind_causal_execution_context(
            causal_parity_audit, monkeypatch, root_fd, root_fd, True, target
        )
        real_read = causal_parity_audit.os.read
        mutated = False

        def replace_named_source(descriptor, size):
            nonlocal mutated
            block = real_read(descriptor, size)
            if block and not mutated:
                mutated = True
                target.parent.chmod(0o755)
                target.rename(target.with_suffix(".authenticated"))
                target.write_bytes(b"replacement!!\n")
                target.chmod(0o444)
                target.parent.chmod(0o555)
            return block

        monkeypatch.setattr(causal_parity_audit.os, "read", replace_named_source)
        with pytest.raises(causal_parity_audit.ParityAuditError, match="changed while hashing"):
            causal_parity_audit.file_sha256(bound_source)
        assert mutated
    finally:
        os.close(root_fd)
        for directory in roots_to_restore:
            directory.chmod(0o755)


def test_causal_source_directory_fstat_failures_do_not_leak_fds(
    causal_parity_audit, tmp_path, monkeypatch
):
    def fd_count():
        return len(os.listdir("/proc/self/fd"))

    before = fd_count()
    real_open = os.open
    real_fstat = os.fstat
    tracked: set[int] = set()

    def track_component_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if kwargs.get("dir_fd") is not None:
            tracked.add(descriptor)
        return descriptor

    def fail_tracked_fstat(descriptor):
        if descriptor in tracked:
            raise OSError("injected component fstat failure")
        return real_fstat(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(causal_parity_audit.os, "open", track_component_open)
        patch.setattr(causal_parity_audit.os, "fstat", fail_tracked_fstat)
        with pytest.raises(causal_parity_audit.ParityAuditError, match="cannot open"):
            causal_parity_audit._open_absolute_directory(tmp_path, "test directory")
    assert fd_count() == before

    relative = causal_parity_audit.AUDIT_SOURCE_RELATIVE
    root = tmp_path / "live"
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fd-leak-test\n")
    monkeypatch.setattr(causal_parity_audit, "PROJECT_ROOT", root)
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    bind_causal_execution_context(
        causal_parity_audit, monkeypatch, root_fd, root_fd, False, source
    )
    root_info = os.fstat(root_fd)
    before = fd_count()
    tracked.clear()

    def expected_root(_path, _label):
        return os.dup(root_fd), root_info

    with monkeypatch.context() as patch:
        patch.setattr(causal_parity_audit, "_open_absolute_directory", expected_root)
        patch.setattr(causal_parity_audit.os, "open", track_component_open)
        patch.setattr(causal_parity_audit.os, "fstat", fail_tracked_fstat)
        with pytest.raises(causal_parity_audit.ParityAuditError, match="cannot open"):
            causal_parity_audit.file_sha256(
                f"/proc/self/fd/{root_fd}/{relative.as_posix()}"
            )
    assert fd_count() == before
    os.close(root_fd)


def test_causal_live_source_hash_rejects_writable_permissions(
    causal_parity_audit, tmp_path, monkeypatch
):
    relative = causal_parity_audit.AUDIT_SOURCE_RELATIVE
    root = tmp_path / "live"
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"live-causal-source\n")
    monkeypatch.setattr(causal_parity_audit, "PROJECT_ROOT", root)
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    fd_source = f"/proc/self/fd/{root_fd}/{relative.as_posix()}"
    try:
        bind_causal_execution_context(
            causal_parity_audit, monkeypatch, root_fd, root_fd, False, source
        )
        assert causal_parity_audit.file_sha256(fd_source) == hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        for path, unsafe_mode, safe_mode in (
            (root, 0o775, 0o755),
            (source.parent, 0o775, 0o755),
            (source, 0o664, 0o644),
        ):
            bind_causal_execution_context(
                causal_parity_audit, monkeypatch, root_fd, root_fd, False, source
            )
            path.chmod(unsafe_mode)
            try:
                with pytest.raises(causal_parity_audit.ParityAuditError):
                    causal_parity_audit.file_sha256(fd_source)
            finally:
                path.chmod(safe_mode)
    finally:
        os.close(root_fd)


@pytest.mark.parametrize("surface", ("root", "directory", "file"))
def test_causal_live_source_hash_rejects_mode_change_during_read(
    causal_parity_audit, tmp_path, monkeypatch, surface
):
    relative = causal_parity_audit.AUDIT_SOURCE_RELATIVE
    root = tmp_path / surface
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"live-causal-mode-race\n")
    monkeypatch.setattr(causal_parity_audit, "PROJECT_ROOT", root)
    target = {"root": root, "directory": source.parent, "file": source}[surface]
    changed_mode = {"root": 0o700, "directory": 0o700, "file": 0o600}[surface]
    original_mode = stat.S_IMODE(target.stat().st_mode)
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    fd_source = f"/proc/self/fd/{root_fd}/{relative.as_posix()}"
    real_read = causal_parity_audit.os.read
    mutated = False
    bind_causal_execution_context(
        causal_parity_audit, monkeypatch, root_fd, root_fd, False, source
    )

    def change_mode(descriptor, size):
        nonlocal mutated
        block = real_read(descriptor, size)
        if block and not mutated:
            mutated = True
            target.chmod(changed_mode)
        return block

    monkeypatch.setattr(causal_parity_audit.os, "read", change_mode)
    try:
        with pytest.raises(causal_parity_audit.ParityAuditError, match="changed while hashing"):
            causal_parity_audit.file_sha256(fd_source)
        assert mutated
    finally:
        os.close(root_fd)
        target.chmod(original_mode)


@pytest.mark.parametrize("surface", ("snapshot", "venv-site"))
def test_isolated_bootstrap_pins_import_surfaces_after_preimport_validation(
    submit, tmp_path, surface
):
    identity = interpreter_identity(submit)
    root = tmp_path / "snapshot"
    vsite = tmp_path / "venv-site"
    bsite = tmp_path / "base-site"
    for directory in (root, vsite, bsite):
        directory.mkdir()
    identity["venv_site_packages"] = str(vsite)
    identity["base_site_packages"] = str(bsite)
    marker = tmp_path / "target-started"
    gate = tmp_path / "continue-import"
    main_source = (
        "import os, sys, time\n"
        "with open(sys.argv[1], 'w', encoding='utf-8') as stream: stream.write('ready')\n"
        "while not os.path.exists(sys.argv[2]): time.sleep(0.01)\n"
        "import exp23_unclaimed_swap_helper_7fd3\n"
    )
    main = root / "main.py"
    main.write_text(main_source, encoding="utf-8")
    inventory = {"main.py": submit.file_sha256(main)}
    main.chmod(0o444)
    root.chmod(0o555)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    if surface == "snapshot":
        (replacement / "main.py").write_text(main_source, encoding="utf-8")
        (replacement / "main.py").chmod(0o444)
    (replacement / "exp23_unclaimed_swap_helper_7fd3.py").write_text(
        "print('HELPER=INJECTED')\n", encoding="utf-8"
    )
    (replacement / "exp23_unclaimed_swap_helper_7fd3.py").chmod(0o444)
    replacement.chmod(0o555 if surface == "snapshot" else 0o755)
    command = submit.isolated_python_command(
        [identity["lexical_executable"], str(main), str(marker), str(gate)],
        root,
        identity,
        intercept_python_children=False,
        snapshot_inventory=inventory,
    )
    process = subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        deadline = time.monotonic() + 5.0
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), process.communicate(timeout=1)[1]
        if surface == "snapshot":
            root.rename(tmp_path / "authenticated-snapshot")
            replacement.rename(root)
        else:
            vsite.rename(tmp_path / "authenticated-venv-site")
            replacement.rename(vsite)
        gate.write_text("continue", encoding="ascii")
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode != 0, (stdout, stderr)
        assert "HELPER=INJECTED" not in stdout
    finally:
        if process.poll() is None:
            gate.write_text("continue", encoding="ascii")
            process.kill()
            process.communicate()
        for directory in tmp_path.iterdir():
            if directory.is_dir() and not directory.is_symlink():
                directory.chmod(0o755)


def test_snapshot_inventory_rejects_parent_symlink(submit, tmp_path):
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    package = root / submit.PACKAGE_RELATIVE
    package.mkdir(parents=True)
    outside.mkdir()
    (outside / "payload.py").write_text("pass\n", encoding="utf-8")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    (package / "protocol.sha256").write_text("0" * 64 + "\n", encoding="ascii")
    digest = submit.file_sha256(outside / "payload.py")
    fake = SimpleNamespace(
        source_contract=lambda _root: {
            "source_files": {"linked/payload.py": digest},
            "source_sha256": "0" * 64,
            "runtime_sha256": "1" * 64,
        },
        protocol_sha256=lambda _package: "2" * 64,
        PROTOCOL_FILES=(),
        SNAPSHOT_IMPORT_FILES={},
    )
    with pytest.raises(submit.SubmissionError, match="symlink"):
        submit.snapshot_inventory(root, fake, "2" * 64)


def test_frozen_inventory_closes_protocol_verification_copy_window(
    submit, tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    package = root / submit.PACKAGE_RELATIVE
    package.mkdir(parents=True)
    core = root / "core.py"
    core.write_text("CORE = 1\n", encoding="utf-8")
    protected = package / "submit.py"
    protected.write_text("SAFE = True\n", encoding="utf-8")
    (package / "manifest.json").write_text("{}\n", encoding="utf-8")
    protocol_names = ("manifest.json", "submit.py")
    protocol_files = {
        name: submit.file_sha256(package / name) for name in protocol_names
    }
    protocol = submit.stable_hash(
        {"schema_version": 1, "files": protocol_files}
    )
    (package / "protocol.sha256").write_text(protocol + "\n", encoding="ascii")
    source_contract = {
        "source_files": {"core.py": submit.file_sha256(core)},
        "source_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
    }
    campaign = SimpleNamespace(
        PROTOCOL_FILES=protocol_names,
        SNAPSHOT_IMPORT_FILES={},
        source_contract=lambda _root: source_contract,
    )
    real_hash = submit._hash_relative_regular
    attacked = False

    def mutate_after_verified_hash(root_arg, relative, label, *, capture=False):
        nonlocal attacked
        result = real_hash(root_arg, relative, label, capture=capture)
        if relative == submit.PACKAGE_RELATIVE / "submit.py" and not attacked:
            attacked = True
            protected.write_text("MALICIOUS = True\n", encoding="utf-8")
        return result

    monkeypatch.setattr(submit, "_hash_relative_regular", mutate_after_verified_hash)
    inventory = submit.snapshot_inventory(
        root, campaign, protocol, source_contract=source_contract
    )
    assert attacked
    assert inventory[str(submit.PACKAGE_RELATIVE / "submit.py")] == protocol_files[
        "submit.py"
    ]
    campaign.PROTOCOL_FILES = ("manifest.json",)
    with pytest.raises(submit.SubmissionError, match="source bytes changed"):
        submit.create_source_snapshot(root, tmp_path / "sealed" / "repo", inventory)


def test_submit_implementation_uses_only_preflight_inventory(submit):
    source = inspect.getsource(submit._submit_campaign_impl)
    assert "_validated_preflight_inventory(preflight)" in source
    assert "snapshot_inventory(" not in source
    assert "load_campaign(" not in source


def test_snapshot_verifier_rejects_extra_directory_and_writable_root(submit, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    artifact = snapshot / "x"
    artifact.write_bytes(b"x")
    artifact.chmod(0o444)
    snapshot.chmod(0o555)
    inventory = {"x": submit.file_sha256(artifact)}
    submit.verify_snapshot_files(snapshot, inventory)
    snapshot.chmod(0o755)
    with pytest.raises(submit.SubmissionError, match="writable"):
        submit.verify_snapshot_files(snapshot, inventory)
    extra = snapshot / "extra"
    extra.mkdir()
    snapshot.chmod(0o555)
    extra.chmod(0o555)
    with pytest.raises(submit.SubmissionError, match="directory coverage"):
        submit.verify_snapshot_files(snapshot, inventory)


def test_child_cache_tree_is_private_contained_and_ephemeral(submit, tmp_path):
    hostile_tmp = tmp_path / "must-not-be-used"
    cache_root = None
    with submit._ephemeral_child_environment(
        environ={"PATH": "/usr/bin:/bin", "TMPDIR": str(hostile_tmp)}
    ) as environment:
        directories = {
            key: Path(environment[key])
            for key in submit.EPHEMERAL_CHILD_DIRECTORIES
        }
        cache_root = directories["HOME"].parent
        assert cache_root.is_dir()
        assert cache_root.parent == Path("/tmp")
        assert all(path.parent == cache_root for path in directories.values())
        assert all(path.is_dir() and os.access(path, os.W_OK) for path in directories.values())
        assert all("/proc/" not in str(path) for path in directories.values())
        for index, path in enumerate(directories.values()):
            (path / f"cache-{index}").write_text("ephemeral", encoding="utf-8")
    assert cache_root is not None and not cache_root.exists()
    assert not hostile_tmp.exists()


def test_sealed_snapshot_preflight_uses_live_inputs_and_ephemeral_cache(
    submit, tmp_path
):
    """Exercise the copied-tree subprocess mode that live --test-only omits."""

    input_root = tmp_path / "live-repository"
    exp20_manifest = (
        input_root
        / "experiments/20-treewm-grounded-gauge-pilot-v2/manifest.json"
    )
    exp20_manifest.parent.mkdir(parents=True)
    exp20_manifest.write_text('{"settings": []}\n', encoding="utf-8")
    exp20_outputs = input_root / "outputs/treewm-grounded-gauge-pilot-v2-launch2"
    exp20_outputs.mkdir(parents=True)
    live_before = sorted(
        (str(path.relative_to(input_root)), path.stat().st_mode, path.stat().st_size)
        for path in input_root.rglob("*")
    )

    source = tmp_path / "snapshot-source"
    package = source / submit.PACKAGE_RELATIVE
    package.mkdir(parents=True)
    manifest = {
        "paths": {
            "python": json.loads((PACKAGE / "manifest.json").read_text())["paths"][
                "python"
            ],
            "prospective_run_root": "outputs/snapshot-regression",
            "run_root": str(input_root / "outputs/snapshot-regression"),
        }
    }
    (package / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    probe = r'''#!/usr/bin/env python3
import argparse, hashlib, json, os, pathlib, stat

parser = argparse.ArgumentParser()
parser.add_argument("--_snapshot-preflight", action="store_true", required=True)
parser.add_argument("--snapshot-root", type=pathlib.Path, required=True)
parser.add_argument("--audit-input-root", type=pathlib.Path, required=True)
parser.add_argument("--protocol-sha256", required=True)
parser.add_argument("--inventory-json", required=True)
args = parser.parse_args()
root = args.snapshot_root.resolve(strict=True)
inputs = args.audit_input_root.resolve(strict=True)
assert pathlib.Path(__file__).resolve(strict=True).is_relative_to(root)
assert inputs != root
assert (inputs / "experiments/20-treewm-grounded-gauge-pilot-v2/manifest.json").is_file()
assert (inputs / "outputs/treewm-grounded-gauge-pilot-v2-launch2").is_dir()
assert not (root.stat().st_mode & 0o222)
inventory = json.loads(args.inventory_json)
for relative, expected in inventory.items():
    path = root / relative
    assert path.is_file() and not path.is_symlink()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
cache_variables = (
    "HOME", "TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
    "XDG_DATA_HOME", "XDG_STATE_HOME", "TORCH_HOME", "HF_HOME",
    "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR",
)
cache_paths = [pathlib.Path(os.environ[name]).resolve(strict=True) for name in cache_variables]
cache_root = cache_paths[0].parent
assert all(path.parent == cache_root and path.is_dir() for path in cache_paths)
assert all("/proc/" not in str(path) for path in cache_paths)
for index, path in enumerate(cache_paths):
    (path / f"probe-{index}").write_text("temporary", encoding="utf-8")
manifest = json.loads((pathlib.Path(__file__).parent / "manifest.json").read_text())
value = {
    "manifest": manifest,
    "launches": [{} for _ in range(20)],
    "compositions": [],
    "cache_root": str(cache_root),
    "verification": {
        "audit_replays": {},
        "scientific_output_fingerprint_before": "same",
        "scientific_output_fingerprint_after": "same",
        "import_containment": "all_treewm_modules_inside_snapshot",
        "audit_input_root": str(inputs),
    },
}
print("EXP23_SNAPSHOT_PREFLIGHT=" + json.dumps(value, sort_keys=True, separators=(",", ":")))
'''
    (package / "submit.py").write_text(probe, encoding="utf-8")
    inventory = {
        str(path.relative_to(source)): submit.file_sha256(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    sealed = tmp_path / "sealed" / "repo"
    try:
        submit.create_source_snapshot(source, sealed, inventory)
        result = submit._snapshot_preflight(
            sealed,
            input_root,
            "a" * 64,
            inventory,
            runner=submit._default_runner,
        )
        assert result["verification"]["audit_input_root"] == str(input_root)
        assert not Path(result["cache_root"]).exists()
        live_after = sorted(
            (str(path.relative_to(input_root)), path.stat().st_mode, path.stat().st_size)
            for path in input_root.rglob("*")
        )
        assert live_after == live_before
    finally:
        if sealed.exists():
            for directory in sorted(
                (path for path in sealed.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
            ):
                directory.chmod(0o755)
            sealed.chmod(0o755)
        if sealed.parent.exists():
            sealed.parent.chmod(0o755)


def test_snapshot_audit_command_separates_execution_and_input_roots(
    submit, tmp_path, monkeypatch
):
    snapshot = tmp_path / "snapshot"
    package = snapshot / submit.PACKAGE_RELATIVE
    package.mkdir(parents=True)
    weight_lock = json.loads((PACKAGE / "weight_audit.lock.json").read_text())
    (package / "weight_audit.lock.json").write_text(
        json.dumps(weight_lock), encoding="utf-8"
    )
    (package / "weight_audit.py").write_text("raise AssertionError('runner stub')\n")
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    identity = interpreter_identity(submit)
    checked = {}
    cache_roots = []
    commands = []

    def verify_inputs(execution_root, input_root, lock):
        checked["execution_root"] = execution_root
        checked["input_root"] = input_root
        assert lock == weight_lock

    def runner(command, cwd, environment, _timeout):
        commands.append(list(command))
        assert cwd == snapshot
        cache_root = Path(environment["MPLCONFIGDIR"]).parent
        cache_roots.append(cache_root)
        assert cache_root.is_dir()
        (Path(environment["MPLCONFIGDIR"]) / "fontlist-test.json").write_text(
            "temporary", encoding="utf-8"
        )
        result_identity = weight_lock["result_identity"]
        result = {
            "artifact_sha256": result_identity["artifact_sha256"],
            "rows_sha256": result_identity["rows_sha256"],
            "summary_sha256": result_identity["summary_sha256"],
            "row_count": result_identity["row_count"],
        }
        stdout = (
            "EXP23_WEIGHT_AUDIT_SUMMARY=" + submit.canonical_json(result) + "\n"
        ).encode("utf-8")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(submit, "interpreter_contract", lambda _manifest: identity)
    monkeypatch.setattr(submit, "_verify_external_audit_inputs", verify_inputs)
    monkeypatch.setattr(
        submit,
        "AUDITS",
        (("weight", "weight_audit.py", ("--summary-only",), "EXP23_WEIGHT_AUDIT_SUMMARY="),),
    )
    records, _results = submit.rerun_audit_locks(
        snapshot,
        SimpleNamespace(),
        manifest,
        audit_input_root=REPO,
        runner=runner,
        snapshot_inventory={
            str(submit.PACKAGE_RELATIVE / "weight_audit.lock.json"):
                submit.file_sha256(package / "weight_audit.lock.json")
        },
    )
    assert set(records) == {"weight"}
    assert checked == {"execution_root": snapshot, "input_root": REPO.resolve()}
    assert len(commands) == 1
    assert str(package / "weight_audit.py") in commands[0]
    project_index = commands[0].index("--project-root")
    assert commands[0][project_index + 1] == str(REPO.resolve())
    lock_index = commands[0].index("--weight-lock-sha256")
    assert commands[0][lock_index + 1] == submit.file_sha256(
        package / "weight_audit.lock.json"
    )
    assert all(not root.exists() for root in cache_roots)


def test_live_static_audit_executes_from_writable_repo_with_frozen_lock(
    submit, tmp_path, monkeypatch
):
    """The live static path binds its lock without applying sealed-tree rules."""

    repo = tmp_path / "writable-live-repository"
    package = repo / submit.PACKAGE_RELATIVE
    package.mkdir(parents=True)
    identity = {
        "artifact_sha256": "a" * 64,
        "rows_sha256": "b" * 64,
        "summary_sha256": "c" * 64,
        "row_count": 40,
    }
    lock_path = package / "weight_audit.lock.json"
    lock_path.write_text(
        json.dumps({"result_identity": identity}), encoding="utf-8"
    )
    program = package / "weight_audit.py"
    program.write_text(
        "import argparse,json\n"
        "p=argparse.ArgumentParser(); p.add_argument('--summary-only',action='store_true'); "
        "p.add_argument('--project-root'); p.add_argument('--weight-lock-sha256'); p.parse_args()\n"
        f"print('EXP23_WEIGHT_AUDIT_SUMMARY='+json.dumps({identity!r},sort_keys=True,separators=(',',':')))\n",
        encoding="utf-8",
    )
    (repo / "extra-live-file.txt").write_text("allowed outside snapshot union\n")
    pinned_manifest = json.loads((PACKAGE / "manifest.json").read_text())
    manifest = {
        "paths": {
            "python": pinned_manifest["paths"]["python"],
            "prospective_run_root": "outputs/live-static-fixture",
            "run_root": str(repo / "outputs/live-static-fixture"),
        }
    }
    inventory = {
        str(submit.PACKAGE_RELATIVE / lock_path.name): submit.file_sha256(lock_path)
    }
    monkeypatch.setattr(submit, "_verify_external_audit_inputs", lambda *_args: None)
    monkeypatch.setattr(
        submit,
        "AUDITS",
        (("weight", "weight_audit.py", ("--summary-only",), "EXP23_WEIGHT_AUDIT_SUMMARY="),),
    )
    records, _results = submit.rerun_audit_locks(
        repo,
        SimpleNamespace(),
        manifest,
        audit_input_root=repo,
        runner=submit._default_runner,
        snapshot_inventory=inventory,
    )
    assert records["weight"]["artifact_sha256"] == "a" * 64
    assert repo.stat().st_mode & 0o200
    assert (repo / "extra-live-file.txt").is_file()


@pytest.mark.parametrize("mode", ["missing", "wrong"])
def test_audit_replay_rejects_missing_or_wrong_frozen_weight_lock_hash(
    submit, tmp_path, monkeypatch, mode
):
    snapshot = tmp_path / "snapshot"
    package = snapshot / submit.PACKAGE_RELATIVE
    package.mkdir(parents=True)
    lock_path = package / "weight_audit.lock.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    identity = interpreter_identity(submit)
    relative = str(submit.PACKAGE_RELATIVE / lock_path.name)
    inventory = (
        {"unrelated.json": "a" * 64}
        if mode == "missing"
        else {relative: "b" * 64}
    )
    called = False

    def forbidden_runner(*_args):
        nonlocal called
        called = True
        raise AssertionError("unbound lock reached audit runner")

    monkeypatch.setattr(submit, "interpreter_contract", lambda _manifest: identity)
    monkeypatch.setattr(submit, "_verify_external_audit_inputs", lambda *_args: None)
    with pytest.raises(submit.SubmissionError, match="hash is malformed|differs"):
        submit.rerun_audit_locks(
            snapshot,
            SimpleNamespace(),
            manifest,
            audit_input_root=REPO,
            runner=forbidden_runner,
            snapshot_inventory=inventory,
        )
    assert called is False


def _checkpoint_fixture(tmp_path: Path, payload: bytes = b"sealed checkpoint"):
    run_dir = tmp_path / "setting" / "treewm" / "run-armgs-seed108"
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(payload)
    return run_dir, checkpoint


def test_checkpoint_run_discovery_requires_exactly_one_regular_directory(
    weight_audit, tmp_path
):
    output = tmp_path / "outputs"
    tree = output / "antmaze-large" / "treewm"
    (tree / "one-armgs-seed108").mkdir(parents=True)
    (tree / "two-armgs-seed108").mkdir()
    with pytest.raises(weight_audit.AuditError, match="exactly one"):
        weight_audit.exact_checkpoint_run(output, "antmaze-large", 108)


def test_checkpoint_hash_mismatch_never_reaches_unsafe_load(weight_audit, tmp_path):
    run_dir, _checkpoint = _checkpoint_fixture(tmp_path, b"untrusted pickle")
    calls = []

    class FakeTorch:
        @staticmethod
        def load(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("unverified checkpoint reached torch.load")

    with pytest.raises(weight_audit.AuditError, match="SHA256 differs"):
        weight_audit.load_frozen_checkpoint(
            run_dir, hashlib.sha256(b"different").hexdigest(), FakeTorch
        )
    assert calls == []


def test_same_inode_checkpoint_mutation_never_reaches_unsafe_load(
    weight_audit, tmp_path, monkeypatch
):
    original = b"A" * 4096
    run_dir, checkpoint = _checkpoint_fixture(tmp_path, original)
    expected = hashlib.sha256(original).hexdigest()
    real_read = weight_audit.os.read
    attacked = False
    calls = []

    def mutate_during_read(descriptor, size):
        nonlocal attacked
        # Force more than one source read so an in-place edit changes bytes that
        # have not yet entered the authenticated private copy.
        block = real_read(descriptor, min(size, 1024))
        if block and not attacked:
            attacked = True
            with checkpoint.open("r+b") as handle:
                handle.seek(0)
                handle.write(b"B" * len(original))
                handle.flush()
                os.fsync(handle.fileno())
        return block

    class FakeTorch:
        @staticmethod
        def load(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("raced checkpoint reached torch.load")

    monkeypatch.setattr(weight_audit.os, "read", mutate_during_read)
    with pytest.raises(weight_audit.AuditError, match="changed while|SHA256 differs"):
        weight_audit.load_frozen_checkpoint(run_dir, expected, FakeTorch)
    assert attacked and calls == []


def test_verified_checkpoint_loads_only_from_private_copy(weight_audit, tmp_path):
    payload = b"authenticated checkpoint bytes"
    run_dir, checkpoint = _checkpoint_fixture(tmp_path, payload)
    seen = []

    class FakeTorch:
        @staticmethod
        def load(handle, **kwargs):
            assert hasattr(handle, "read")
            assert Path(str(getattr(handle, "name", ""))) != checkpoint
            assert kwargs == {"map_location": "cpu", "weights_only": False}
            seen.append(handle.read())
            return {"verified": True}

    result, digest = weight_audit.load_frozen_checkpoint(
        run_dir, hashlib.sha256(payload).hexdigest(), FakeTorch
    )
    assert result == {"verified": True}
    assert digest == hashlib.sha256(payload).hexdigest()
    assert seen == [payload]


def test_checkpoint_and_weight_lock_symlinks_are_rejected_before_load(
    weight_audit, tmp_path, monkeypatch
):
    run_dir, checkpoint = _checkpoint_fixture(tmp_path)
    target = tmp_path / "outside.pt"
    target.write_bytes(checkpoint.read_bytes())
    checkpoint.unlink()
    checkpoint.symlink_to(target)
    calls = []

    class FakeTorch:
        @staticmethod
        def load(*args, **kwargs):
            calls.append((args, kwargs))

    with pytest.raises(OSError):
        weight_audit.load_frozen_checkpoint(
            run_dir, hashlib.sha256(target.read_bytes()).hexdigest(), FakeTorch
        )
    assert calls == []

    real_lock = tmp_path / "real-lock.json"
    real_lock.write_text("{}\n", encoding="utf-8")
    linked_lock = tmp_path / "weight_audit.lock.json"
    linked_lock.symlink_to(real_lock)
    monkeypatch.setattr(weight_audit, "CHECKPOINT_LOCK_PATH", linked_lock)
    with pytest.raises(OSError):
        weight_audit.load_weight_lock(hashlib.sha256(real_lock.read_bytes()).hexdigest())


def _external_control_fixture(weight_audit, tmp_path):
    project = tmp_path / "repository"
    manifest = (
        project
        / "experiments"
        / "20-treewm-grounded-gauge-pilot-v2"
        / "manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"settings": [{"id": value} for value in weight_audit.SETTINGS]}),
        encoding="utf-8",
    )
    mapping = {"exp20/manifest.json": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    launches = {}
    output = project / "outputs" / "treewm-grounded-gauge-pilot-v2-launch2"
    for setting in weight_audit.SETTINGS:
        launches[setting] = {}
        for seed in weight_audit.CHECKPOINT_SEEDS:
            run_dir = output / setting / "treewm" / f"fixture-armgs-seed{seed}"
            run_dir.mkdir(parents=True)
            launch = run_dir / "GAUGE_PILOT_V2_LAUNCH.json"
            launch.write_text(
                json.dumps({"setting": setting, "seed": seed}), encoding="utf-8"
            )
            key = f"{setting}/seed{seed}/GAUGE_PILOT_V2_LAUNCH.json"
            mapping[key] = hashlib.sha256(launch.read_bytes()).hexdigest()
            launches[setting][seed] = launch
    return project, manifest, launches, {"external_input_sha256": mapping}


@pytest.mark.parametrize("mutation", ["whitespace", "unused_field"])
def test_launch_exact_byte_mutation_rejected_before_checkpoint_use(
    weight_audit, tmp_path, mutation
):
    project, _manifest, launches, lock = _external_control_fixture(
        weight_audit, tmp_path
    )
    launch = launches[weight_audit.SETTINGS[0]][108]
    if mutation == "whitespace":
        launch.write_bytes(launch.read_bytes() + b"\n")
    else:
        value = json.loads(launch.read_text())
        value["unused"] = True
        launch.write_text(json.dumps(value), encoding="utf-8")
    unsafe_load_calls = []
    with pytest.raises(weight_audit.AuditError, match="SHA256 differs"):
        weight_audit.load_frozen_external_inputs(project, lock)
        unsafe_load_calls.append("checkpoint use")
    assert unsafe_load_calls == []


def test_manifest_and_launch_component_attacks_are_rejected_before_use(
    weight_audit, tmp_path
):
    project, manifest, launches, lock = _external_control_fixture(
        weight_audit, tmp_path
    )
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(weight_audit.AuditError, match="SHA256 differs"):
        weight_audit.load_frozen_external_inputs(project, lock)

    project, _manifest, launches, lock = _external_control_fixture(
        weight_audit, tmp_path / "second"
    )
    launch = launches[weight_audit.SETTINGS[0]][108]
    outside = tmp_path / "outside-launch.json"
    outside.write_bytes(launch.read_bytes())
    launch.unlink()
    launch.symlink_to(outside)
    with pytest.raises(OSError):
        weight_audit.load_frozen_external_inputs(project, lock)

    project, _manifest, _launches, lock = _external_control_fixture(
        weight_audit, tmp_path / "third"
    )
    tree_root = (
        project
        / "outputs"
        / "treewm-grounded-gauge-pilot-v2-launch2"
        / weight_audit.SETTINGS[0]
        / "treewm"
    )
    actual_tree = tree_root.with_name("treewm-real")
    tree_root.rename(actual_tree)
    tree_root.symlink_to(actual_tree, target_is_directory=True)
    with pytest.raises(OSError):
        weight_audit.load_frozen_external_inputs(project, lock)


def test_same_inode_control_json_mutation_is_rejected_before_use(
    weight_audit, tmp_path, monkeypatch
):
    project, _manifest, launches, lock = _external_control_fixture(
        weight_audit, tmp_path
    )
    setting = weight_audit.SETTINGS[0]
    launch = launches[setting][108]
    expected = lock["external_input_sha256"][
        f"{setting}/seed108/GAUGE_PILOT_V2_LAUNCH.json"
    ]
    original = launch.read_bytes()
    real_read = weight_audit.os.read
    attacked = False

    def mutate_unread_bytes(descriptor, size):
        nonlocal attacked
        block = real_read(descriptor, min(size, 8))
        if block and not attacked:
            attacked = True
            with launch.open("r+b") as handle:
                handle.seek(8)
                handle.write(b"X" * max(len(original) - 8, 0))
                handle.flush()
                os.fsync(handle.fileno())
        return block

    monkeypatch.setattr(weight_audit.os, "read", mutate_unread_bytes)
    with pytest.raises(weight_audit.AuditError, match="changed while|SHA256 differs"):
        weight_audit.read_json(
            launch,
            expected_sha256=expected,
            label="mutated frozen launch",
        )
    assert attacked


def test_missing_or_swapped_weight_lock_rejected_before_input_use(
    weight_audit, tmp_path, monkeypatch
):
    lock_path = tmp_path / "weight_audit.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "checkpoint_sha256": {},
                "external_input_sha256": {},
            }
        ),
        encoding="utf-8",
    )
    expected = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    monkeypatch.setattr(weight_audit, "CHECKPOINT_LOCK_PATH", lock_path)
    loaded = weight_audit.load_weight_lock(expected)
    with pytest.raises(weight_audit.AuditError, match="checkpoint hash map"):
        weight_audit.frozen_checkpoint_sha256(loaded)
    with pytest.raises(weight_audit.AuditError, match="external-input hash map"):
        weight_audit.frozen_external_input_sha256(loaded)
    lock_path.write_text('{"swapped":true}\n', encoding="utf-8")
    with pytest.raises(weight_audit.AuditError, match="SHA256 differs"):
        weight_audit.load_weight_lock(expected)


def test_prefix_audit_reuses_authenticated_weight_and_external_maps(
    prefix_target_audit,
):
    source = inspect.getsource(prefix_target_audit.run)
    assert "audit.load_weight_lock(expected_weight_lock_sha256)" in source
    assert "audit.load_frozen_external_inputs(project_root, weight_lock)" in source
    assert "audit.load_frozen_checkpoint(" in source
    assert "torch.load(" not in source


def test_weight_leaf_and_checkpoint_fifo_reject_promptly(weight_audit, tmp_path):
    fifo = tmp_path / "artifact.fifo"
    os.mkfifo(fifo)
    with pytest.raises(weight_audit.AuditError):
        weight_audit.file_sha256(fifo)

    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    os.mkfifo(checkpoints / "latest.pt")
    with pytest.raises(weight_audit.AuditError):
        weight_audit._open_checkpoint(run)


def test_weight_private_checkpoint_detects_loader_mutation(weight_audit, tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"AAAA")
    expected = hashlib.sha256(b"AAAA").hexdigest()

    class MutatingTorch:
        @staticmethod
        def load(handle, **_kwargs):
            assert os.pwrite(handle.fileno(), b"BBBB", 0) == 4
            os.fsync(handle.fileno())
            return {"forged": True}

    with pytest.raises(weight_audit.AuditError, match="changed during torch.load"):
        weight_audit.load_frozen_checkpoint(run, expected, MutatingTorch)


def test_weight_run_fingerprint_binds_root_inode(weight_audit, tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "payload").write_bytes(b"same")
    before = weight_audit._run_fingerprint(root)
    old = tmp_path / "old-run"
    root.rename(old)
    root.mkdir()
    (old / "payload").rename(root / "payload")
    after = weight_audit._run_fingerprint(root)
    assert after != before


def test_weight_run_fingerprint_projects_symlink_inode_and_text_not_target(
    weight_audit, tmp_path
):
    run = tmp_path / "run"
    outside = tmp_path / "outside"
    outside_dir = outside / "directory"
    run.mkdir()
    wandb = run / "wandb"
    wandb.mkdir()
    outside_dir.mkdir(parents=True)
    outside_file = outside / "file.bin"
    outside_file.write_bytes(b"AAAA")
    (outside_dir / "payload.bin").write_bytes(b"BBBB")
    (wandb / "file-link").symlink_to("../../outside/file.bin")
    (wandb / "directory-link").symlink_to(
        "../../outside/directory", target_is_directory=True
    )
    (wandb / "external-absolute-link").symlink_to(outside_file)
    (wandb / "dangling-link").symlink_to("missing-target")
    inode_link = wandb / "same-text-new-inode"
    inode_link.symlink_to("missing-same-target")

    baseline = weight_audit._run_fingerprint(run)
    assert weight_audit._run_fingerprint(run) == baseline

    # A link is historical metadata, never an alternate content/import root.
    outside_file.write_bytes(b"CCCC")
    (outside_dir / "payload.bin").write_bytes(b"DDDD")
    outside_dir.chmod(0)
    try:
        assert weight_audit._run_fingerprint(run) == baseline
    finally:
        outside_dir.chmod(0o755)

    text_link = wandb / "dangling-link"
    text_link.unlink()
    text_link.symlink_to("different-missing-target")
    assert weight_audit._run_fingerprint(run) != baseline

    before_inode_swap = weight_audit._run_fingerprint(run)
    held = os.open(inode_link, os.O_PATH | os.O_NOFOLLOW)
    try:
        inode_link.unlink()
        inode_link.symlink_to("missing-same-target")
        assert weight_audit._run_fingerprint(run) != before_inode_swap
    finally:
        os.close(held)


def test_weight_run_fingerprint_rejects_concurrent_symlink_replacement(
    weight_audit, tmp_path, monkeypatch
):
    run = tmp_path / "run"
    wandb = run / "wandb"
    wandb.mkdir(parents=True)
    link = wandb / "debug.log"
    link.symlink_to("original-target")
    real_readlink = weight_audit.os.readlink
    attacked = False

    def replace_after_open(path, *, dir_fd=None):
        nonlocal attacked
        if path == b"" and not attacked:
            attacked = True
            link.unlink()
            link.symlink_to("replacement-target")
        return real_readlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(weight_audit.os, "readlink", replace_after_open)
    with pytest.raises(weight_audit.AuditError, match="symlink changed|directory changed"):
        weight_audit._run_fingerprint(run)
    assert attacked


def test_weight_run_fingerprint_rejects_post_read_consumed_launch_symlink(
    weight_audit, tmp_path
):
    run = tmp_path / "run"
    run.mkdir()
    launch = run / "GAUGE_PILOT_V2_LAUNCH.json"
    payload = b'{"sealed":true}\n'
    launch.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    assert weight_audit.read_json(
        launch, expected_sha256=digest, label="synthetic consumed launch"
    ) == {"sealed": True}
    identical = tmp_path / "identical-launch.json"
    identical.write_bytes(payload)
    launch.unlink()
    launch.symlink_to(identical)
    with pytest.raises(weight_audit.AuditError, match="non-W&B symlink"):
        weight_audit._run_fingerprint(run)


def test_causal_output_projection_ignores_only_sealed_submission_state(
    causal_parity_audit, tmp_path
):
    assert "_verify_output_projection_regression()" in inspect.getsource(
        causal_parity_audit.run
    )
    absent = tmp_path / "absent"
    baseline = causal_parity_audit._output_tree_fingerprint(absent)

    empty = tmp_path / "empty"
    empty.mkdir()
    assert causal_parity_audit._output_tree_fingerprint(empty) == baseline

    claimed = tmp_path / "claimed"
    snapshot = claimed / "state" / "submission" / "source-snapshot" / "repo"
    journal = claimed / "state" / "submission" / "journal"
    snapshot.mkdir(parents=True)
    journal.mkdir()
    (snapshot / "sealed.py").write_text("VALUE = 1\n", encoding="utf-8")
    (journal / "0000_CLAIMED.json").write_text("{}\n", encoding="utf-8")
    assert causal_parity_audit._output_tree_fingerprint(claimed) == baseline

    unexpected_state = tmp_path / "unexpected-state"
    (unexpected_state / "state" / "submission").mkdir(parents=True)
    (unexpected_state / "state" / "scientific-write.json").write_text(
        "{}\n", encoding="utf-8"
    )
    assert causal_parity_audit._output_tree_fingerprint(unexpected_state) != baseline

    scientific = tmp_path / "scientific"
    cell = scientific / "antmaze-large" / "GS" / "seed110"
    cell.mkdir(parents=True)
    (cell / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    assert causal_parity_audit._output_tree_fingerprint(scientific) != baseline

    hostile = tmp_path / "hostile"
    hostile_submission = hostile / "state" / "submission"
    hostile_submission.mkdir(parents=True)
    (hostile_submission / "escape").symlink_to(tmp_path / "outside")
    with pytest.raises(causal_parity_audit.ParityAuditError, match="contains symlink"):
        causal_parity_audit._output_tree_fingerprint(hostile)

    invalid_submission = tmp_path / "invalid-submission"
    (invalid_submission / "state").mkdir(parents=True)
    (invalid_submission / "state" / "submission").write_text("bad\n", encoding="utf-8")
    with pytest.raises(causal_parity_audit.ParityAuditError, match="root is not a directory"):
        causal_parity_audit._output_tree_fingerprint(invalid_submission)


def test_security_walkers_reject_unreadable_hidden_and_special_entries(
    submit, causal_parity_audit, report, cancel, weight_audit, tmp_path
):
    walkers = (
        (lambda root: submit._secure_tree_rows(root, "test tree"), submit.SubmissionError, True),
        (causal_parity_audit._secure_output_rows, causal_parity_audit.ParityAuditError, True),
        (lambda root: report._secure_tree_rows(root, "test tree", hash_files=True), report.ReportError, True),
        (lambda root: cancel._secure_tree_rows(root, "test tree"), cancel.CancellationError, True),
        (weight_audit._run_fingerprint, weight_audit.AuditError, True),
    )
    for walker_index, (walker, error, rejects_symlink) in enumerate(walkers):
        unreadable = tmp_path / f"walker-{walker_index}-unreadable"
        hidden = unreadable / ".hidden"
        hidden.mkdir(parents=True)
        (hidden / "escape").symlink_to(tmp_path / "outside")
        hidden.chmod(0)
        try:
            with pytest.raises(error):
                walker(unreadable)
        finally:
            hidden.chmod(0o700)

        symlink_root = tmp_path / f"walker-{walker_index}-symlink"
        symlink_root.mkdir()
        (symlink_root / ".hidden-link").symlink_to(tmp_path / "outside")
        if rejects_symlink:
            with pytest.raises(error):
                walker(symlink_root)
        else:
            walker(symlink_root)

        fifo_root = tmp_path / f"walker-{walker_index}-fifo"
        fifo_root.mkdir()
        os.mkfifo(fifo_root / "blocked")
        with pytest.raises(error):
            walker(fifo_root)

        socket_root = tmp_path / f"walker-{walker_index}-socket"
        socket_root.mkdir()
        endpoint = socket_root / "endpoint.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(endpoint))
            with pytest.raises(error):
                walker(socket_root)
        finally:
            server.close()

        hardlink_root = tmp_path / f"walker-{walker_index}-hardlink"
        hardlink_root.mkdir()
        outside = tmp_path / f"walker-{walker_index}-outside-file"
        outside.write_bytes(b"shared")
        os.link(outside, hardlink_root / "shared")
        with pytest.raises(error):
            walker(hardlink_root)


def test_security_existence_probes_reject_inaccessible_and_enotdir(
    submit, causal_parity_audit, report, cancel, worker, tmp_path
):
    missing = tmp_path / "genuine-enoent"
    assert not submit._lexical_exists(missing)
    assert not causal_parity_audit._lexical_exists(missing, "missing test entry")
    assert not report._lexical_exists(missing)
    assert not cancel._lexical_exists(missing)
    assert not worker.lexical_exists(missing)

    blocked = tmp_path / "blocked"
    hidden_run = blocked / "run"
    (hidden_run / "setting").mkdir(parents=True)
    (hidden_run / "setting" / "payload.bin").write_bytes(b"scientific")
    blocked.chmod(0)
    inaccessible_manifest = {
        "paths": {"run_root": str(hidden_run)},
        "design": {"settings": ["setting"]},
    }
    try:
        with pytest.raises(submit.SubmissionError, match="determine whether"):
            submit._output_tree_fingerprint(inaccessible_manifest)
        with pytest.raises(submit.SubmissionError, match="determine whether"):
            submit._scientific_output_fingerprint(inaccessible_manifest)
        with pytest.raises(submit.SubmissionError, match="determine whether"):
            submit._namespace_is_fresh(
                inaccessible_manifest, tmp_path / "absent-submission"
            )
        with pytest.raises(causal_parity_audit.ParityAuditError, match="determine whether"):
            causal_parity_audit._output_tree_fingerprint(hidden_run)
        for probe, error in (
            (report._lexical_exists, report.ReportError),
            (cancel._lexical_exists, cancel.CancellationError),
            (worker.lexical_exists, worker.LifecycleError),
        ):
            with pytest.raises(error, match="determine whether"):
                probe(hidden_run)
    finally:
        blocked.chmod(0o755)

    regular_parent = tmp_path / "regular-parent"
    regular_parent.write_text("not a directory", encoding="utf-8")
    enotdir_run = regular_parent / "run"
    enotdir_manifest = {
        "paths": {"run_root": str(enotdir_run)},
        "design": {"settings": ["setting"]},
    }
    with pytest.raises(submit.SubmissionError, match="determine whether"):
        submit._output_tree_fingerprint(enotdir_manifest)
    with pytest.raises(submit.SubmissionError, match="determine whether"):
        submit._scientific_output_fingerprint(enotdir_manifest)
    with pytest.raises(submit.SubmissionError, match="determine whether"):
        submit._namespace_is_fresh(enotdir_manifest, tmp_path / "still-absent")
    with pytest.raises(causal_parity_audit.ParityAuditError, match="determine whether"):
        causal_parity_audit._output_tree_fingerprint(enotdir_run)
    for probe, error in (
        (report._lexical_exists, report.ReportError),
        (cancel._lexical_exists, cancel.CancellationError),
        (worker.lexical_exists, worker.LifecycleError),
    ):
        with pytest.raises(error, match="determine whether"):
            probe(enotdir_run)


def test_security_type_checks_use_authenticated_lstat_and_strict_resolution(
    submit, causal_parity_audit, report, cancel, tmp_path, monkeypatch
):
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    expected_python = Path(manifest["paths"]["python"])
    expected_target = os.readlink(expected_python)
    monkeypatch.setattr(Path, "is_symlink", lambda _path: False)
    identity = submit.interpreter_contract(manifest)
    assert identity["lexical_symlink_target"] == expected_target
    for function in (
        submit.interpreter_contract,
        report.activate_isolated_runtime,
        cancel.activate_isolated_runtime,
    ):
        assert ".is_symlink(" not in inspect.getsource(function)

    fake_project = tmp_path / "project"
    fake_project.mkdir()
    (fake_project / "scripts").write_text("not a directory", encoding="utf-8")
    launch = {
        "argv": ["python", str(fake_project / "scripts" / "train.py"), "x=y"]
    }
    with pytest.raises(causal_parity_audit.ParityAuditError, match="determine whether"):
        causal_parity_audit._launch_pair_identity(launch, project_root=fake_project)
    assert ".is_symlink(" not in inspect.getsource(causal_parity_audit)


def test_content_fingerprints_catch_held_fd_same_size_mtime_mutation(
    submit, causal_parity_audit, report, cancel, weight_audit, tmp_path
):
    run_root = tmp_path / "run"
    setting = run_root / "setting"
    setting.mkdir(parents=True)
    payload = setting / "payload.bin"
    payload.write_bytes(b"AAAA")
    manifest = {
        "paths": {"run_root": str(run_root)},
        "design": {"settings": ["setting"]},
    }

    def fingerprints():
        return {
            "submit-full": submit._output_tree_fingerprint(manifest),
            "submit-scientific": submit._scientific_output_fingerprint(manifest),
            "causal": causal_parity_audit._output_tree_fingerprint(run_root),
            "report": report.stable_hash(
                report._secure_tree_rows(run_root, "test tree", hash_files=True)
            ),
            "cancel": cancel.stable_hash(cancel._secure_tree_rows(run_root, "test tree")),
            "weight": weight_audit._run_fingerprint(run_root),
        }

    before = fingerprints()
    original = payload.stat()
    descriptor = os.open(payload, os.O_WRONLY)
    try:
        assert os.pwrite(descriptor, b"BBBB", 0) == 4
        os.fsync(descriptor)
        os.utime(payload, ns=(original.st_atime_ns, original.st_mtime_ns))
        after = fingerprints()
    finally:
        os.close(descriptor)
    assert set(before) == set(after)
    assert all(after[name] != before[name] for name in before)


def test_snapshot_cleanup_descends_mode_zero_without_chmodding_symlink_target(
    submit, tmp_path
):
    root = tmp_path / "private"
    nested = root / "nested"
    nested.mkdir(parents=True)
    payload = nested / "payload"
    payload.write_bytes(b"x")
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    outside.chmod(0o640)
    (nested / "escape").symlink_to(outside)
    payload.chmod(0)
    nested.chmod(0)
    root.chmod(0)
    anomalies = submit._restore_private_tree_modes(root, "test private tree")
    assert anomalies == ["symlink:nested/escape"]
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(nested.stat().st_mode) == 0o700
    assert stat.S_IMODE(payload.stat().st_mode) == 0o600
    assert stat.S_IMODE(outside.stat().st_mode) == 0o640


def test_isolated_bootstrap_uses_fd_stable_compiled_script(submit):
    compile(submit.ISOLATED_RUN_CODE, "<isolated-run-code>", "exec")
    assert "os.walk" not in submit.ISOLATED_RUN_CODE
    assert "runpy.run_path" not in submit.ISOLATED_RUN_CODE


def test_snapshot_test_uses_production_seal_and_removes_it(
    submit, tmp_path, monkeypatch
):
    repo = tmp_path / "repository"
    package = repo / submit.PACKAGE_RELATIVE
    package.mkdir(parents=True)
    probe = repo / "probe.py"
    probe.write_text("VALUE = 1\n", encoding="utf-8")
    source_sha = "b" * 64
    manifest = {
        "campaign_id": "snapshot-test-fixture",
        "core_binding": {"trainer_code_fingerprint": source_sha},
        "design": {"settings": []},
        "paths": {
            "python": sys.executable,
            "prospective_run_root": "outputs/snapshot-test-fixture",
            "run_root": str(repo / "outputs/snapshot-test-fixture"),
        },
    }
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (package / "weight_audit.lock.json").write_text("{}\n", encoding="utf-8")
    protocol_files = {
        name: submit.file_sha256(package / name)
        for name in ("manifest.json", "weight_audit.lock.json")
    }
    protocol = submit.stable_hash({"schema_version": 1, "files": protocol_files})
    (package / "protocol.sha256").write_text(protocol + "\n", encoding="ascii")
    source_contract = {
        "source_sha256": source_sha,
        "source_files": {"probe.py": submit.file_sha256(probe)},
        "runtime_sha256": "d" * 64,
    }
    campaign = SimpleNamespace(
        PROTOCOL_FILES=("manifest.json", "weight_audit.lock.json"),
        SNAPSHOT_IMPORT_FILES={},
        read_json=lambda path: json.loads(Path(path).read_text()),
        validate_manifest=lambda *_args: None,
        verify_protocol_lock=lambda _package: protocol,
        protocol_sha256=lambda _package: protocol,
        source_contract=lambda _root: dict(source_contract),
    )
    calls = []
    sealed_roots = []
    real_inventory = submit.snapshot_inventory
    real_create = submit.create_source_snapshot

    def inventory(*args, **kwargs):
        calls.append("inventory")
        return real_inventory(*args, **kwargs)

    def create(*args):
        calls.append("create")
        result = real_create(*args)
        sealed_roots.append(result)
        return result

    def copied_preflight(snapshot_root, audit_input_root, claimed, exact, *, runner):
        calls.append("preflight")
        assert audit_input_root == repo
        assert claimed == protocol
        submit.verify_snapshot_files(snapshot_root, exact)
        assert not snapshot_root.stat().st_mode & 0o222
        return {
            "manifest": manifest,
            "launches": [{} for _ in range(20)],
            "verification": {
                "audit_replays": {
                    name: {"artifact_sha256": str(index) * 64}
                    for index, name in enumerate(
                        ("weight", "prefix_target", "resolved_config", "causal_parity"),
                        start=1,
                    )
                },
                "import_containment": "all_treewm_modules_inside_snapshot",
                "audit_input_root": str(repo),
            },
        }

    monkeypatch.setattr(submit, "reject_inherited_environment", lambda: None)
    monkeypatch.setattr(submit, "activate_isolated_runtime", lambda _manifest: {"test": True})
    monkeypatch.setattr(submit, "load_campaign", lambda _root: campaign)
    monkeypatch.setattr(submit, "snapshot_inventory", inventory)
    monkeypatch.setattr(submit, "create_source_snapshot", create)
    monkeypatch.setattr(submit, "_snapshot_preflight", copied_preflight)
    for name in list(sys.modules):
        if name == "treewm" or name.startswith("treewm."):
            monkeypatch.delitem(sys.modules, name)

    result = submit.snapshot_test(repo, runner=lambda *_args: None)
    assert result["status"] == "snapshot_test_verified"
    assert result["temporary_tree_removed"] is True
    assert result["persistent_writes_performed"] == 0
    assert result["scheduler_calls"] == 0
    assert calls == ["inventory", "create", "preflight"]
    assert sealed_roots and all(not path.exists() for path in sealed_roots)


def test_snapshot_test_cli_never_enters_submit_path(
    submit, tmp_path, monkeypatch, capsys
):
    repo = tmp_path / "repository"
    package = repo / submit.PACKAGE_RELATIVE
    package.mkdir(parents=True)
    (package / "manifest.json").write_text("{}\n", encoding="utf-8")
    called = []

    def verified(root):
        called.append(root)
        return {"status": "snapshot_test_verified", "scheduler_calls": 0}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("snapshot test entered submission/scheduler preflight")

    monkeypatch.setattr(submit, "snapshot_test", verified)
    monkeypatch.setattr(submit, "activate_isolated_runtime", forbidden)
    monkeypatch.setattr(submit, "static_preflight", forbidden)
    monkeypatch.setattr(submit, "submit_campaign", forbidden)
    monkeypatch.setattr(submit, "recover_transaction", forbidden)
    assert submit.main(["--snapshot-test", "--repo-root", str(repo)]) == 0
    assert called == [repo]
    assert json.loads(capsys.readouterr().out)["scheduler_calls"] == 0


def test_exclusive_json_never_exposes_partial_final_path(submit, tmp_path, monkeypatch):
    real_link = submit.os.link

    def fail_link(*args, **kwargs):
        raise OSError("injected publication failure")

    monkeypatch.setattr(submit.os, "link", fail_link)
    target = tmp_path / "SEALED.json"
    with pytest.raises(OSError):
        submit.exclusive_json(target, {"value": 1})
    assert not target.exists()
    assert not list(tmp_path.glob(".SEALED.json.tmp.*"))
    monkeypatch.setattr(submit.os, "link", real_link)


def test_git_commands_use_pinned_binary_and_read_only_sanitized_env(submit, monkeypatch):
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout="value\n", stderr="")

    monkeypatch.setattr(submit.subprocess, "run", run)
    assert submit._git_command(REPO, "rev-parse", "HEAD") == "value"
    assert captured["command"][0] == "/usr/bin/git"
    assert captured["env"] == {
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "LC_ALL": "C",
    }


def test_nonzero_sbatch_preserves_parseable_exact_id(submit, tmp_path):
    calls = 0

    def runner(command, cwd, environment):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(command, 1, stdout="4312\n", stderr="lost response")
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="squeue unavailable")

    with pytest.raises(submit.SchedulerSubmissionError) as caught:
        submit._submit_one(
            ["sbatch"], job_name="exact", comment="token", squeue="squeue",
            cwd=tmp_path, runner=runner,
        )
    assert caught.value.job_ids == ("4312",)


def test_federated_sbatch_suffix_is_rejected(submit, tmp_path):
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="4312;foreign\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ]
    )
    with pytest.raises(submit.SchedulerSubmissionError):
        submit._submit_one(
            ["sbatch"], job_name="exact", comment="token", squeue="squeue",
            cwd=tmp_path, runner=lambda *_args: next(responses),
        )


def test_paused_submitter_excludes_recovery_until_owner_exits(submit, tmp_path):
    submission = tmp_path / "outputs" / "campaign" / "state" / "submission"
    submission.parents[2].mkdir()
    owner_ready = threading.Event()
    release_owner = threading.Event()
    errors = []

    def owner():
        try:
            with submit._TransactionLock(submission):
                owner_ready.set()
                assert release_owner.wait(5)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=owner)
    thread.start()
    assert owner_ready.wait(5)
    with pytest.raises(submit.SubmissionError, match="transaction is active"):
        with submit._TransactionLock(submission):
            pass
    release_owner.set()
    thread.join(5)
    assert not thread.is_alive() and not errors
    # A dead/exited owner releases the kernel lock and makes crash recovery eligible.
    with submit._TransactionLock(submission):
        pass


def test_successful_abort_cancellation_is_idempotent_recovery_evidence(submit):
    ids = ["100", "101"]
    aborted = {
        "cancellation": {
            "job_ids": ids,
            "command": ["/usr/bin/scancel", *ids],
            "stdout": "",
            "stderr": "",
        },
        "cancellation_error": None,
    }
    assert submit._validated_successful_cancellation(
        aborted, "/usr/bin/scancel", ids
    ) == set(ids)
    aborted["cancellation"]["command"] = ["scancel", *ids]
    with pytest.raises(submit.SubmissionError, match="command differs"):
        submit._validated_successful_cancellation(aborted, "/usr/bin/scancel", ids)


class GateStub:
    GAUGE_EXACT_TAGS = ("gauge",)
    METHOD_EXACT_TAGS = ("method", "data/validation_fixed_sample_count")
    GRADIENT_NORM_TAGS = ("grad",)
    GRADIENT_CLIP_TAGS = ("clip",)
    TRAIN_PREFIX = "train/p/"
    PREFIX = "val/p/"
    PREFIX_COMMON_SUFFIXES = ("one",)


def exact_scalars():
    train = tuple(range(50, 25_001, 50))
    validation = tuple(range(1000, 25_001, 1000))
    return {
        **{tag: {step: 1.0 for step in train} for tag in ("gauge", "grad", "clip", "train/p/one")},
        **{tag: {step: 1.0 for step in validation} for tag in ("method", "val/p/one")},
        "data/validation_fixed_sample_count": {step: 5120.0 for step in (0, *validation)},
    }


def test_reporter_requires_exact_full_axes(report):
    manifest = {"scientific_contract": {"training_telemetry_every_updates": 50, "validation_every_updates": 1000}}
    scalars = exact_scalars()
    report.validate_boundary_axes(scalars, GateStub, manifest)
    scalars["grad"][51] = 1.0
    with pytest.raises(report.ReportError, match="axis differs"):
        report.validate_boundary_axes(scalars, GateStub, manifest)


def test_event_parser_excludes_periodic_terminal_eval_collision(report, tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    sampler = {"global_sample_size": 5120, "seed": 1701}
    text = "<pre>" + json.dumps(sampler, sort_keys=True, indent=2) + "</pre>"
    writer = SummaryWriter(str(tmp_path), filename_suffix=".generation")
    writer.add_text("meta/fixed_validation_sample", text, 0)
    writer.add_scalar("train/loss_total", 2.0, 50)
    writer.add_scalar("eval/success_rate", 0.2, 25_000)
    writer.add_scalar("eval/success_rate", 0.8, 25_000)
    writer.flush()
    writer.close()
    parsed = report.parse_event_files(tmp_path, sampler)
    assert "eval/success_rate" not in parsed["scalars"]
    assert parsed["excluded_eval_tags"] == ["eval/success_rate"]
    assert parsed["scalars"]["train/loss_total"] == {50: 2.0}


def _write_test_event(directory: Path, sampler: dict, scalar: float) -> Path:
    from torch.utils.tensorboard import SummaryWriter

    text = "<pre>" + json.dumps(sampler, sort_keys=True, indent=2) + "</pre>"
    writer = SummaryWriter(str(directory))
    writer.add_text("meta/fixed_validation_sample", text, 0)
    writer.add_scalar("train/loss_total", scalar, 50)
    writer.flush()
    writer.close()
    return next(directory.glob("events.out.tfevents.*"))


def test_event_parser_is_bound_to_anonymous_private_fd(
    report, tmp_path, monkeypatch
):
    from tensorboard.backend.event_processing import event_accumulator as accumulator_module

    sampler = {"global_sample_size": 5120, "seed": 1701}
    live = tmp_path / "live"
    forged = tmp_path / "forged"
    live.mkdir()
    forged.mkdir()
    _write_test_event(live, sampler, 1.0)
    forged_bytes = _write_test_event(forged, sampler, 9.0).read_bytes()
    real_accumulator = accumulator_module.EventAccumulator

    class PathSwapAccumulator(real_accumulator):
        def Reload(self):
            # Recreate the predictable private pathname after the authenticated
            # inode has been unlinked. The parser must remain pinned to its fd.
            (Path(self.path) / "event-000.tfevents").write_bytes(forged_bytes)
            return super().Reload()

    monkeypatch.setattr(accumulator_module, "EventAccumulator", PathSwapAccumulator)
    parsed = report.parse_event_files(live, sampler)
    assert parsed["scalars"]["train/loss_total"] == {50: 1.0}


@pytest.mark.parametrize("corrupt_hparams", [False, True])
def test_event_parser_rejects_trailing_corrupt_tfrecord(
    report, tmp_path, corrupt_hparams
):
    sampler = {"global_sample_size": 5120, "seed": 1701}
    live = tmp_path / "live"
    live.mkdir()
    training = _write_test_event(live, sampler, 1.0)
    target = training
    if corrupt_hparams:
        from torch.utils.tensorboard import SummaryWriter

        hparams = live / "hparams"
        writer = SummaryWriter(str(hparams))
        writer.add_scalar("hparams/metric", 1.0, 0)
        writer.close()
        target = next(hparams.glob("events.out.tfevents.*"))
    with target.open("ab") as handle:
        handle.write(b"CORRUPT_TRAILING_RECORD")
    with pytest.raises(report.ReportError, match="TFRecord"):
        report.parse_event_files(live, sampler)


def test_event_parser_rejects_symlink_hparams(report, tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    sampler = {"x": 1}
    writer = SummaryWriter(str(tmp_path))
    writer.add_text(
        "meta/fixed_validation_sample",
        "<pre>" + json.dumps(sampler, sort_keys=True, indent=2) + "</pre>",
        0,
    )
    writer.close()
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "hparams").symlink_to(outside, target_is_directory=True)
    with pytest.raises(report.ReportError, match="hparams"):
        report.parse_event_files(tmp_path, sampler)


def test_report_publication_is_atomic_and_idempotent(report, tmp_path):
    bundle = {"schema_version": 1, "cells": []}
    decision = {"status": "rejected", "gate_sha256": "a" * 64}
    provenance = {"schema_version": 1}
    first = report.publish_report(tmp_path, "b" * 64, bundle, decision, provenance)
    second = report.publish_report(tmp_path, "b" * 64, bundle, decision, provenance)
    assert first == second
    assert (tmp_path / "report" / "REPORT_COMMIT.json").is_file()
    assert not list(tmp_path.glob(".report.tmp.*"))


def test_report_failure_before_rename_publishes_nothing(report, tmp_path, monkeypatch):
    real = report.seal_json
    calls = 0

    def fail_second(path, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected staging failure")
        return real(path, value)

    monkeypatch.setattr(report, "seal_json", fail_second)
    with pytest.raises(OSError):
        report.publish_report(
            tmp_path,
            "b" * 64,
            {"schema_version": 1},
            {"status": "rejected", "gate_sha256": "a" * 64},
            {"schema_version": 1},
        )
    assert not (tmp_path / "report").exists()
    assert not list(tmp_path.glob(".report.tmp.*"))


def test_report_rejects_broken_cancellation_latch(report, tmp_path):
    snapshot = tmp_path / "snapshot"
    submission = tmp_path / "submission"
    snapshot.mkdir()
    submission.mkdir()
    (submission / "CANCEL_REQUESTED.json").symlink_to(tmp_path / "missing-latch-target")
    with pytest.raises(report.ReportError, match="cancelled/ambiguous"):
        report.assemble_report(snapshot, submission, "a" * 64)


def test_report_commit_and_cancel_latch_are_linearly_ordered(
    report, cancel, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    submission.mkdir()
    report_entered = threading.Event()
    allow_report_commit = threading.Event()
    cancel_acquired = threading.Event()
    failures = []

    def fake_publish(root, *_args):
        report_entered.set()
        assert allow_report_commit.wait(5)
        assert not os.path.lexists(root / "CANCEL_REQUESTED.json")
        report.seal_json(root / "REPORT_COMMIT.test.json", {"status": "committed"})
        return {"status": "committed"}

    monkeypatch.setattr(report, "_publish_report_locked", fake_publish)

    def report_thread():
        try:
            report.publish_report(submission, "a" * 64, {}, {}, {})
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    def cancel_thread():
        try:
            with cancel._ReportCancelLock(submission):
                cancel_acquired.set()
                assert (submission / "REPORT_COMMIT.test.json").is_file()
                cancel.seal_json(
                    submission / "CANCEL_REQUESTED.json", {"status": "cancel_requested"}
                )
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    reporter = threading.Thread(target=report_thread)
    canceller = threading.Thread(target=cancel_thread)
    reporter.start()
    assert report_entered.wait(5)
    canceller.start()
    assert not cancel_acquired.wait(0.1)
    allow_report_commit.set()
    reporter.join(5)
    canceller.join(5)
    assert not reporter.is_alive() and not canceller.is_alive() and not failures
    assert (submission / "REPORT_COMMIT.test.json").is_file()
    assert (submission / "CANCEL_REQUESTED.json").is_file()


def test_cancel_latch_precedes_scheduler_and_result_is_durable(
    submit, cancel, tmp_path, monkeypatch
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    scancel = tmp_path / "scancel"
    scancel.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    scancel.chmod(0o755)
    receipt = {
        "campaign_id": "treewm-executable-prefix-repair-pilot-v1-launch3",
        "submission_sha256": "c" * 64,
        "train_array_job_id": "100",
        "report_job_id": "101",
        "snapshot_root": str(snapshot),
    }
    manifest = {"execution": {"scancel": str(scancel)}}
    monkeypatch.setattr(cancel, "validate_receipt", lambda _root: (receipt, {}, manifest))
    monkeypatch.setenv("SLURM_CONF", "/tmp/hostile-slurm.conf")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/hostile-libraries")

    def run(command, **kwargs):
        assert (tmp_path / "CANCEL_REQUESTED.json").is_file()
        assert list((tmp_path / "cancellation").glob("CANCEL_CALL.*.json"))
        assert kwargs["env"] == {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
        }
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(cancel.subprocess, "run", run)
    result = cancel.explicit_cancel(tmp_path)
    assert result["job_ids"] == ["100", "101"]
    result_files = list((tmp_path / "cancellation").glob("CANCEL_RESULT.*.json"))
    assert len(result_files) == 1
    sealed = json.loads(result_files[0].read_text())
    assert sealed["returncode"] == 0 and sealed["command"][-2:] == ["100", "101"]
    assert cancel.scheduler_environment() == submit._scheduler_environment()
