"""Focused fail-closed tests for Exp23 submission, cancellation, and reporting."""

from __future__ import annotations

import ast
import copy
import fcntl
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


def scheduler_contract() -> dict:
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    return manifest["execution"]["scheduler_control_plane"]


def scheduler_config_bytes() -> bytes:
    return (
        "ClusterName=cs-oci-ord\n"
        "SlurmctldHost=cs-oci-ord-a\n"
        "SlurmctldHost=cs-oci-ord-b\n"
        "SlurmctldPort=6817\n"
        "AuthType=auth/munge\n"
        "GresTypes=gpu\n"
        "CliFilterPlugins=lua\n"
        "JobSubmitPlugins=lua\n"
        "CommunicationParameters=NoAddrCache\n"
    ).encode("utf-8")


def scheduler_observation(submit) -> dict:
    payload = scheduler_config_bytes()
    return {
        "schema_version": 1,
        "trust_model": submit.SCHEDULER_TRUST_MODEL,
        "config": {
            "path": scheduler_contract()["slurm_conf"],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "identity": {"size": len(payload)},
        },
        "critical": {
            key: scheduler_contract()[key]
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
        "cli_filter_policy": {"files": {}, "tree_sha256": "a" * 64},
    }


def scheduler_fallback(submit) -> dict:
    payload = scheduler_config_bytes()
    return {
        "schema_version": 1,
        "purpose": (
            "accepted-job exact reconciliation, cancellation, dependency verification, "
            "and wave-zero release only; never submission or compute-side execution"
        ),
        "encoding": "base64",
        "payload_base64": submit.base64.b64encode(payload).decode("ascii"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "source_control_plane": scheduler_observation(submit),
    }


def trainer_smoke_record(
    submit,
    *,
    inventory: dict[str, str],
    launch_sha256: str,
    resolved_config_sha256: str,
    full_output_fingerprint: str,
    scientific_output_fingerprint: str,
) -> dict:
    return {
        "schema_version": 1,
        "status": "sealed_trainer_hydra_composition_verified",
        "cell_index": 0,
        "python_flags": ["-P", "-S", "-B"],
        "entry_relative_path": str(submit.PACKAGE_RELATIVE / "train_entry.py"),
        "config_package_relative_path": "configs/__init__.py",
        "config_package_sha256": hashlib.sha256(b"").hexdigest(),
        "snapshot_inventory_sha256": submit.stable_hash(inventory),
        "launch_sha256": launch_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "stdout_sha256": hashlib.sha256(b"config").hexdigest(),
        "stdout_bytes": 6,
        "cuda_visible_devices": "",
        "full_output_fingerprint_before": full_output_fingerprint,
        "full_output_fingerprint_after": full_output_fingerprint,
        "scientific_output_fingerprint_before": scientific_output_fingerprint,
        "scientific_output_fingerprint_after": scientific_output_fingerprint,
        "persistent_writes_performed": 0,
        "scheduler_calls": 0,
    }


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


def test_launch8_transaction_lock_path_is_exact(submit):
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    run_root = Path(manifest["paths"]["run_root"])
    submission_root = run_root / "state" / "submission"
    expected = run_root.parents[1] / manifest["paths"]["transaction_lock"]
    assert submit._transaction_lock_path(submission_root) == expected
    assert expected.name == ".exp23-c85fcaba919d617f.transaction.lock"


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


def test_late_report_parity_and_accepted_dependency_precede_ready(submit):
    source = inspect.getsource(submit._submit_campaign_impl)
    late = source.index("report_test_completed, report_test_observation")
    zero_query = source.index("_assert_job_absent(", late)
    zero_job = source.index("zero_job_after_test", zero_query)
    real_report = source.index("report_id, report_record = _submit_one")
    accepted = source.index("report_dependency_evidence = _accepted_dependency_evidence")
    journal = source.index('"REPORT_SUBMITTED"')
    ready = source.index('"READY_TO_COMMIT"')
    cancellation = source.index(
        "abort_evidence = _initial_exception_reconcile_and_cancel", ready
    )
    assert late < zero_query < zero_job < real_report < accepted < journal < ready < cancellation


def test_two_wave_transaction_killpoint_partition_is_fail_closed(submit):
    source = inspect.getsource(submit._submit_campaign_impl)
    wave0 = source.index("wave0_id, wave0_record = _submit_one")
    hold = source.index("wave0_hold = _accepted_wave0_hold_evidence")
    wave1 = source.index("wave1_id, wave1_record = _submit_one")
    wave1_dependency = source.index(
        "wave1_dependency_evidence = _accepted_dependency_evidence"
    )
    report = source.index("report_id, report_record = _submit_one")
    report_dependency = source.index(
        "report_dependency_evidence = _accepted_dependency_evidence"
    )
    authorization = source.index('"SUBMISSION_AUTHORIZATION.json"')
    authorization_journal = source.index('"DAG_AUTHORIZED"')
    ready_journal = source.index('"READY_TO_COMMIT"')
    ready_latch = source.index("ready_to_commit = True")
    receipt = source.index('"SUBMISSION_RECEIPT.json"')
    release = source.index("release_evidence = _release_authorized_wave0")
    release_journal = source.index('"WAVE0_RELEASED"')
    exception = source.index("except BaseException as exc:")
    recovery_required = source.index("if ready_to_commit:", exception)
    cancel_context = source.index("cancel_context = {", recovery_required)
    reconcile = source.index(
        "abort_evidence = _initial_exception_reconcile_and_cancel", cancel_context
    )
    assert (
        wave0
        < hold
        < wave1
        < wave1_dependency
        < report
        < report_dependency
        < authorization
        < authorization_journal
        < ready_journal
        < ready_latch
        < receipt
        < release
        < release_journal
        < exception
    )
    # Every cut before READY enters exact three-role reconciliation/cancellation;
    # every cut from READY onward raises for recovery before either can execute.
    assert recovery_required < cancel_context < reconcile
    helper = inspect.getsource(submit._initial_exception_reconcile_and_cancel)
    assert 'roles = ("wave0", "wave1", "report")' in helper
    assert "_append_recovery_cancel_attempt(" in helper


def test_recovery_orders_receipt_reconstruction_before_idempotent_release(submit):
    source = inspect.getsource(submit._recover_transaction_locked)
    committed = source.index(
        "if committed_fast_path_allowed and _lexical_exists(receipt_path):"
    )
    ready = source.index(
        "if committed_fast_path_allowed and _lexical_exists(ready_path):"
    )
    validate_authorization = source.index(
        "_validated_dag_authorization(submission_root, submission_sha256, receipt)",
        ready,
    )
    seal_receipt = source.index("exclusive_json(receipt_path, receipt)", ready)
    finish = source.index("return finish_committed_receipt(", seal_receipt)
    abort_recovery = source.index("prior_recovery_path", ready)
    assert committed < ready < validate_authorization < seal_receipt < finish < abort_recovery


def _release_evidence(submit, *, reason="None", job_id="7000", job_name="wave0"):
    observation = scheduler_observation(submit)
    stdout = (
        f"JobId={job_id} JobName={job_name} JobState=PENDING "
        f"Reason={reason} Comment=treewm-exp23:{'c' * 64}\n"
    )
    return {
        "release_command": ["scontrol", "release", job_id],
        "release_returncode": 0,
        "release_stdout": "",
        "release_stderr": "",
        "show_command": ["scontrol", "show", "job", job_id, "--oneliner"],
        "show_returncode": 0,
        "show_stdout": stdout,
        "show_stderr": "",
        "state": "PENDING",
        "reason": reason,
        "release_scheduler_control_plane": observation,
        "show_scheduler_control_plane": observation,
    }


def test_durable_wave0_release_evidence_is_exact_and_supports_crash_after_release(submit):
    expected = scheduler_observation(submit)
    full = _release_evidence(submit)
    assert submit._validated_wave0_release_evidence(
        full,
        scontrol="scontrol",
        job_id="7000",
        job_name="wave0",
        comment="treewm-exp23:" + "c" * 64,
        expected_observation=expected,
    ) == full
    already_released = {
        **full,
        "release_command": None,
        "release_returncode": None,
        "release_scheduler_control_plane": None,
    }
    assert submit._validated_wave0_release_evidence(
        already_released,
        scontrol="scontrol",
        job_id="7000",
        job_name="wave0",
        comment="treewm-exp23:" + "c" * 64,
        expected_observation=expected,
    ) == already_released


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ("held", "identity/state"),
        ("wrong-job", "identity/state"),
        ("wrong-command", "command/result"),
        ("wrong-control", "command/result"),
        ("extra", "fields"),
    ],
)
def test_durable_wave0_release_evidence_rejects_forgery(submit, mutation, pattern):
    expected = scheduler_observation(submit)
    evidence = _release_evidence(submit)
    if mutation == "held":
        evidence = _release_evidence(submit, reason="JobHeldUser")
    elif mutation == "wrong-job":
        evidence["show_stdout"] = evidence["show_stdout"].replace(
            "JobId=7000", "JobId=9999"
        )
    elif mutation == "wrong-command":
        evidence["release_command"] = ["scontrol", "release", "9999"]
    elif mutation == "wrong-control":
        evidence["release_scheduler_control_plane"] = {"forged": True}
    else:
        evidence["unexpected"] = True
    with pytest.raises(submit.SubmissionError, match=pattern):
        submit._validated_wave0_release_evidence(
            evidence,
            scontrol="scontrol",
            job_id="7000",
            job_name="wave0",
            comment="treewm-exp23:" + "c" * 64,
            expected_observation=expected,
        )


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


def _trainer_smoke_unit_inputs(submit, tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    package = snapshot / submit.PACKAGE_RELATIVE
    package.mkdir(parents=True)
    (snapshot / "scripts").mkdir()
    (package / "train_entry.py").write_text("# entry\n", encoding="utf-8")
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "design": {"settings": []},
                "paths": {
                    "run_root": str(tmp_path / "prospective-output"),
                    "python": "/pinned/python",
                },
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "scripts/train.py").write_text("# trainer\n", encoding="utf-8")
    config = {"seed": 110, "objective_version": "smoke"}
    launch_body = {
        "schema_version": 1,
        "campaign_id": "treewm-executable-prefix-repair-pilot-v1-launch8",
        "cell": {"index": 0, "seed": 110},
        "argv": [
            "/pinned/python",
            str(snapshot / "scripts/train.py"),
            "resume=auto",
            "train.steps=25000",
        ],
        "environment": {
            "MUJOCO_GL": "egl",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        },
    }
    launch = {**launch_body, "launch_sha256": submit.stable_hash(launch_body)}
    expected = {
        "resolved_config": config,
        "resolved_config_sha256": submit.stable_hash(config),
    }
    inventory = {"probe": "a" * 64}
    return snapshot, inventory, launch, expected, config


def test_trainer_bootstrap_smoke_uses_exact_entry_flags_and_hides_cuda(
    submit, tmp_path, monkeypatch
):
    snapshot, inventory, launch, expected, config = _trainer_smoke_unit_inputs(
        submit, tmp_path
    )
    monkeypatch.setattr(submit, "verify_snapshot_files", lambda *_args: None)
    monkeypatch.setattr(
        submit,
        "interpreter_contract",
        lambda _manifest: {"lexical_executable": "/pinned/python"},
    )
    observed = {}

    def runner(command, cwd, environment, timeout):
        observed.update(
            command=list(command), cwd=cwd, environment=dict(environment), timeout=timeout
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=("seed: 110\nobjective_version: smoke\n").encode(),
            stderr=b"",
        )

    record = submit.trainer_bootstrap_smoke(
        snapshot, inventory, launch, expected, runner=runner
    )
    assert observed["command"][:5] == [
        "/pinned/python",
        "-P",
        "-S",
        "-B",
        str(snapshot / submit.PACKAGE_RELATIVE / "train_entry.py"),
    ]
    assert "--_hydra-composition-smoke" in observed["command"]
    assert str(snapshot / "scripts/train.py") not in observed["command"][:5]
    assert observed["cwd"] == snapshot
    assert observed["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert observed["environment"]["PYTHONHASHSEED"] == "110"
    assert record["python_flags"] == ["-P", "-S", "-B"]
    assert record["resolved_config_sha256"] == submit.stable_hash(config)
    assert record["full_output_fingerprint_before"] == record[
        "full_output_fingerprint_after"
    ]


def test_trainer_bootstrap_smoke_rejects_resolved_config_mismatch(
    submit, tmp_path, monkeypatch
):
    snapshot, inventory, launch, expected, _config = _trainer_smoke_unit_inputs(
        submit, tmp_path
    )
    monkeypatch.setattr(submit, "verify_snapshot_files", lambda *_args: None)
    monkeypatch.setattr(
        submit,
        "interpreter_contract",
        lambda _manifest: {"lexical_executable": "/pinned/python"},
    )

    def runner(command, *_args):
        return subprocess.CompletedProcess(
            command, 0, stdout=b"seed: 111\nobjective_version: smoke\n", stderr=b""
        )

    with pytest.raises(submit.SubmissionError, match="smoke config differs"):
        submit.trainer_bootstrap_smoke(
            snapshot, inventory, launch, expected, runner=runner
        )


def test_trainer_bootstrap_smoke_rejects_output_mutation(
    submit, tmp_path, monkeypatch
):
    snapshot, inventory, launch, expected, _config = _trainer_smoke_unit_inputs(
        submit, tmp_path
    )
    monkeypatch.setattr(submit, "verify_snapshot_files", lambda *_args: None)
    monkeypatch.setattr(
        submit,
        "interpreter_contract",
        lambda _manifest: {"lexical_executable": "/pinned/python"},
    )

    def runner(command, *_args):
        output = tmp_path / "prospective-output"
        output.mkdir()
        (output / "unexpected").write_text("mutation", encoding="utf-8")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b"seed: 110\nobjective_version: smoke\n",
            stderr=b"",
        )

    with pytest.raises(submit.SubmissionError, match="changed the prospective output tree"):
        submit.trainer_bootstrap_smoke(
            snapshot, inventory, launch, expected, runner=runner
        )


def _real_trainer_smoke_case(
    submit, tmp_path: Path, *, config_marker: bytes | None
):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from treewm.utils.provenance import trainer_code_fingerprint

    case_root = tmp_path
    source = case_root / "source"
    package = source / submit.PACKAGE_RELATIVE
    prospective_output = case_root / "prospective-output"
    manifest = {
        "design": {"settings": []},
        "paths": {
            "python": str(
                json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))[
                    "paths"
                ]["python"]
            ),
            "run_root": str(prospective_output),
        },
    }
    source_files = set(trainer_code_fingerprint(REPO)["files"])
    source_files.update(
        {
            str(submit.PACKAGE_RELATIVE / "train_entry.py"),
            str(submit.PACKAGE_RELATIVE / "worker.py"),
            str(submit.PACKAGE_RELATIVE / "train.slurm"),
        }
    )
    for relative in sorted(source_files):
        if relative == "configs/__init__.py" and config_marker is None:
            continue
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = (REPO / relative).read_bytes()
        if relative == "configs/__init__.py" and config_marker is not None:
            payload = config_marker
        destination.write_bytes(payload)
    package.mkdir(parents=True, exist_ok=True)
    (package / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    inventory = {
        str(path.relative_to(source)): submit.file_sha256(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    snapshot = case_root / "source-snapshot/repo"
    submit.create_source_snapshot(source, snapshot, inventory)

    trainer_args = [
        "experiment=treewm_v2_grounded_executable_prefix_pilot_v1",
        "seed=110",
        "device=cpu",
        "resume=auto",
        "train.steps=25000",
        f"hydra.run.dir={prospective_output / 'hydra'}",
        "hydra.job.chdir=false",
    ]
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO / "configs")
    ):
        expected_config = OmegaConf.to_container(
            compose(config_name="base", overrides=trainer_args), resolve=True
        )
    launch_body = {
        "schema_version": 1,
        "campaign_id": "treewm-executable-prefix-repair-pilot-v1-launch8",
        "cell": {"index": 0, "seed": 110},
        "argv": [
            manifest["paths"]["python"],
            str(snapshot / "scripts/train.py"),
            *trainer_args,
        ],
        "environment": {
            "MUJOCO_GL": "egl",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        },
    }
    launch = {**launch_body, "launch_sha256": submit.stable_hash(launch_body)}
    expected = {
        "resolved_config": expected_config,
        "resolved_config_sha256": submit.stable_hash(expected_config),
    }
    return case_root, snapshot, inventory, launch, expected, prospective_output


def test_real_sealed_trainer_bootstrap_smoke_resolves_exact_hydra_config(
    submit, tmp_path
):
    case_root, snapshot, inventory, launch, expected, prospective_output = (
        _real_trainer_smoke_case(submit, tmp_path, config_marker=b"")
    )
    try:
        record = submit.trainer_bootstrap_smoke(
            snapshot, inventory, launch, expected
        )
        assert record["status"] == "sealed_trainer_hydra_composition_verified"
        assert record["resolved_config_sha256"] == expected[
            "resolved_config_sha256"
        ]
        assert record["config_package_sha256"] == hashlib.sha256(b"").hexdigest()
        assert record["python_flags"] == ["-P", "-S", "-B"]
        assert record["cuda_visible_devices"] == ""
        assert record["full_output_fingerprint_before"] == record[
            "full_output_fingerprint_after"
        ]
        assert record["scientific_output_fingerprint_before"] == record[
            "scientific_output_fingerprint_after"
        ]
        assert not prospective_output.exists()
    finally:
        submit._restore_private_tree_modes(case_root, "real trainer smoke case")


@pytest.mark.parametrize("config_marker", [None, b"# replacement marker\n"])
def test_real_sealed_trainer_bootstrap_smoke_rejects_config_marker_drift(
    submit, tmp_path, config_marker
):
    case_root, snapshot, inventory, launch, expected, prospective_output = (
        _real_trainer_smoke_case(
            submit, tmp_path, config_marker=config_marker
        )
    )
    try:
        with pytest.raises(
            submit.SubmissionError, match="omits/replaces exact import marker"
        ):
            submit.trainer_bootstrap_smoke(
                snapshot, inventory, launch, expected
            )
        assert not prospective_output.exists()
    finally:
        submit._restore_private_tree_modes(case_root, "adversarial trainer smoke case")


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
stable = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
launches = [{"launch_sha256": hashlib.sha256(f"launch-{index}".encode()).hexdigest()} for index in range(20)]
compositions = [
    {
        "index": index,
        "launch_sha256": launches[index]["launch_sha256"],
        "resolved_config_sha256": hashlib.sha256(f"config-{index}".encode()).hexdigest(),
        "stdout_sha256": hashlib.sha256(f"stdout-{index}".encode()).hexdigest(),
    }
    for index in range(20)
]
full_output = "5" * 64
scientific_output = "6" * 64
smoke = {
    "schema_version": 1,
    "status": "sealed_trainer_hydra_composition_verified",
    "cell_index": 0,
    "python_flags": ["-P", "-S", "-B"],
    "entry_relative_path": "experiments/23-treewm-executable-prefix-repair-pilot-v1/train_entry.py",
    "config_package_relative_path": "configs/__init__.py",
    "config_package_sha256": hashlib.sha256(b"").hexdigest(),
    "snapshot_inventory_sha256": stable(inventory),
    "launch_sha256": launches[0]["launch_sha256"],
    "resolved_config_sha256": compositions[0]["resolved_config_sha256"],
    "stdout_sha256": hashlib.sha256(b"config").hexdigest(),
    "stdout_bytes": 6,
    "cuda_visible_devices": "",
    "full_output_fingerprint_before": full_output,
    "full_output_fingerprint_after": full_output,
    "scientific_output_fingerprint_before": scientific_output,
    "scientific_output_fingerprint_after": scientific_output,
    "persistent_writes_performed": 0,
    "scheduler_calls": 0,
}
value = {
    "manifest": manifest,
    "launches": launches,
    "compositions": compositions,
    "cache_root": str(cache_root),
    "verification": {
        "audit_replays": {},
        "full_output_fingerprint_before": full_output,
        "full_output_fingerprint_after": full_output,
        "scientific_output_fingerprint_before": scientific_output,
        "scientific_output_fingerprint_after": scientific_output,
        "import_containment": "all_treewm_modules_inside_snapshot",
        "audit_input_root": str(inputs),
        "trainer_bootstrap_smoke": smoke,
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
        launches = [
            {"launch_sha256": hashlib.sha256(f"launch-{index}".encode()).hexdigest()}
            for index in range(20)
        ]
        compositions = [
            {
                "index": index,
                "launch_sha256": launches[index]["launch_sha256"],
                "resolved_config_sha256": hashlib.sha256(
                    f"config-{index}".encode()
                ).hexdigest(),
                "stdout_sha256": hashlib.sha256(
                    f"stdout-{index}".encode()
                ).hexdigest(),
            }
            for index in range(20)
        ]
        full_output = submit._output_tree_fingerprint(manifest)
        scientific_output = submit._scientific_output_fingerprint(manifest)
        return {
            "manifest": manifest,
            "launches": launches,
            "compositions": compositions,
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
                "full_output_fingerprint_before": full_output,
                "full_output_fingerprint_after": full_output,
                "scientific_output_fingerprint_before": scientific_output,
                "scientific_output_fingerprint_after": scientific_output,
                "trainer_bootstrap_smoke": trainer_smoke_record(
                    submit,
                    inventory=exact,
                    launch_sha256=launches[0]["launch_sha256"],
                    resolved_config_sha256=compositions[0][
                        "resolved_config_sha256"
                    ],
                    full_output_fingerprint=full_output,
                    scientific_output_fingerprint=scientific_output,
                ),
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


def test_nonzero_sbatch_preserves_parseable_exact_id(submit, tmp_path, monkeypatch):
    calls = 0

    monkeypatch.setattr(
        submit, "_scheduler_control_plane_observation", lambda _value: scheduler_observation(submit)
    )

    def runner(command, cwd, environment, inherited_fds):
        nonlocal calls
        assert inherited_fds == ()
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(command, 1, stdout="4312\n", stderr="lost response")
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="squeue unavailable")

    with pytest.raises(submit.SchedulerSubmissionError) as caught:
        submit._submit_one(
            ["sbatch"], job_name="exact", comment="token", squeue="squeue",
            cwd=tmp_path, runner=runner, control_plane=scheduler_contract(),
            fallback=scheduler_fallback(submit),
        )
    assert caught.value.job_ids == ("4312",)


@pytest.mark.parametrize("response", ("0\n", "0007\n"))
def test_noncanonical_sbatch_id_is_never_a_provenance_claim(
    submit, tmp_path, monkeypatch, response
):
    monkeypatch.setattr(
        submit,
        "_scheduler_control_plane_observation",
        lambda _value: scheduler_observation(submit),
    )
    expected_user = submit.pwd.getpwuid(os.getuid()).pw_name
    calls = 0

    def runner(command, cwd, environment, inherited_fds):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(
                command, 1, stdout=response, stderr="lost response"
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"7|exact|{expected_user}|PENDING|token\n",
            stderr="",
        )

    with pytest.raises(submit.SchedulerSubmissionError) as caught:
        submit._submit_one(
            ["sbatch"],
            job_name="exact",
            comment="token",
            squeue="squeue",
            cwd=tmp_path,
            runner=runner,
            control_plane=scheduler_contract(),
            fallback=scheduler_fallback(submit),
        )
    assert caught.value.job_ids == ("7",)
    assert response.strip() not in caught.value.job_ids


def test_federated_sbatch_suffix_is_rejected(submit, tmp_path, monkeypatch):
    monkeypatch.setattr(
        submit, "_scheduler_control_plane_observation", lambda _value: scheduler_observation(submit)
    )
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
            control_plane=scheduler_contract(), fallback=scheduler_fallback(submit),
        )


def test_scheduler_config_parser_rejects_include_duplicate_and_missing_policy(submit):
    contract = scheduler_contract()
    valid = scheduler_config_bytes()
    assert submit._parse_slurm_config(valid, contract)["cluster_name"] == "cs-oci-ord"
    for payload, pattern in (
        (b"Include=/tmp/hostile.conf\n" + valid, "include"),
        (valid + b"ClusterName=cs-oci-ord\n", "ClusterName"),
        (valid.replace(b"CliFilterPlugins=lua\n", b""), "CliFilterPlugins"),
        (valid.replace(b"JobSubmitPlugins=lua\n", b""), "JobSubmitPlugins"),
        (valid.replace(b"AuthType=auth/munge\n", b"AuthType=auth/none\n"), "AuthType"),
    ):
        with pytest.raises(submit.SubmissionError, match=pattern):
            submit._parse_slurm_config(payload, contract)


def test_scheduler_authenticator_rejects_symlink_and_wrong_mode_files(submit):
    root = Path(scheduler_contract()["slurm_conf"]).parent
    with pytest.raises(submit.SubmissionError, match="symlink|regular mode"):
        submit._root_owned_regular_observation(
            root / "slurmdbd.conf", "known canonical-tree symlink adversary"
        )
    with pytest.raises(submit.SubmissionError, match="regular mode 0644"):
        submit._root_owned_regular_observation(
            root / "job_submit.lua_01_04_25", "known wrong-mode policy adversary"
        )


def test_scheduler_policy_missing_module_and_special_entry_rejected(
    submit, tmp_path, monkeypatch
):
    policy = tmp_path / "cli_filters"
    policy.mkdir()
    (policy / "util.lua").write_text("return {}\n", encoding="utf-8")

    def opened(_path, _label):
        descriptor = os.open(policy, os.O_RDONLY | os.O_DIRECTORY)
        return [descriptor], [os.fstat(descriptor)]

    monkeypatch.setattr(submit, "_root_owned_directory_chain", opened)
    with pytest.raises(submit.SubmissionError, match="incomplete"):
        submit._scheduler_policy_observation(tmp_path / "slurm.conf")

    for name in submit.SCHEDULER_REQUIRED_POLICY_MODULES:
        (policy / name).write_text("return {}\n", encoding="utf-8")
    (policy / "hostile").symlink_to(tmp_path)
    with pytest.raises(submit.SubmissionError, match="unsafe"):
        submit._scheduler_policy_observation(tmp_path / "slurm.conf")


def test_scheduler_environment_replaces_hostile_inheritance(submit, monkeypatch):
    monkeypatch.setenv("SLURM_CONF", "/tmp/hostile.conf")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/hostile.so")
    assert submit._scheduler_environment(scheduler_contract()) == {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "SLURM_CONF": scheduler_contract()["slurm_conf"],
    }


def test_live_canonical_scheduler_control_plane_authenticates_read_only(submit):
    observation = submit._scheduler_control_plane_observation(scheduler_contract())
    assert observation["critical"]["cluster_name"] == "cs-oci-ord"
    assert observation["config"]["path"] == scheduler_contract()["slurm_conf"]
    assert len(observation["cli_filter_policy"]["files"]) >= 7


def test_live_fallback_roundtrip_and_memfd_squeue_are_read_only(submit):
    control_plane = scheduler_contract()
    fallback = submit._scheduler_fallback_config(control_plane)
    binding, payload = submit._validated_scheduler_fallback(
        fallback, control_plane, fallback["source_control_plane"]
    )
    assert len(payload) == binding["size"] and binding["size"] > 0
    assert hashlib.sha256(payload).hexdigest() == binding["sha256"]
    user = submit.pwd.getpwuid(os.getuid()).pw_name
    completed, observation = submit._fallback_scheduler_call(
        [
            "/usr/local/bin/squeue",
            "--noheader",
            "--name=treewm-exp23-never-created-fallback-regression",
            f"--user={user}",
            "--format=%A",
        ],
        REPO,
        control_plane,
        fallback,
        submit._default_scheduler_runner,
    )
    assert completed.returncode == 0 and not completed.stdout.strip()
    assert observation["mode"] == "sealed_original_config_fallback"


def test_compute_worker_exposes_no_scheduler_client_surface(submit, worker):
    del submit
    source = inspect.getsource(worker).lower()
    for scheduler_client in ("scontrol", "squeue", "sbatch", "scancel", "sacct"):
        assert scheduler_client not in source
    for removed_api in (
        "authenticated_requeue",
        "_open_root_owned_scheduler_executable",
        "_sealed_scheduler_config_descriptor",
        "_default_scheduler_runner",
    ):
        assert not hasattr(worker, removed_api)


def test_scheduler_preclaim_is_exactly_ten_read_only_calls(
    submit, monkeypatch
):
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    stable = scheduler_observation(submit)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        submit, "_scheduler_control_plane_observation", lambda _value: stable
    )

    def runner(command, _cwd, environment, inherited_fds):
        values = list(command)
        commands.append(values)
        assert environment == submit._scheduler_environment(scheduler_contract())
        assert inherited_fds == ()
        executable = Path(values[0]).name
        if executable == "scontrol":
            return subprocess.CompletedProcess(
                values,
                0,
                stdout=(
                    "ClusterName = cs-oci-ord\n"
                    "SlurmctldHost[0] = cs-oci-ord-a\n"
                    "SlurmctldHost[1] = cs-oci-ord-b\n"
                    "SlurmctldPort = 6817\n"
                    "AuthType = auth/munge\n"
                    "GresTypes = gpu\n"
                    "CliFilterPlugins = lua\n"
                    "JobSubmitPlugins = lua\n"
                ),
                stderr="",
            )
        if executable == "squeue":
            return subprocess.CompletedProcess(values, 0, stdout="", stderr="")
        assert executable == "sbatch" and "--test-only" in values
        if any(str(value).endswith("report.slurm") for value in values):
            role = "report"
        else:
            role = "wave0" if "--hold" in values else "wave1"
        execution = manifest["execution"]
        job_name = next(value.split("=", 1)[1] for value in values if value.startswith("--job-name="))
        comment = next(value.split("=", 1)[1] for value in values if value.startswith("--comment="))
        output = next(value.split("=", 1)[1] for value in values if value.startswith("--output="))
        options = {
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
            options.update(
                {
                    "array": "0-19%20",
                    "gpus-per-node": str(execution["gpus_per_task"]),
                    "no-requeue": "no-requeue",
                    "signal": f"B:USR1@{execution['signal_seconds_before_end']}",
                }
            )
            if role == "wave0":
                options["hold"] = "set"
        partition = (
            str(execution["gpu_partitions"]).split(",")[0]
            if role in {"wave0", "wave1"}
            else str(execution["cpu_partition"])
        )
        stderr = "\n".join(
            [
                "sbatch: defined options",
                *(f"sbatch: {key} : {value}" for key, value in options.items()),
                "sbatch: end of defined options",
                f"sbatch: Job 999 to start at now using {execution['cpus_per_task']} processors on nodes node in partition {partition}",
            ]
        )
        return subprocess.CompletedProcess(values, 0, stdout="", stderr=stderr + "\n")

    result = submit.scheduler_preclaim_test(REPO, manifest, runner=runner)
    assert result["scheduler_calls"] == 10
    assert result["scheduler_mutation_calls"] == 0
    assert len(commands) == 10
    assert [Path(command[0]).name for command in commands].count("sbatch") == 3
    assert all(
        Path(command[0]).name in {"scontrol", "squeue", "sbatch"}
        for command in commands
    )
    assert all(
        Path(command[0]).name != "sbatch" or "--test-only" in command
        for command in commands
    )
    wave0_sbatch, wave1_sbatch, report_sbatch = [
        command for command in commands if Path(command[0]).name == "sbatch"
    ]
    assert "--array=0-19%20" in wave0_sbatch
    assert "--array=0-19%20" in wave1_sbatch
    assert "--hold" in wave0_sbatch and "--hold" not in wave1_sbatch
    assert not any(value.startswith("--array=") for value in report_sbatch)
    dag_commands = (wave0_sbatch, wave1_sbatch, report_sbatch)
    assert all(any(value.startswith("--comment=") for value in command) for command in dag_commands)
    assert all(any(value.startswith("--output=") for value in command) for command in dag_commands)
    assert all("--parsable" in command for command in dag_commands)
    assert result["dependency_tests"] == submit.DEPENDENCY_TEST_REQUIREMENT


def test_late_report_test_only_requires_exact_dependency_and_kill_policy(submit):
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    execution = manifest["execution"]
    comment = "treewm-exp23:" + "c" * 64
    output = "/sealed/submission/logs/report_%j.out"
    dependency = "afterok:7000"
    options = {
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "cpus-per-task": str(execution["cpus_per_task"]),
        "comment": comment,
        "dependency": dependency,
        "export": "NONE",
        "job-name": "exact-report",
        "kill-on-invalid-dep": "yes",
        "mem": str(execution["memory_per_task"]),
        "nodes": "1",
        "ntasks-per-node": "1",
        "open-mode": "a",
        "output": output,
        "parsable": "set",
        "partition": str(execution["cpu_partition"]),
        "qos": "normal",
        "test-only": "set",
        "time": str(execution["walltime"]),
        "verbose": "3",
    }

    def completed(values):
        stderr = "\n".join(
            [
                "sbatch: defined options",
                *(f"sbatch: {key} : {value}" for key, value in values.items()),
                "sbatch: end of defined options",
                (
                    "sbatch: Job 999 to start at now using "
                    f"{execution['cpus_per_task']} processors on nodes node in partition cpu"
                ),
            ]
        )
        return subprocess.CompletedProcess(["sbatch"], 0, stdout="", stderr=stderr + "\n")

    parsed = submit._parse_sbatch_test_only(
        completed(options),
        role="report",
        job_name="exact-report",
        comment=comment,
        output=output,
        manifest=manifest,
        dependency=dependency,
    )
    assert parsed["defined_options"]["dependency"] == dependency
    assert parsed["defined_options"]["kill-on-invalid-dep"] == "yes"
    missing_kill = dict(options)
    del missing_kill["kill-on-invalid-dep"]
    with pytest.raises(submit.SubmissionError, match="options differ"):
        submit._parse_sbatch_test_only(
            completed(missing_kill),
            role="report",
            job_name="exact-report",
            comment=comment,
            output=output,
            manifest=manifest,
            dependency=dependency,
        )


def test_accepted_report_dependency_is_verified_before_ready(
    submit, tmp_path, monkeypatch
):
    stable = scheduler_observation(submit)
    monkeypatch.setattr(
        submit, "_scheduler_control_plane_observation", lambda _value: stable
    )
    observations: list[dict] = []

    def runner(command, _cwd, environment, inherited_fds):
        values = list(command)
        assert values == ["scontrol", "show", "job", "7001", "--oneliner"]
        assert environment == submit._scheduler_environment(scheduler_contract())
        assert inherited_fds == ()
        return subprocess.CompletedProcess(
            values,
            0,
            stdout=(
                "JobId=7001 JobName=exact-report JobState=PENDING "
                "Dependency=afterok:7000_*(unfulfilled) "
                "KillOInInvalidDependent=Yes "
                f"Comment=treewm-exp23:{'c' * 64}\n"
            ),
            stderr="",
        )

    evidence = submit._accepted_dependency_evidence(
        scontrol="scontrol",
        job_id="7001",
        predecessor_id="7000",
        job_name="exact-report",
        role="report",
        predecessor_kind="array",
        comment="treewm-exp23:" + "c" * 64,
        cwd=tmp_path,
        runner=runner,
        control_plane=scheduler_contract(),
        expected_observation=stable,
        observations=observations,
    )
    assert evidence["dependency"] == "afterok:7000_*(unfulfilled)"
    assert evidence["kill_on_invalid_dependency"] == "Yes"
    assert len(observations) == 1
    with pytest.raises(submit.SubmissionError, match="Dependency"):
        submit._scontrol_oneliner_field(
            "Dependency=afterok:7000 Dependency=afterany:7000\n", "Dependency"
        )


@pytest.mark.parametrize(
    "flow,role,predecessor_kind,canonical_dependency,swapped_dependency",
    (
        (
            "production",
            "wave1",
            "array",
            "afterok:7000_*(unfulfilled)",
            "afterok:7000(unfulfilled)",
        ),
        (
            "production",
            "report",
            "array",
            "afterok:7000_*(unfulfilled)",
            "afterok:7000(unfulfilled)",
        ),
        (
            "canary",
            "wave1",
            "scalar",
            "afterok:7000(unfulfilled)",
            "afterok:7000_*(unfulfilled)",
        ),
        (
            "canary",
            "report",
            "scalar",
            "afterok:7000(unfulfilled)",
            "afterok:7000_*(unfulfilled)",
        ),
    ),
    ids=(
        "production-wave1-array-predecessor",
        "production-report-array-predecessor",
        "canary-wave1-scalar-predecessor",
        "canary-report-scalar-predecessor",
    ),
)
def test_dependency_evidence_accepts_only_the_explicit_predecessor_kind(
    submit,
    tmp_path,
    monkeypatch,
    flow,
    role,
    predecessor_kind,
    canonical_dependency,
    swapped_dependency,
):
    stable = scheduler_observation(submit)
    monkeypatch.setattr(
        submit, "_scheduler_control_plane_observation", lambda _value: stable
    )
    observed_dependency = canonical_dependency

    def runner(command, _cwd, _environment, _inherited_fds):
        values = list(command)
        return subprocess.CompletedProcess(
            values,
            0,
            stdout=(
                f"JobId=7001 JobName={flow}-{role} JobState=PENDING "
                f"Dependency={observed_dependency} "
                "KillOInInvalidDependent=Yes "
                f"Comment=treewm-exp23:{'d' * 64}\n"
            ),
            stderr="",
        )

    kwargs = {
        "scontrol": "scontrol",
        "job_id": "7001",
        "predecessor_id": "7000",
        "job_name": f"{flow}-{role}",
        "role": role,
        "predecessor_kind": predecessor_kind,
        "comment": "treewm-exp23:" + "d" * 64,
        "cwd": tmp_path,
        "runner": runner,
        "control_plane": scheduler_contract(),
        "expected_observation": stable,
        "observations": [],
    }
    accepted = submit._accepted_dependency_evidence(**kwargs)
    assert accepted["dependency"] == canonical_dependency
    observed_dependency = swapped_dependency
    with pytest.raises(submit.SubmissionError, match="dependency differs"):
        submit._accepted_dependency_evidence(**kwargs)


@pytest.mark.parametrize("invalid_kind", ("both", None, [], True))
def test_dependency_evidence_rejects_invalid_predecessor_kind_before_scheduler_call(
    submit, tmp_path, invalid_kind
):
    called = False

    def runner(*_args):
        nonlocal called
        called = True
        raise AssertionError("scheduler runner must not be called")

    with pytest.raises(
        submit.SubmissionError,
        match="accepted dependency predecessor kind differs",
    ):
        submit._accepted_dependency_evidence(
            scontrol="scontrol",
            job_id="7001",
            predecessor_id="7000",
            job_name="exact-report",
            role="report",
            predecessor_kind=invalid_kind,
            comment="treewm-exp23:" + "d" * 64,
            cwd=tmp_path,
            runner=runner,
            control_plane=scheduler_contract(),
            expected_observation={},
            observations=[],
        )
    assert not called


def test_dependency_predecessor_kind_is_bound_at_all_four_runtime_edges(submit):
    production_source = inspect.getsource(submit._submit_campaign_impl)
    canary_source = (PACKAGE / "two_wave_canary.py").read_text(encoding="utf-8")
    assert production_source.count('predecessor_kind="array"') == 2
    assert 'predecessor_kind="scalar"' not in production_source
    assert canary_source.count('predecessor_kind="scalar"') == 2
    assert 'predecessor_kind="array"' not in canary_source


@pytest.mark.parametrize(
    "dependency_fields,kill_fields,error",
    [
        (["afterok:7000(unfulfilled)"], ["Yes"], "dependency differs"),
        (["afterok:7000_0(unfulfilled)"], ["Yes"], "dependency differs"),
        (["afterok:7001_*(unfulfilled)"], ["Yes"], "dependency differs"),
        (["afterok:7000_[0-19](unfulfilled)"], ["Yes"], "dependency differs"),
        (["afterok:7000_*"], ["Yes"], "dependency differs"),
        (["afterok:7000_*(fulfilled)"], ["Yes"], "dependency differs"),
        (
            ["afterok:7000_*(unfulfilled),afterany:7000_*(unfulfilled)"],
            ["Yes"],
            "dependency differs",
        ),
        ([], ["Yes"], "Dependency"),
        (
            ["afterok:7000_*(unfulfilled)", "afterok:7000_*(unfulfilled)"],
            ["Yes"],
            "Dependency",
        ),
        (["afterok:7000_*(unfulfilled)"], ["No"], "invalid-dependency policy"),
        (["afterok:7000_*(unfulfilled)"], [], "KillOInInvalidDependent"),
        (
            ["afterok:7000_*(unfulfilled)"],
            ["Yes", "Yes"],
            "KillOInInvalidDependent",
        ),
    ],
    ids=[
        "scalar",
        "single-array-task",
        "wrong-parent",
        "wrong-array-suffix",
        "missing-state",
        "fulfilled-state",
        "multiple-dependencies",
        "missing-dependency",
        "duplicate-dependency",
        "kill-disabled",
        "kill-missing",
        "kill-duplicate",
    ],
)
def test_accepted_report_dependency_rejects_noncanonical_scheduler_records(
    submit,
    tmp_path,
    monkeypatch,
    dependency_fields,
    kill_fields,
    error,
):
    stable = scheduler_observation(submit)
    monkeypatch.setattr(
        submit, "_scheduler_control_plane_observation", lambda _value: stable
    )

    def runner(command, _cwd, _environment, _inherited_fds):
        fields = [
            "JobId=7001",
            "JobName=exact-report",
            "JobState=PENDING",
            *(f"Dependency={value}" for value in dependency_fields),
            *(f"KillOInInvalidDependent={value}" for value in kill_fields),
            f"Comment=treewm-exp23:{'c' * 64}",
        ]
        return subprocess.CompletedProcess(
            list(command), 0, stdout=" ".join(fields) + "\n", stderr=""
        )

    with pytest.raises(submit.SubmissionError, match=error):
        submit._accepted_dependency_evidence(
            scontrol="scontrol",
            job_id="7001",
            predecessor_id="7000",
            job_name="exact-report",
            role="report",
            predecessor_kind="array",
            comment="treewm-exp23:" + "c" * 64,
            cwd=tmp_path,
            runner=runner,
            control_plane=scheduler_contract(),
            expected_observation=stable,
            observations=[],
        )


def test_scheduler_trust_model_explicitly_bounds_mutable_external_runtime(submit):
    trust = scheduler_contract()["trust_model"]
    assert trust == submit.SCHEDULER_TRUST_MODEL
    assert "config and Lua policy bytes are observation-bound" in trust
    assert "clients, plugin binaries, and shared libraries are trusted mutable" in trust


def test_accepted_id_is_exactly_cancelled_with_retained_config_after_critical_drift(
    submit, tmp_path, monkeypatch
):
    stable = scheduler_observation(submit)
    observed = 0
    calls: list[tuple[list[str], dict[str, str], tuple[int, ...]]] = []
    expected_user = submit.pwd.getpwuid(os.getuid()).pw_name

    def observe(_control):
        nonlocal observed
        observed += 1
        if observed == 1:
            return stable
        raise submit.SubmissionError("injected critical controller drift")

    def runner(command, _cwd, environment, inherited_fds):
        values = list(command)
        fds = tuple(inherited_fds)
        calls.append((values, dict(environment), fds))
        if Path(values[0]).name == "sbatch":
            assert environment["SLURM_CONF"] == scheduler_contract()["slurm_conf"]
            assert not fds
            return subprocess.CompletedProcess(values, 0, stdout="4312\n", stderr="")
        assert environment["SLURM_CONF"].startswith("/proc/self/fd/")
        assert len(fds) == 1
        assert os.pread(fds[0], len(scheduler_config_bytes()), 0) == scheduler_config_bytes()
        if Path(values[0]).name == "squeue":
            return subprocess.CompletedProcess(
                values,
                0,
                stdout=f"4312|exact|{expected_user}|PENDING|token\n",
                stderr="",
            )
        assert values == ["scancel", "4312"]
        return subprocess.CompletedProcess(values, 0, stdout="", stderr="")

    monkeypatch.setattr(submit, "_scheduler_control_plane_observation", observe)
    fallback = scheduler_fallback(submit)
    with pytest.raises(submit.SchedulerSubmissionError) as caught:
        submit._submit_one(
            ["sbatch"],
            job_name="exact",
            comment="token",
            squeue="squeue",
            cwd=tmp_path,
            runner=runner,
            control_plane=scheduler_contract(),
            fallback=fallback,
        )
    assert caught.value.job_ids == ("4312",)
    cancellation = submit._cancel_exact(
        "scancel",
        caught.value.job_ids,
        tmp_path,
        runner,
        scheduler_contract(),
        fallback,
        stable,
    )
    assert cancellation["job_ids"] == ["4312"]
    assert [Path(command[0]).name for command, _env, _fds in calls] == [
        "sbatch",
        "squeue",
        "scancel",
    ]
    assert all(
        "--test-only" not in command or Path(command[0]).name == "sbatch"
        for command, _env, _fds in calls
    )


def test_retained_scheduler_config_can_never_authorize_sbatch(
    submit, tmp_path
):
    with pytest.raises(submit.SubmissionError, match="restricted"):
        submit._fallback_scheduler_call(
            ["sbatch", "--test-only"],
            tmp_path,
            scheduler_contract(),
            scheduler_fallback(submit),
            lambda *_args: pytest.fail("forbidden fallback sbatch reached runner"),
        )


def test_submission_authorization_binds_preclaim_fallback_and_every_real_call(
    submit, tmp_path, monkeypatch
):
    authorized = scheduler_observation(submit)
    drifted = json.loads(json.dumps(authorized))
    drifted["cli_filter_policy"]["tree_sha256"] = "b" * 64
    fallback = scheduler_fallback(submit)
    fallback["source_control_plane"] = drifted
    with pytest.raises(submit.SubmissionError, match="changed after the exact preclaim"):
        submit._validated_scheduler_fallback(
            fallback, scheduler_contract(), authorized
        )

    monkeypatch.setattr(
        submit, "_scheduler_control_plane_observation", lambda _value: drifted
    )
    with pytest.raises(submit.SchedulerBoundaryError, match="preclaim authorization") as caught:
        submit._scheduler_call(
            ["sbatch"],
            tmp_path,
            scheduler_contract(),
            lambda *_args: pytest.fail("drifted policy reached real sbatch"),
            authorized,
        )
    assert caught.value.completed is None


def test_policy_drift_before_report_never_calls_report_and_cancels_train_exactly(
    submit, tmp_path, monkeypatch
):
    authorized = scheduler_observation(submit)
    fallback = scheduler_fallback(submit)
    drifted = False
    expected_user = submit.pwd.getpwuid(os.getuid()).pw_name
    calls: list[list[str]] = []

    def observe(_value):
        if drifted:
            changed = json.loads(json.dumps(authorized))
            changed["cli_filter_policy"]["tree_sha256"] = "c" * 64
            return changed
        return authorized

    def runner(command, _cwd, environment, inherited_fds):
        values = list(command)
        calls.append(values)
        executable = Path(values[0]).name
        if executable == "sbatch":
            assert any(str(value).endswith("train.slurm") for value in values)
            return subprocess.CompletedProcess(values, 0, stdout="5100\n", stderr="")
        if executable == "squeue":
            if environment["SLURM_CONF"].startswith("/proc/self/fd/"):
                name = next(value.split("=", 1)[1] for value in values if value.startswith("--name="))
                stdout = (
                    f"5100|{name}|{expected_user}|PENDING|train-token\n"
                    if name == "train"
                    else ""
                )
                return subprocess.CompletedProcess(values, 0, stdout=stdout, stderr="")
            return subprocess.CompletedProcess(
                values,
                0,
                stdout=f"5100|train|{expected_user}|PENDING|train-token\n",
                stderr="",
            )
        assert executable == "scancel" and values == ["scancel", "5100"]
        return subprocess.CompletedProcess(values, 0, stdout="", stderr="")

    monkeypatch.setattr(submit, "_scheduler_control_plane_observation", observe)
    train_id, _record = submit._submit_one(
        ["sbatch", "train.slurm"],
        job_name="train",
        comment="train-token",
        squeue="squeue",
        cwd=tmp_path,
        runner=runner,
        control_plane=scheduler_contract(),
        fallback=fallback,
        expected_observation=authorized,
    )
    assert train_id == "5100"
    drifted = True
    with pytest.raises(submit.SchedulerSubmissionError) as caught:
        submit._submit_one(
            ["sbatch", "report.slurm"],
            job_name="report",
            comment="report-token",
            squeue="squeue",
            cwd=tmp_path,
            runner=runner,
            control_plane=scheduler_contract(),
            fallback=fallback,
            expected_observation=authorized,
        )
    assert caught.value.job_ids == ()
    cancellation = submit._cancel_exact(
        "scancel",
        [train_id],
        tmp_path,
        runner,
        scheduler_contract(),
        fallback,
        authorized,
    )
    assert cancellation["job_ids"] == ["5100"]
    assert not any(
        Path(command[0]).name == "sbatch"
        and any(str(value).endswith("report.slurm") for value in command)
        for command in calls
    )


def test_recovery_reconciliation_and_cancel_survive_critical_canonical_drift(
    submit, tmp_path, monkeypatch
):
    fallback = scheduler_fallback(submit)
    expected_user = submit.pwd.getpwuid(os.getuid()).pw_name
    calls: list[list[str]] = []
    monkeypatch.setattr(
        submit,
        "_scheduler_control_plane_observation",
        lambda _value: (_ for _ in ()).throw(
            submit.SubmissionError("injected recovery controller drift")
        ),
    )

    def runner(command, _cwd, environment, inherited_fds):
        values = list(command)
        calls.append(values)
        assert environment["SLURM_CONF"].startswith("/proc/self/fd/")
        assert len(inherited_fds) == 1
        if Path(values[0]).name == "squeue":
            return subprocess.CompletedProcess(
                values,
                0,
                stdout=f"6100|train|{expected_user}|PENDING|token\n",
                stderr="",
            )
        assert values == ["scancel", "6100"]
        return subprocess.CompletedProcess(values, 0, stdout="", stderr="")

    recovered = submit._reconcile_job_ids(
        "squeue",
        "train",
        "token",
        tmp_path,
        runner,
        scheduler_contract(),
        fallback=fallback,
    )
    assert recovered == ["6100"]
    cancellation = submit._cancel_exact(
        "scancel",
        recovered,
        tmp_path,
        runner,
        scheduler_contract(),
        fallback,
        fallback["source_control_plane"],
    )
    assert cancellation["job_ids"] == ["6100"]
    assert [Path(command[0]).name for command in calls] == ["squeue", "scancel"]


def test_recovery_reconciliation_binds_optional_arguments_by_keyword(submit):
    tree = ast.parse(inspect.getsource(submit._recovery_census_rounds))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_reconcile_job_ids"
    ]
    assert len(calls) == 1 and len(calls[0].args) == 6
    keywords = {item.arg: item.value for item in calls[0].keywords}
    assert set(keywords) == {"fallback", "expected_observation", "observations"}
    assert isinstance(keywords["fallback"], ast.Name)
    assert keywords["fallback"].id == "fallback"
    assert isinstance(keywords["expected_observation"], ast.Name)
    assert keywords["expected_observation"].id == "expected_observation"
    assert isinstance(keywords["observations"], ast.Name)
    assert keywords["observations"].id == "observations"


def test_recovery_census_is_reconstructed_from_exact_attempt_stdout(submit):
    roles = {
        "wave0": "exp23-launch8-token-wave0",
        "wave1": "exp23-launch8-token-wave1",
        "report": "exp23-launch8-token-report",
    }
    comment = "treewm-exp23:" + "c" * 64
    user = submit.pwd.getpwuid(os.getuid()).pw_name
    attempts = []
    rounds = []
    cursor = 0
    for round_index in range(3):
        ids = {}
        spans = {}
        for role in ("wave0", "wave1", "report"):
            active = ["123"] if role == "wave0" else []
            stdout = (
                f"123|{roles[role]}|{user}|RUNNING|{comment}\n"
                if active
                else ""
            )
            attempts.append(
                {
                    "command": [
                        "/fixture/squeue",
                        "--noheader",
                        f"--name={roles[role]}",
                        "--format=%A|%j|%u|%T|%k",
                    ],
                    "mode": "authenticated_canonical",
                    "returncode": 0,
                    "stdout": stdout,
                    "stderr": "",
                    "control_plane": {"schema_version": 1},
                    "canonical_boundary_error": None,
                }
            )
            ids[role] = active
            spans[role] = {"start": cursor, "stop": cursor + 1}
            cursor += 1
        rounds.append(
            {
                "round": round_index,
                "job_ids_by_role": ids,
                "scheduler_attempt_spans_by_role": spans,
            }
        )
    validated, settled, stop = submit._validated_recovery_census_rounds(
        rounds,
        attempts=attempts,
        role_names=roles,
        squeue="/fixture/squeue",
        comment=comment,
        label="fixture",
        expected_start=0,
    )
    assert validated == rounds
    assert settled == {"wave0": ["123"], "wave1": [], "report": []}
    assert stop == 9

    forged = json.loads(json.dumps(rounds))
    forged[0]["job_ids_by_role"]["wave1"] = ["999"]
    with pytest.raises(submit.SubmissionError, match="not derived"):
        submit._validated_recovery_census_rounds(
            forged,
            attempts=attempts,
            role_names=roles,
            squeue="/fixture/squeue",
            comment=comment,
            label="fixture",
            expected_start=0,
        )
    with pytest.raises(submit.SubmissionError, match="span"):
        submit._validated_recovery_census_rounds(
            rounds,
            attempts=[],
            role_names=roles,
            squeue="/fixture/squeue",
            comment=comment,
            label="fixture",
            expected_start=0,
        )
    duplicate_rounds = copy.deepcopy(rounds)
    duplicate_attempts = copy.deepcopy(attempts)
    for round_index in range(3):
        wave1_attempt = round_index * 3 + 1
        duplicate_attempts[wave1_attempt]["stdout"] = (
            f"123|{roles['wave1']}|{user}|RUNNING|{comment}\n"
        )
        duplicate_rounds[round_index]["job_ids_by_role"]["wave1"] = ["123"]
    with pytest.raises(submit.SubmissionError, match="multiple roles"):
        submit._validated_recovery_census_rounds(
            duplicate_rounds,
            attempts=duplicate_attempts,
            role_names=roles,
            squeue="/fixture/squeue",
            comment=comment,
            label="fixture",
            expected_start=0,
        )


def test_recovery_cancel_history_is_append_only_and_survives_lost_response(
    submit, tmp_path, monkeypatch
):
    journal = tmp_path / "journal"
    journal.mkdir()
    context = {
        "schema_version": 1,
        "status": "scheduler_calling",
        "campaign_id": submit.CAMPAIGN_ID,
        "submission_sha256": "c" * 64,
        "claim_token": "b" * 64,
        "role": "recovery_cancel",
        "transaction_lock": {
            "path": str(tmp_path / "lock"),
            "device": 1,
            "inode": 2,
            "uid": os.getuid(),
            "mode": 0o600,
        },
    }

    def scheduler_call(command, *_args, **_kwargs):
        return (
            subprocess.CompletedProcess(command, 0, stdout="cancelled\n", stderr=""),
            {"schema_version": 1, "mode": "fixture"},
        )

    monkeypatch.setattr(submit, "_scheduler_call", scheduler_call)
    observations = []
    history, cancellation = submit._append_recovery_cancel_attempt(
        journal,
        calling_prefix="CALLING_RECOVERY_CANCEL",
        result_prefix="RECOVERY_CANCEL_RESULT",
        context=context,
        scancel="/fixture/scancel",
        job_ids=["100", "101", "102"],
        cwd=tmp_path,
        runner=lambda *_args: None,
        control_plane={},
        fallback={},
        expected_observation={"schema_version": 1, "mode": "fixture"},
        observations=observations,
    )
    assert len(history) == 1 and cancellation["job_ids"] == ["100", "101", "102"]
    submit.exclusive_json(
        journal / "CALLING_RECOVERY_CANCEL_0001.json",
        {
            **context,
            "attempt_index": 1,
            "job_ids": ["100"],
            "command": ["/fixture/scancel", "100"],
        },
    )
    history = submit._validated_recovery_cancel_history(
        journal,
        calling_prefix="CALLING_RECOVERY_CANCEL",
        result_prefix="RECOVERY_CANCEL_RESULT",
        context=context,
        scancel="/fixture/scancel",
        expected_control_plane={"schema_version": 1, "mode": "fixture"},
        fallback={},
    )
    assert history[1]["cancellation"] is None
    history, _ = submit._append_recovery_cancel_attempt(
        journal,
        calling_prefix="CALLING_RECOVERY_CANCEL",
        result_prefix="RECOVERY_CANCEL_RESULT",
        context=context,
        scancel="/fixture/scancel",
        job_ids=["100"],
        cwd=tmp_path,
        runner=lambda *_args: None,
        control_plane={},
        fallback={},
        expected_observation={"schema_version": 1, "mode": "fixture"},
        observations=observations,
    )
    assert [item["attempt_index"] for item in history] == [0, 1, 2]
    assert history[1]["result_sha256"] is None
    assert history[2]["job_ids"] == ["100"]


def test_recovery_attempt_ledger_binds_exact_control_planes_and_terminal_summary(
    submit, tmp_path, monkeypatch
):
    expected = {"schema_version": 1, "mode": "canonical-fixture"}
    fallback = {
        "schema_version": 1,
        "sha256": "a" * 64,
        "size": 17,
        "source_control_plane": {"critical": {"cluster_name": "fixture"}},
    }
    canonical = {
        "command": ["/fixture/squeue"],
        "mode": "authenticated_canonical",
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "control_plane": expected,
        "canonical_boundary_error": None,
    }
    assert submit._validated_scheduler_attempt_ledger(
        [canonical], expected_control_plane=expected, fallback=fallback
    ) == [canonical]
    forged = copy.deepcopy(canonical)
    forged["control_plane"] = {"schema_version": 1, "mode": "forged"}
    with pytest.raises(submit.SubmissionError, match="canonical binding"):
        submit._validated_scheduler_attempt_ledger(
            [forged], expected_control_plane=expected, fallback=fallback
        )
    bool_coerced = copy.deepcopy(canonical)
    bool_coerced["control_plane"]["schema_version"] = True
    with pytest.raises(submit.SubmissionError, match="canonical binding"):
        submit._validated_scheduler_attempt_ledger(
            [bool_coerced], expected_control_plane=expected, fallback=fallback
        )
    fallback_row = {
        **canonical,
        "mode": "sealed_original_config_fallback",
        "control_plane": {
            "schema_version": 1,
            "mode": "sealed_original_config_fallback",
            "sha256": "a" * 64,
            "size": 17,
            "critical": {"cluster_name": "fixture"},
        },
        "canonical_boundary_error": "canonical unavailable",
    }
    assert submit._validated_scheduler_attempt_ledger(
        [fallback_row], expected_control_plane=expected, fallback=fallback
    ) == [fallback_row]
    fallback_bool_coerced = copy.deepcopy(fallback_row)
    fallback_bool_coerced["control_plane"]["size"] = True
    with pytest.raises(submit.SubmissionError, match="fallback binding"):
        submit._validated_scheduler_attempt_ledger(
            [fallback_bool_coerced],
            expected_control_plane=expected,
            fallback=fallback,
        )

    journal = tmp_path / "journal"
    journal.mkdir()
    context = {
        "schema_version": 1,
        "status": "scheduler_calling",
        "campaign_id": submit.CAMPAIGN_ID,
        "submission_sha256": "c" * 64,
        "claim_token": "b" * 64,
        "role": "recovery_cancel",
        "transaction_lock": {
            "path": str(tmp_path / "lock"),
            "device": 1,
            "inode": 2,
            "uid": os.getuid(),
            "mode": 0o600,
        },
    }

    def scheduler_call(command, *_args, **_kwargs):
        return (
            subprocess.CompletedProcess(command, 0, stdout="cancelled\n", stderr=""),
            expected,
        )

    monkeypatch.setattr(submit, "_scheduler_call", scheduler_call)
    submit._append_recovery_cancel_attempt(
        journal,
        calling_prefix="CALLING_RECOVERY_CANCEL",
        result_prefix="RECOVERY_CANCEL_RESULT",
        context=context,
        scancel="/fixture/scancel",
        job_ids=["100"],
        cwd=tmp_path,
        runner=lambda *_args: None,
        control_plane={},
        fallback=fallback,
        expected_observation=expected,
        observations=[],
    )
    result_path = journal / "RECOVERY_CANCEL_RESULT_0000.json"
    result = submit.read_json(result_path)
    result["cancellation"]["stdout"] = "forged\n"
    result_path.chmod(0o600)
    result_path.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    result_path.chmod(0o444)
    with pytest.raises(submit.SubmissionError, match="terminal summary"):
        submit._validated_recovery_cancel_history(
            journal,
            calling_prefix="CALLING_RECOVERY_CANCEL",
            result_prefix="RECOVERY_CANCEL_RESULT",
            context=context,
            scancel="/fixture/scancel",
            expected_control_plane=expected,
            fallback=fallback,
        )


def test_initial_exception_cancel_is_intent_first_and_shared_with_recovery(submit):
    helper = inspect.getsource(submit._append_recovery_cancel_attempt)
    calling = helper.index("calling_sha256 = exclusive_json")
    scheduler = helper.index("cancellation = _cancel_exact", calling)
    result = helper.index("exclusive_json(", scheduler)
    assert calling < scheduler < result

    scientific = inspect.getsource(submit._submit_campaign_impl)
    exception = scientific.index("except BaseException as exc:")
    append = scientific.index(
        "abort_evidence = _initial_exception_reconcile_and_cancel", exception
    )
    aborted = scientific.index('"ABORTED"', append)
    assert append < aborted


@pytest.mark.parametrize(
    (
        "prior_claims,exception_claims,active_role,live_role,live_id,"
        "expected_claims,expected_authority,expected_claims_by_role"
    ),
    [
        (
            {"wave0": [], "wave1": [], "report": []},
            ["999999"],
            "wave0",
            None,
            None,
            ["999999"],
            [],
            {"wave0": ["999999"], "wave1": [], "report": []},
        ),
        (
            {"wave0": ["888888"], "wave1": [], "report": []},
            [],
            "wave0",
            "wave1",
            "7000",
            ["7000", "888888"],
            ["7000"],
            {"wave0": ["888888"], "wave1": ["7000"], "report": []},
        ),
        (
            {"wave0": ["7000"], "wave1": [], "report": []},
            ["7000", "7999"],
            "wave1",
            "wave1",
            "8000",
            ["7000", "7999", "8000"],
            ["8000"],
            {
                "wave0": ["7000"],
                "wave1": ["7999", "8000"],
                "report": [],
            },
        ),
    ],
    ids=["fake-stdout-id", "stale-prior-id", "stale-prior-role-stdout"],
)
def test_initial_exception_cancels_only_fresh_exact_scheduler_identity(
    submit,
    tmp_path,
    monkeypatch,
    prior_claims,
    exception_claims,
    active_role,
    live_role,
    live_id,
    expected_claims,
    expected_authority,
    expected_claims_by_role,
):
    journal = tmp_path / "journal"
    journal.mkdir()
    stable = {"schema_version": 1, "mode": "fixture"}
    monkeypatch.setattr(
        submit, "_scheduler_control_plane_observation", lambda _value: stable
    )
    monkeypatch.setattr(submit, "_scheduler_environment", lambda _value: {})
    roles = {
        "wave0": "exact-wave0",
        "wave1": "exact-wave1",
        "report": "exact-report",
    }
    comment = "treewm-exp23:" + "c" * 64
    user = submit.pwd.getpwuid(os.getuid()).pw_name
    calls = []

    def runner(command, _cwd, _environment, _inherited_fds):
        values = list(command)
        calls.append(values)
        if Path(values[0]).name == "squeue":
            name = next(item.split("=", 1)[1] for item in values if item.startswith("--name="))
            stdout = (
                f"{live_id}|{name}|{user}|PENDING|{comment}\n"
                if live_role is not None and name == roles[live_role]
                else ""
            )
            return subprocess.CompletedProcess(values, 0, stdout=stdout, stderr="")
        assert values == ["/fixture/scancel", *expected_authority]
        return subprocess.CompletedProcess(values, 0, stdout="", stderr="")

    result = submit._initial_exception_reconcile_and_cancel(
        exception_job_ids=exception_claims,
        active_role=active_role,
        prior_claimed_ids_by_role=prior_claims,
        role_names=roles,
        squeue="/fixture/squeue",
        scancel="/fixture/scancel",
        scheduler_comment=comment,
        cancel_directory=journal,
        cancel_calling_prefix="CALLING_RECOVERY_CANCEL",
        cancel_result_prefix="RECOVERY_CANCEL_RESULT",
        snapshot_root=tmp_path,
        scheduler_runner=runner,
        control_plane={},
        scheduler_fallback={},
        expected_observation=stable,
        scheduler_observations=[],
        cancel_context={
            "schema_version": 1,
            "status": "scheduler_calling",
            "campaign_id": submit.CAMPAIGN_ID,
            "submission_sha256": "c" * 64,
            "claim_token": "b" * 64,
            "role": "recovery_cancel",
            "transaction_lock": {
                "path": str(tmp_path / "lock"),
                "device": 1,
                "inode": 2,
                "uid": os.getuid(),
                "mode": 0o600,
            },
        },
    )
    assert result["known_job_ids"] == expected_claims
    assert result["job_ids_by_role"] == expected_claims_by_role
    assert result["cancellation_authority_job_ids"] == expected_authority
    scancel_calls = [row for row in calls if Path(row[0]).name == "scancel"]
    assert scancel_calls == (
        [["/fixture/scancel", *expected_authority]] if expected_authority else []
    )

    canary = load("two_wave_canary")
    canary_source = inspect.getsource(canary.submit_real_canary)
    exception = canary_source.index("except BaseException as exc:")
    census = canary_source.index(
        "submit._recovery_census_rounds(", exception
    )
    marker = canary_source.index(
        "_seal_historical_recycled_cleanup_observation(", census
    )
    cancel = canary_source.index(
        "submit._append_recovery_cancel_attempt(", marker
    )
    aborted = canary_source.index('"CANARY_ABORTED.json"', cancel)
    assert exception < census < marker < cancel < aborted


def _minimal_recovery_fixture(
    tmp_path, submit, monkeypatch, *, with_prerequisite=False
):
    repo = tmp_path / "repo"
    submission = repo / "outputs/run/state/submission"
    snapshot = submission / "source-snapshot/repo"
    package = snapshot / submit.PACKAGE_RELATIVE
    journal = submission / "journal"
    package.mkdir(parents=True)
    journal.mkdir()
    executables = {}
    for name in ("sbatch", "scontrol", "squeue", "scancel"):
        path = tmp_path / name
        path.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
        path.chmod(0o755)
        executables[name] = str(path)
    audits = {
        "weight_audit": {"artifact_sha256": "1" * 64},
        "prefix_target_contract": {"artifact_sha256": "2" * 64},
        "resolved_config_contract": {"artifact_sha256": "3" * 64},
        "causal_parity_contract": {"artifact_sha256": "4" * 64},
    }
    manifest = {
        "campaign_id": submit.CAMPAIGN_ID,
        "paths": {"run_root": str(repo / "outputs/run")},
        "execution": {
            **executables,
            "scheduler_control_plane": scheduler_contract(),
        },
        **audits,
    }
    if with_prerequisite:
        live_manifest = json.loads(
            (PACKAGE / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["launch_contract"] = {
            "real_gpu_two_wave_canary": live_manifest["launch_contract"][
                "real_gpu_two_wave_canary"
            ]
        }
        artifact_source = PACKAGE / "canary2_acceptance_provenance.json"
        artifact_target = package / artifact_source.name
        artifact_target.write_bytes(artifact_source.read_bytes())
        artifact_target.chmod(0o444)
    (package / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (package / "protocol.sha256").write_text("e" * 64 + "\n", encoding="ascii")
    for path in (package / "manifest.json", package / "protocol.sha256"):
        path.chmod(0o444)
    package.chmod(0o555)
    package.parent.chmod(0o555)
    snapshot.chmod(0o555)
    observation = scheduler_observation(submit)
    inventory = {"fixture": "f" * 64}
    if with_prerequisite:
        inventory[
            (submit.PACKAGE_RELATIVE / "canary2_acceptance_provenance.json").as_posix()
        ] = submit.file_sha256(package / "canary2_acceptance_provenance.json")
    contract = {
        "schema_version": 1,
        "status": "sealed_for_submission",
        "submission_root": str(submission),
        "snapshot_root": str(snapshot),
        "snapshot_inventory": inventory,
        "snapshot_inventory_sha256": submit.stable_hash(inventory),
        "manifest_sha256": submit.stable_hash(manifest),
        "package_protocol_sha256": "e" * 64,
        "trainer_code_fingerprint": "a" * 64,
        "runtime_sha256": "b" * 64,
        "orchestration_interpreter": "fixture-python",
        "weight_audit_artifact_sha256": "1" * 64,
        "prefix_target_artifact_sha256": "2" * 64,
        "resolved_config_artifact_sha256": "3" * 64,
        "causal_parity_artifact_sha256": "4" * 64,
        "scheduler_control_plane_contract": scheduler_contract(),
        "scheduler_preclaim": {"scheduler_control_plane": observation},
        "scheduler_fallback_config": {"fixture": True},
    }
    if with_prerequisite:
        projection = submit._validated_production_authorization_prerequisite(
            manifest,
            allow_missing=False,
            package_protocol_sha256="e" * 64,
        )
        assert projection is not None
        contract["production_authorization_prerequisite"] = projection
    contract_path = submission / "SUBMISSION_CONTRACT.json"
    submission_sha256 = submit.exclusive_json(contract_path, contract)
    submit.append_journal(
        submission,
        0,
        "CLAIMED",
        {
            "campaign_id": submit.CAMPAIGN_ID,
            "submission_root": str(submission),
            "claim_token": "b" * 64,
            "scientific_output_fingerprint": "d" * 64,
        },
    )
    submit.append_journal(
        submission,
        2,
        "CONTRACT_SEALED",
        {"submission_sha256": submission_sha256, "launch_count": 20},
    )
    lock_binding = {
        "path": str(repo / "outputs/.fixture.transaction.lock"),
        "device": 1,
        "inode": 2,
        "uid": os.getuid(),
        "mode": 0o600,
    }
    fake_campaign = SimpleNamespace(
        read_json=lambda _path: {},
        validate_manifest=lambda *_args: None,
        source_contract=lambda _root: {
            "source_sha256": "a" * 64,
            "runtime_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(submit, "reject_inherited_environment", lambda: None)
    monkeypatch.setattr(submit, "verify_snapshot_files", lambda *_args: None)
    monkeypatch.setattr(
        submit, "activate_isolated_runtime", lambda _manifest: "fixture-python"
    )
    monkeypatch.setattr(submit, "load_campaign", lambda _root: fake_campaign)
    monkeypatch.setattr(submit, "load_dag_evidence", lambda _root: SimpleNamespace())
    monkeypatch.setattr(
        submit,
        "_validated_scheduler_preclaim",
        lambda *_args: {"scheduler_control_plane": observation},
    )
    monkeypatch.setattr(
        submit,
        "_validated_scheduler_fallback",
        lambda *_args: ({"fixture": True}, b"fixture"),
    )
    monkeypatch.setattr(
        submit, "_scheduler_control_plane_observation", lambda _value: observation
    )
    monkeypatch.setattr(submit, "_leased_transaction_lock_binding", lambda _runner: lock_binding)
    monkeypatch.setattr(submit.time, "sleep", lambda _seconds: None)
    return repo, submission, submission_sha256, lock_binding


def _committed_recovery_fixture(
    tmp_path, submit, monkeypatch, *, released=False, with_prerequisite=False
):
    repo, submission, submission_sha256, lock_binding = _minimal_recovery_fixture(
        tmp_path, submit, monkeypatch, with_prerequisite=with_prerequisite
    )
    journal = submission / "journal"
    ids = {"wave0": "7000", "wave1": "8000", "report": "9000"}
    dependencies = {
        "wave0": "none",
        "wave1": "afterok:7000",
        "report": "afterok:8000",
    }
    for role, ordinal in (("wave0", 3), ("wave1", 4), ("report", 5)):
        calling_sha256 = submit.exclusive_json(
            journal / f"CALLING_{role.upper()}.json",
            {
                "schema_version": 1,
                "status": "scheduler_calling",
                "campaign_id": submit.CAMPAIGN_ID,
                "submission_sha256": submission_sha256,
                "claim_token": "b" * 64,
                "role": role,
                "job_name": f"fixture-{role}",
                "scheduler_comment": f"treewm-exp23:{submission_sha256}",
                "command": ["/fixture/sbatch", role],
                "transaction_lock": lock_binding,
            },
        )
        submit.append_journal(
            submission,
            ordinal,
            f"{role.upper()}_SUBMITTED",
            {
                "job_id": ids[role],
                "command": ["/fixture/sbatch", role],
                "returncode": 0,
                "stdout": f"{ids[role]}\n",
                "stderr": "",
                "reconciled_job_ids": [ids[role]],
                "scheduler_control_plane": scheduler_observation(submit),
                "calling_sha256": calling_sha256,
            },
        )
    evidence_sha256 = "a" * 64
    authorization = {
        "schema_version": 1,
        "status": "authorized_two_wave_dag",
        "campaign_id": submit.CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "array": "0-19%20",
        "job_ids": ids,
        "dependencies": dependencies,
        "kill_on_invalid_dependency": {"wave1": "yes", "report": "yes"},
        "within_wave_requeue": False,
        "wave0_submitted_held": True,
        "accepted_job_evidence_sha256": evidence_sha256,
        "authorized_at_utc": "2026-08-28T00:00:00Z",
    }
    authorization_sha256 = submit.exclusive_json(
        submission / "SUBMISSION_AUTHORIZATION.json", authorization
    )
    submit.append_journal(
        submission,
        6,
        "DAG_AUTHORIZED",
        {
            "submission_authorization_sha256": authorization_sha256,
            "accepted_job_evidence_sha256": evidence_sha256,
            "job_ids": ids,
            "dependencies": dependencies,
        },
    )
    receipt = {
        "schema_version": 1,
        "status": "committed_two_wave_dag",
        "campaign_id": submit.CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "submission_authorization_sha256": authorization_sha256,
        "array": "0-19%20",
        "wave0_array_job_id": ids["wave0"],
        "wave1_array_job_id": ids["wave1"],
        "report_job_id": ids["report"],
        "wave1_dependency": dependencies["wave1"],
        "report_dependency": dependencies["report"],
        "kill_on_invalid_dependency": {"wave1": "yes", "report": "yes"},
        "within_wave_requeue": False,
        "wave0_submitted_held": True,
    }
    submit.append_journal(submission, 7, "READY_TO_COMMIT", receipt)
    submit.exclusive_json(submission / "SUBMISSION_RECEIPT.json", receipt)
    monkeypatch.setattr(
        submit,
        "load_dag_evidence",
        lambda _root: SimpleNamespace(
            validate_dag_records=lambda *_args, **_kwargs: evidence_sha256
        ),
    )
    if released:
        release_calling_sha256 = submit.exclusive_json(
            journal / "CALLING_WAVE0_RELEASE.json",
            {
                "schema_version": 1,
                "status": "scheduler_calling",
                "campaign_id": submit.CAMPAIGN_ID,
                "submission_sha256": submission_sha256,
                "claim_token": "b" * 64,
                "role": "wave0_release",
                "job_name": f"exp23-launch8-{submission_sha256[:16]}-wave0",
                "scheduler_comment": f"treewm-exp23:{submission_sha256}",
                "command": [str(tmp_path / "scontrol"), "release", ids["wave0"]],
                "transaction_lock": lock_binding,
            },
        )
        submit.append_journal(
            submission,
            8,
            "WAVE0_RELEASED",
            {
                "wave0_array_job_id": ids["wave0"],
                "submission_authorization_sha256": authorization_sha256,
                "calling_sha256": release_calling_sha256,
                "release_evidence": {},
            },
        )
        monkeypatch.setattr(
            submit, "_validated_wave0_release_evidence", lambda *_args, **_kwargs: None
        )
    return repo, submission, submission_sha256, receipt


def _live_canary2_prerequisite_fixture(submit):
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    value = submit._validated_production_authorization_prerequisite(
        manifest,
        allow_missing=False,
        package_protocol_sha256="f" * 64,
    )
    assert value is not None
    return value


@pytest.mark.parametrize(
    "scheduler_case",
    (
        "no-jobs",
        "no-jobs-missing-scancel",
        "live-job",
        "live-job-missing-scontrol",
        "live-job-committed-latch",
        "ambiguous",
    ),
)
def test_stale_snapshot_canary_prerequisite_is_cleanup_only_before_commit_or_release(
    submit, tmp_path, monkeypatch, scheduler_case
):
    repo, submission, submission_sha256, _lock = _minimal_recovery_fixture(
        tmp_path, submit, monkeypatch
    )
    # A pre-canary snapshot may already have reached durable READY, but it may
    # never turn that old authority into a receipt or wave-zero release.
    submit.append_journal(
        submission,
        7,
        "READY_TO_COMMIT",
        {
            "schema_version": 1,
            "status": "committed_two_wave_dag",
            "campaign_id": submit.CAMPAIGN_ID,
            "submission_sha256": submission_sha256,
            "submission_authorization_sha256": "9" * 64,
            "array": "0-19%20",
            "wave0_array_job_id": "7000",
            "wave1_array_job_id": "8000",
            "report_job_id": "9000",
            "wave1_dependency": "afterok:7000",
            "report_dependency": "afterok:8000",
            "kill_on_invalid_dependency": {"wave1": "yes", "report": "yes"},
            "within_wave_requeue": False,
            "wave0_submitted_held": True,
        },
    )
    prerequisite = _live_canary2_prerequisite_fixture(submit)
    if scheduler_case == "live-job-missing-scontrol":
        (tmp_path / "scontrol").unlink()
    if scheduler_case == "no-jobs-missing-scancel":
        (tmp_path / "scancel").unlink()
    if scheduler_case == "live-job-committed-latch":
        submit.exclusive_json(
            submission / "CANCEL_REQUESTED.json",
            {
                "schema_version": 1,
                "status": "cancel_requested",
                "campaign_id": submit.CAMPAIGN_ID,
                "submission_sha256": submission_sha256,
                "wave0_array_job_id": "7000",
                "wave1_array_job_id": "8000",
                "report_job_id": "9000",
            },
        )
    token = submission_sha256[:16]
    names = {
        role: f"exp23-launch8-{token}-{role}"
        for role in ("wave0", "wave1", "report")
    }
    comment = f"treewm-exp23:{submission_sha256}"
    user = submit.pwd.getpwuid(os.getuid()).pw_name
    calls = []
    wave0_queries = 0
    active = scheduler_case.startswith("live-job")

    def runner(command, _cwd, _environment, _inherited_fds=()):
        nonlocal active, wave0_queries
        values = list(command)
        calls.append(values)
        executable = Path(values[0]).name
        if executable == "squeue":
            name = next(
                item.split("=", 1)[1]
                for item in values
                if item.startswith("--name=")
            )
            if name == names["wave0"]:
                wave0_queries += 1
            visible = name == names["wave0"] and (
                active
                or (
                    scheduler_case == "ambiguous"
                    and wave0_queries in {1, 3}
                )
            )
            stdout = (
                f"7000|{name}|{user}|PENDING|{comment}\n" if visible else ""
            )
            return subprocess.CompletedProcess(values, 0, stdout=stdout, stderr="")
        assert executable == "scancel"
        assert values[1:] == ["7000"]
        marker = submission / "journal/PREREQUISITE_MISSING.json"
        assert marker.is_file()
        marker_value = submit.read_json(marker)
        assert marker_value["live_verified_job_ids"] == ["7000"]
        assert marker_value["authorization_allowed"] is False
        assert not (submission / "SUBMISSION_RECEIPT.json").exists()
        assert not (submission / "journal/0008_WAVE0_RELEASED.json").exists()
        active = False
        return subprocess.CompletedProcess(values, 0, stdout="", stderr="")

    if scheduler_case == "ambiguous":
        with pytest.raises(
            submit.SubmissionError,
            match="did not settle",
        ):
            submit._recover_transaction_locked(
                repo,
                submission,
                scheduler_runner=runner,
                live_production_prerequisite=prerequisite,
                report_cancel_lock_lease=_lock,
            )
        assert not any(Path(row[0]).name == "scancel" for row in calls)
        assert not (submission / "CANCEL_REQUESTED.json").exists()
        assert not (submission / "journal/PREREQUISITE_MISSING.json").exists()
        assert not (submission / "journal/9000_RECOVERY_CANCELLED.json").exists()
        assert not (submission / "journal/9001_PRODUCTION_PREREQUISITE_MISSING.json").exists()
        assert not (submission / "SUBMISSION_RECEIPT.json").exists()
        return

    result = submit._recover_transaction_locked(
        repo,
        submission,
        scheduler_runner=runner,
        live_production_prerequisite=prerequisite,
        report_cancel_lock_lease=_lock,
    )
    assert result["status"] == (
        "production_authorization_prerequisite_missing_cleanup_terminal"
    )
    cleanup = result["cleanup_recovery"]
    assert cleanup["live_verified_job_ids"] == (
        ["7000"] if scheduler_case.startswith("live-job") else []
    )
    assert [Path(row[0]).name for row in calls].count("scancel") == (
        1 if scheduler_case.startswith("live-job") else 0
    )
    assert not any(Path(row[0]).name == "scontrol" for row in calls)
    assert not (submission / "SUBMISSION_RECEIPT.json").exists()
    assert not (submission / "journal/0008_WAVE0_RELEASED.json").exists()
    marker = submit.read_json(submission / "journal/PREREQUISITE_MISSING.json")
    assert marker["authorization_allowed"] is False
    assert marker["receipt_publication_allowed"] is False
    assert marker["release_allowed"] is False
    terminal = submit.read_json(
        submission / "journal/9001_PRODUCTION_PREREQUISITE_MISSING.json"
    )
    assert terminal["recovery_terminal_sha256"] == submit.file_sha256(
        submission / "journal/9000_RECOVERY_CANCELLED.json"
    )
    assert terminal["report_allowed"] is False
    if scheduler_case == "live-job-committed-latch":
        retried = submit._recover_transaction_locked(
            repo,
            submission,
            scheduler_runner=runner,
            live_production_prerequisite=prerequisite,
            report_cancel_lock_lease=_lock,
        )
        assert retried["status"] == (
            "production_authorization_prerequisite_missing_cleanup_terminal"
        )
        assert [Path(row[0]).name for row in calls].count("scancel") == 1


@pytest.mark.parametrize(
    "surface", ("contract_missing", "inventory_missing", "artifact_raw_mismatch")
)
def test_partial_or_detached_snapshot_canary_prerequisite_fails_before_cleanup_mutation(
    submit, tmp_path, monkeypatch, surface
):
    repo, submission, _old_submission_sha256, _lock = _minimal_recovery_fixture(
        tmp_path, submit, monkeypatch, with_prerequisite=True
    )
    contract_path = submission / "SUBMISSION_CONTRACT.json"
    seal_path = submission / "journal/0002_CONTRACT_SEALED.json"
    contract = submit.read_json(contract_path)
    artifact_key = (
        submit.PACKAGE_RELATIVE / "canary2_acceptance_provenance.json"
    ).as_posix()
    if surface == "contract_missing":
        contract.pop("production_authorization_prerequisite")
    elif surface == "inventory_missing":
        contract["snapshot_inventory"].pop(artifact_key)
        contract["snapshot_inventory_sha256"] = submit.stable_hash(
            contract["snapshot_inventory"]
        )
    else:
        artifact = (
            Path(contract["snapshot_root"])
            / submit.PACKAGE_RELATIVE
            / "canary2_acceptance_provenance.json"
        )
        artifact.chmod(0o644)
        artifact.write_bytes(artifact.read_bytes() + b" ")
        artifact.chmod(0o444)

    if surface != "artifact_raw_mismatch":
        contract_path.chmod(0o644)
        contract_path.write_text(
            json.dumps(contract, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        contract_path.chmod(0o444)
        submission_sha256 = submit.file_sha256(contract_path)
        seal_path.chmod(0o644)
        seal_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "record": "contract_sealed",
                    "submission_sha256": submission_sha256,
                    "launch_count": 20,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        seal_path.chmod(0o444)
    before = {
        str(path.relative_to(submission)): submit.file_sha256(path)
        for path in submission.rglob("*")
        if path.is_file()
    }
    calls = []

    def runner(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("detached canary evidence must fail before scheduler")

    with pytest.raises(
        submit.SubmissionError,
        match="canary authorization binding differs|detached canary authorization",
    ):
        submit._recover_transaction_locked(
            repo,
            submission,
            scheduler_runner=runner,
            live_production_prerequisite=_live_canary2_prerequisite_fixture(
                submit
            ),
            report_cancel_lock_lease={
                "path": str(submission / ".REPORT_CANCEL.lock"),
                "device": 1,
                "inode": 2,
                "uid": os.getuid(),
                "mode": 0o600,
            },
        )
    after = {
        str(path.relative_to(submission)): submit.file_sha256(path)
        for path in submission.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert calls == []
    assert not os.path.lexists(submission / "CANCEL_REQUESTED.json")
    assert not os.path.lexists(submission / "journal/PREREQUISITE_MISSING.json")
    assert not os.path.lexists(submission / "journal/9000_RECOVERY_CANCELLED.json")


@pytest.mark.parametrize(
    "crash_point",
    ("after-marker", "after-scancel", "after-recovery-terminal"),
)
def test_stale_snapshot_prerequisite_cleanup_resumes_every_durable_prefix(
    submit, tmp_path, monkeypatch, crash_point
):
    repo, submission, submission_sha256, lock_binding = _minimal_recovery_fixture(
        tmp_path, submit, monkeypatch
    )
    prerequisite = _live_canary2_prerequisite_fixture(submit)
    name = f"exp23-launch8-{submission_sha256[:16]}-wave0"
    comment = f"treewm-exp23:{submission_sha256}"
    user = submit.pwd.getpwuid(os.getuid()).pw_name
    active = True
    calls = []

    def runner(command, _cwd, _environment, _inherited_fds=()):
        nonlocal active
        values = list(command)
        calls.append(values)
        if Path(values[0]).name == "squeue":
            queried = next(
                item.split("=", 1)[1]
                for item in values
                if item.startswith("--name=")
            )
            stdout = (
                f"7000|{name}|{user}|PENDING|{comment}\n"
                if active and queried == name
                else ""
            )
            return subprocess.CompletedProcess(values, 0, stdout=stdout, stderr="")
        assert Path(values[0]).name == "scancel"
        assert values[1:] == ["7000"]
        assert (submission / "journal/PREREQUISITE_MISSING.json").is_file()
        active = False
        return subprocess.CompletedProcess(values, 0, stdout="", stderr="")

    original_cancel = submit._append_recovery_cancel_attempt
    original_append = submit.append_journal
    if crash_point == "after-marker":
        def crash_cancel(*_args, **_kwargs):
            raise RuntimeError("kill after prerequisite marker")

        monkeypatch.setattr(submit, "_append_recovery_cancel_attempt", crash_cancel)
    else:
        target = 9000 if crash_point == "after-scancel" else 9001

        def crash_append(root, index, label, payload):
            if index == target:
                raise RuntimeError(f"kill before journal {target}")
            return original_append(root, index, label, payload)

        monkeypatch.setattr(submit, "append_journal", crash_append)

    with pytest.raises(RuntimeError, match="kill"):
        submit._recover_transaction_locked(
            repo,
            submission,
            scheduler_runner=runner,
            live_production_prerequisite=prerequisite,
            report_cancel_lock_lease=lock_binding,
        )
    assert (submission / "journal/PREREQUISITE_MISSING.json").is_file()
    assert not (submission / "SUBMISSION_RECEIPT.json").exists()
    assert not (submission / "journal/0008_WAVE0_RELEASED.json").exists()

    monkeypatch.setattr(submit, "_append_recovery_cancel_attempt", original_cancel)
    monkeypatch.setattr(submit, "append_journal", original_append)
    recovered = submit._recover_transaction_locked(
        repo,
        submission,
        scheduler_runner=runner,
        live_production_prerequisite=prerequisite,
        report_cancel_lock_lease=lock_binding,
    )
    assert recovered["status"] == (
        "production_authorization_prerequisite_missing_cleanup_terminal"
    )
    reused = submit._recover_transaction_locked(
        repo,
        submission,
        scheduler_runner=runner,
        live_production_prerequisite=prerequisite,
        report_cancel_lock_lease=lock_binding,
    )
    assert reused["status"] == recovered["status"]
    assert [Path(row[0]).name for row in calls].count("scancel") == 1
    assert not any(Path(row[0]).name == "scontrol" for row in calls)


def test_stale_snapshot_prerequisite_cleanup_reconciles_residual_reappearance(
    submit, tmp_path, monkeypatch
):
    repo, submission, submission_sha256, lock_binding = _minimal_recovery_fixture(
        tmp_path, submit, monkeypatch
    )
    prerequisite = _live_canary2_prerequisite_fixture(submit)
    name = f"exp23-launch8-{submission_sha256[:16]}-wave0"
    comment = f"treewm-exp23:{submission_sha256}"
    user = submit.pwd.getpwuid(os.getuid()).pw_name
    active_id = "7000"
    calls = []

    def runner(command, _cwd, _environment, _inherited_fds=()):
        nonlocal active_id
        values = list(command)
        calls.append(values)
        if Path(values[0]).name == "squeue":
            queried = next(
                item.split("=", 1)[1]
                for item in values
                if item.startswith("--name=")
            )
            stdout = (
                f"{active_id}|{name}|{user}|PENDING|{comment}\n"
                if active_id is not None and queried == name
                else ""
            )
            return subprocess.CompletedProcess(values, 0, stdout=stdout, stderr="")
        assert Path(values[0]).name == "scancel"
        assert values[1:] == [active_id]
        active_id = None
        return subprocess.CompletedProcess(values, 0, stdout="", stderr="")

    first = submit._recover_transaction_locked(
        repo,
        submission,
        scheduler_runner=runner,
        live_production_prerequisite=prerequisite,
        report_cancel_lock_lease=lock_binding,
    )
    assert first["status"] == (
        "production_authorization_prerequisite_missing_cleanup_terminal"
    )
    active_id = "7001"
    second = submit._recover_transaction_locked(
        repo,
        submission,
        scheduler_runner=runner,
        live_production_prerequisite=prerequisite,
        report_cancel_lock_lease=lock_binding,
    )
    assert second["status"] == first["status"]
    assert len(second["cleanup_recovery"]["residual_reconciliation_chain"]) == 1
    third = submit._recover_transaction_locked(
        repo,
        submission,
        scheduler_runner=runner,
        live_production_prerequisite=prerequisite,
        report_cancel_lock_lease=lock_binding,
    )
    assert third["status"] == first["status"]
    assert len(third["cleanup_recovery"]["residual_reconciliation_chain"]) == 1
    scancels = [
        row for row in calls if Path(row[0]).name == "scancel"
    ]
    assert [row[1:] for row in scancels] == [["7000"], ["7001"]]
    assert not any(Path(row[0]).name == "scontrol" for row in calls)


def test_explicit_existing_root_recovery_precedes_live_manifest_activation(
    submit, tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    submission = tmp_path / "existing-submission"
    repo.mkdir()
    submission.mkdir()
    calls = []

    def recover(actual_repo, actual_submission):
        calls.append((actual_repo, actual_submission))
        return {
            "status": "production_authorization_prerequisite_missing_cleanup_terminal"
        }

    monkeypatch.setattr(submit, "recover_transaction", recover)
    monkeypatch.setattr(
        submit,
        "read_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("live manifest must not be read before explicit recovery")
        ),
    )
    monkeypatch.setattr(
        submit,
        "activate_isolated_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("live runtime must not activate before explicit recovery")
        ),
    )
    assert submit.main(
        [
            "--submit",
            "--repo-root",
            str(repo),
            "--submission-root",
            str(submission),
        ]
    ) == 2
    assert calls == [(repo, submission)]


def test_prerequisite_denial_evidence_survives_live_authority_restoration(
    submit, tmp_path, monkeypatch
):
    repo, submission, _submission_sha256, lock_binding = _minimal_recovery_fixture(
        tmp_path, submit, monkeypatch
    )
    valid = _live_canary2_prerequisite_fixture(submit)

    def runner(command, _cwd, _environment, _inherited_fds=()):
        values = list(command)
        assert Path(values[0]).name == "squeue"
        return subprocess.CompletedProcess(values, 0, stdout="", stderr="")

    original_append = submit.append_journal

    def crash_before_9000(root, index, label, payload):
        if index == 9000:
            raise RuntimeError("kill after stable prerequisite marker")
        return original_append(root, index, label, payload)

    monkeypatch.setattr(submit, "append_journal", crash_before_9000)
    with pytest.raises(RuntimeError, match="stable prerequisite marker"):
        submit._recover_transaction_locked(
            repo,
            submission,
            scheduler_runner=runner,
            live_production_prerequisite=(
                submit._unavailable_live_production_authorization_prerequisite()
            ),
            report_cancel_lock_lease=lock_binding,
        )
    marker_path = submission / "journal/PREREQUISITE_MISSING.json"
    marker_before = submit.read_json(marker_path)
    assert marker_before["live_production_authorization_prerequisite"]["status"] == (
        "live_production_authorization_prerequisite_unavailable"
    )
    monkeypatch.setattr(submit, "append_journal", original_append)
    recovered = submit._recover_transaction_locked(
        repo,
        submission,
        scheduler_runner=runner,
        live_production_prerequisite=valid,
        report_cancel_lock_lease=lock_binding,
    )
    assert recovered["status"] == (
        "production_authorization_prerequisite_missing_cleanup_terminal"
    )
    assert submit.read_json(marker_path) == marker_before
    terminal = submit.read_json(
        submission / "journal/9001_PRODUCTION_PREREQUISITE_MISSING.json"
    )
    assert terminal["live_production_authorization_prerequisite"] == (
        marker_before["live_production_authorization_prerequisite"]
    )


def test_durable_cleanup_precedence_survives_snapshot_authority_restoration(
    submit, tmp_path, monkeypatch
):
    repo, submission, _submission_sha256, lock_binding = _minimal_recovery_fixture(
        tmp_path, submit, monkeypatch, with_prerequisite=True
    )
    contract = submit.read_json(submission / "SUBMISSION_CONTRACT.json")
    restored = contract["production_authorization_prerequisite"]

    def runner(command, _cwd, _environment, _inherited_fds=()):
        values = list(command)
        assert Path(values[0]).name == "squeue"
        return subprocess.CompletedProcess(values, 0, stdout="", stderr="")

    original_append = submit.append_journal

    def crash_before_9000(root, index, label, payload):
        if index == 9000:
            raise RuntimeError("kill after no-authority decision")
        return original_append(root, index, label, payload)

    monkeypatch.setattr(submit, "append_journal", crash_before_9000)
    with pytest.raises(RuntimeError, match="no-authority decision"):
        submit._recover_transaction_locked(
            repo,
            submission,
            scheduler_runner=runner,
            live_production_prerequisite=(
                submit._unavailable_live_production_authorization_prerequisite()
            ),
            report_cancel_lock_lease=lock_binding,
        )
    marker_before = submit.read_json(
        submission / "journal/PREREQUISITE_MISSING.json"
    )
    assert marker_before["prerequisite_denial_reason"] == "live_unavailable"
    monkeypatch.setattr(submit, "append_journal", original_append)
    recovered = submit._recover_transaction_locked(
        repo,
        submission,
        scheduler_runner=runner,
        live_production_prerequisite=restored,
        report_cancel_lock_lease=lock_binding,
    )
    assert recovered["status"] == (
        "production_authorization_prerequisite_missing_cleanup_terminal"
    )
    assert submit.read_json(
        submission / "journal/PREREQUISITE_MISSING.json"
    ) == marker_before
    assert not (submission / "SUBMISSION_RECEIPT.json").exists()
    assert not (submission / "journal/0008_WAVE0_RELEASED.json").exists()


def test_orphan_prerequisite_terminal_blocks_continuation_before_scheduler(
    submit, tmp_path, monkeypatch
):
    repo, submission, _submission_sha256, _receipt = _committed_recovery_fixture(
        tmp_path, submit, monkeypatch, with_prerequisite=True
    )
    contract = submit.read_json(submission / "SUBMISSION_CONTRACT.json")
    prerequisite = contract["production_authorization_prerequisite"]
    submit.exclusive_json(
        submission / "journal/9001_PRODUCTION_PREREQUISITE_MISSING.json",
        {"schema_version": 1, "status": "detached_terminal_fixture"},
    )
    before = {
        str(path.relative_to(submission)): submit.file_sha256(path)
        for path in submission.rglob("*")
        if path.is_file()
    }
    calls = []

    def runner(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("detached prerequisite terminal must precede scheduler")

    with pytest.raises(
        submit.SubmissionError,
        match="production prerequisite terminal lacks its durable cleanup prefix",
    ):
        submit._recover_transaction_locked(
            repo,
            submission,
            scheduler_runner=runner,
            live_production_prerequisite=prerequisite,
            report_cancel_lock_lease={
                "path": str(submission / ".REPORT_CANCEL.lock"),
                "device": 1,
                "inode": 2,
                "uid": os.getuid(),
                "mode": 0o600,
            },
        )
    after = {
        str(path.relative_to(submission)): submit.file_sha256(path)
        for path in submission.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert calls == []
    assert not (submission / "CANCEL_REQUESTED.json").exists()
    assert not (submission / "journal/0008_WAVE0_RELEASED.json").exists()


def test_public_recovery_holds_transaction_then_report_cancel_locks(
    submit, tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    submission = tmp_path / "run/state/submission"
    repo.mkdir()
    submission.mkdir(parents=True)
    prerequisite = {"schema_version": 1, "status": "fixture"}
    monkeypatch.setattr(
        submit,
        "_validated_live_production_authorization_prerequisite",
        lambda _repo: prerequisite,
    )
    observed = {}

    def recover_locked(
        actual_repo,
        actual_submission,
        *,
        scheduler_runner,
        live_production_prerequisite,
        report_cancel_lock_lease,
    ):
        observed["repo"] = actual_repo
        observed["submission"] = actual_submission
        observed["transaction"] = submit._leased_transaction_lock_binding(
            scheduler_runner
        )
        observed["report"] = dict(report_cancel_lock_lease)
        assert live_production_prerequisite == prerequisite
        for binding in (observed["transaction"], observed["report"]):
            descriptor = os.open(binding["path"], os.O_RDWR)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(descriptor)
        return {"status": "fixture"}

    monkeypatch.setattr(submit, "_recover_transaction_locked", recover_locked)
    assert submit.recover_transaction(repo, submission)["status"] == "fixture"
    assert observed["repo"] == repo
    assert observed["submission"] == submission
    assert observed["transaction"]["path"].endswith(".transaction.lock")
    assert observed["report"]["path"].endswith(".REPORT_CANCEL.lock")


@pytest.mark.parametrize("conflicting_cleanup_prefix", (False, True))
def test_recovery_honors_report_commit_that_won_shared_lock_before_any_scheduler_call(
    submit, report, tmp_path, monkeypatch, conflicting_cleanup_prefix
):
    repo, submission, submission_sha256, _receipt = _committed_recovery_fixture(
        tmp_path, submit, monkeypatch, with_prerequisite=True
    )
    contract = submit.read_json(submission / "SUBMISSION_CONTRACT.json")
    prerequisite = contract["production_authorization_prerequisite"]
    def unavailable_live_authority(_root):
        raise submit.SubmissionError("live package unavailable after report commit")

    monkeypatch.setattr(
        submit,
        "_validated_live_production_authorization_prerequisite",
        unavailable_live_authority,
    )
    (tmp_path / "scontrol").unlink()
    decision_body = {"status": "rejected"}
    decision = {
        **decision_body,
        "gate_sha256": report.stable_hash(decision_body),
    }
    report._publish_report_locked(
        submission,
        submission_sha256,
        {"schema_version": 1},
        decision,
        {
            "schema_version": 1,
            "campaign_id": submit.CAMPAIGN_ID,
            "submission_sha256": submission_sha256,
            "production_authorization_prerequisite": prerequisite,
            "production_authorization_prerequisite_sha256": submit.stable_hash(
                prerequisite
            ),
        },
    )
    if conflicting_cleanup_prefix:
        submit.exclusive_json(
            submission / "journal/PREREQUISITE_MISSING.json",
            {"schema_version": 1, "status": "cleanup_only_fixture"},
        )
    before = {
        str(path.relative_to(submission)): submit.file_sha256(path)
        for path in submission.rglob("*")
        if path.is_file()
    }
    calls = []

    def runner(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("scheduler must not be queried after report commit")

    recover_kwargs = {
        "scheduler_runner": runner,
        "live_production_prerequisite": prerequisite,
        "report_cancel_lock_lease": {
            "path": str(submission / ".REPORT_CANCEL.lock"),
            "device": 1,
            "inode": 2,
            "uid": os.getuid(),
            "mode": 0o600,
        },
    }
    if conflicting_cleanup_prefix:
        with pytest.raises(
            submit.SubmissionError,
            match="report conflicts with a durable cancellation/cleanup prefix",
        ):
            submit._recover_transaction_locked(
                repo, submission, **recover_kwargs
            )
        after = {
            str(path.relative_to(submission)): submit.file_sha256(path)
            for path in submission.rglob("*")
            if path.is_file()
        }
        assert after == before
        assert calls == []
        return
    result = submit._recover_transaction_locked(
        repo, submission, **recover_kwargs
    )
    assert result["status"] == "report_already_committed_before_recovery"
    assert result["recovery"] == "report_commit_precedence"
    assert result["scheduler_calls"] == 0
    assert result["new_jobs_created"] == 0
    assert result["authorization_allowed"] is False
    assert result["release_allowed"] is False
    assert result["report_allowed"] is False
    assert calls == []
    assert not os.path.lexists(submission / "CANCEL_REQUESTED.json")
    assert not os.path.lexists(submission / "journal/PREREQUISITE_MISSING.json")


@pytest.mark.parametrize("boundary", ("receipt", "release"))
def test_recovery_revalidates_canary_prerequisite_at_receipt_and_release_boundaries(
    submit, tmp_path, monkeypatch, boundary
):
    repo, submission, _submission_sha256, _receipt = _committed_recovery_fixture(
        tmp_path, submit, monkeypatch, with_prerequisite=True
    )
    if boundary == "receipt":
        (submission / "SUBMISSION_RECEIPT.json").unlink()
    contract = submit.read_json(submission / "SUBMISSION_CONTRACT.json")
    prerequisite = contract["production_authorization_prerequisite"]
    drifted = copy.deepcopy(prerequisite)
    drifted["state_file_map_canonical_sha256"] = "0" * 64
    observations = (
        [prerequisite, drifted]
        if boundary == "receipt"
        else [prerequisite, prerequisite, drifted]
    )

    def observe_live(_root):
        assert observations
        return observations.pop(0)

    monkeypatch.setattr(
        submit,
        "_validated_live_production_authorization_prerequisite",
        observe_live,
    )
    calls = []

    def runner(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("drifted canary authority must precede scheduler mutation")

    with pytest.raises(
        submit.SubmissionError,
        match="live production authorization changed",
    ):
        submit._recover_transaction_locked(
            repo,
            submission,
            scheduler_runner=runner,
            live_production_prerequisite=prerequisite,
            report_cancel_lock_lease={
                "path": str(submission / ".REPORT_CANCEL.lock"),
                "device": 1,
                "inode": 2,
                "uid": os.getuid(),
                "mode": 0o600,
            },
        )
    assert calls == []
    assert not os.path.lexists(submission / "journal/CALLING_WAVE0_RELEASE.json")
    assert not os.path.lexists(submission / "journal/0008_WAVE0_RELEASED.json")
    if boundary == "receipt":
        assert not os.path.lexists(submission / "SUBMISSION_RECEIPT.json")


@pytest.mark.parametrize("bad_id", (7000, True))
@pytest.mark.parametrize("surface", ("ready", "authorization", "submitted"))
def test_main_recovery_rejects_non_string_persisted_job_ids_without_scheduler_or_writes(
    submit, tmp_path, monkeypatch, surface, bad_id
):
    _repo, submission, submission_sha256, receipt = _committed_recovery_fixture(
        tmp_path, submit, monkeypatch
    )
    if surface == "ready":
        ready = submit.read_json(submission / "journal/0007_READY_TO_COMMIT.json")
        ready["wave0_array_job_id"] = bad_id
        before = {
            str(path.relative_to(submission)): submit.file_sha256(path)
            for path in submission.rglob("*")
            if path.is_file()
        }
        with pytest.raises(submit.SubmissionError, match="job IDs differ"):
            submit._receipt_from_ready_record(ready)
    else:
        path = (
            submission / "SUBMISSION_AUTHORIZATION.json"
            if surface == "authorization"
            else submission / "journal/0003_WAVE0_SUBMITTED.json"
        )
        value = submit.read_json(path)
        if surface == "authorization":
            value["job_ids"]["wave0"] = bad_id
        else:
            value["job_id"] = bad_id
        path.chmod(0o600)
        path.write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        path.chmod(0o444)
        before = {
            str(item.relative_to(submission)): submit.file_sha256(item)
            for item in submission.rglob("*")
            if item.is_file()
        }
        with pytest.raises(submit.SubmissionError, match="job IDs differ|journal differs"):
            submit._validated_dag_authorization(
                submission, submission_sha256, receipt
            )
    after = {
        str(path.relative_to(submission)): submit.file_sha256(path)
        for path in submission.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("bad_id", (7000, True))
def test_main_abort_cancel_and_latch_validators_reject_non_string_job_ids(
    submit, bad_id
):
    abort = {
        "job_ids_by_role": {"wave0": [bad_id], "wave1": [], "report": []},
        "known_job_ids": [bad_id],
    }
    with pytest.raises(submit.SubmissionError, match="wave0 IDs differ"):
        submit._validated_abort_role_ids(abort, "numeric abort")

    cancellation = {
        "cancellation_error": None,
        "cancellation": {
            "job_ids": [bad_id],
            "command": ["/fixture/scancel", bad_id],
            "stdout": "",
            "stderr": "",
            "scheduler_control_plane": {},
            "canonical_boundary_error": None,
            "scheduler_attempts": [{}],
        },
    }
    with pytest.raises(submit.SubmissionError, match="cancellation IDs differ"):
        submit._validated_successful_cancellation(
            cancellation, "/fixture/scancel", ["7000"]
        )

    latch = {
        "schema_version": 1,
        "status": "cancel_requested",
        "campaign_id": submit.CAMPAIGN_ID,
        "submission_sha256": "c" * 64,
        "claim_token": "b" * 64,
        "wave0_array_job_id": bad_id,
        "wave1_array_job_id": None,
        "report_job_id": None,
        "job_ids_by_role": {"wave0": [bad_id], "wave1": [], "report": []},
        "transaction_lock": {
            "path": "/fixture/lock",
            "device": 1,
            "inode": 2,
            "uid": os.getuid(),
            "mode": 0o600,
        },
        "recovery": True,
        "scheduler_id_authority": "fresh_settled_exact_name_census_only",
    }
    with pytest.raises(submit.SubmissionError, match="latch wave0 differs"):
        submit._validated_recovery_cancel_latch(
            latch,
            submission_sha256="c" * 64,
            claim_token="b" * 64,
            transaction_lock=latch["transaction_lock"],
        )


@pytest.mark.parametrize(
    "relative",
    (
        "SUBMISSION_AUTHORIZATION.json",
        "journal/0000_CLAIMED.json",
        "journal/0003_WAVE0_SUBMITTED.json",
        "journal/0004_WAVE1_SUBMITTED.json",
        "journal/0005_REPORT_SUBMITTED.json",
        "journal/CALLING_WAVE0.json",
        "journal/CALLING_WAVE1.json",
        "journal/CALLING_REPORT.json",
        "journal/0006_DAG_AUTHORIZED.json",
    ),
)
@pytest.mark.parametrize("mutation", ("writable", "noncanonical"))
def test_dag_authorization_recovery_rejects_mutable_or_raw_rewritten_prefix(
    submit, tmp_path, monkeypatch, relative, mutation
):
    _repo, submission, submission_sha256, receipt = _committed_recovery_fixture(
        tmp_path, submit, monkeypatch
    )
    path = submission / relative
    if mutation == "writable":
        path.chmod(0o644)
        pattern = "mode differs"
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        path.chmod(0o644)
        path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        path.chmod(0o444)
        pattern = "raw bytes differ"
    with pytest.raises(submit.SubmissionError, match=pattern):
        submit._validated_dag_authorization(
            submission, submission_sha256, receipt
        )


@pytest.mark.parametrize(
    ("relative", "released"),
    (
        ("SUBMISSION_RECEIPT.json", False),
        ("journal/0007_READY_TO_COMMIT.json", False),
        ("journal/CALLING_WAVE0_RELEASE.json", True),
        ("journal/0008_WAVE0_RELEASED.json", True),
    ),
)
@pytest.mark.parametrize("mutation", ("writable", "noncanonical"))
def test_committed_recovery_rejects_mutable_or_raw_rewritten_terminal_prefix(
    submit, tmp_path, monkeypatch, relative, released, mutation
):
    repo, submission, _submission_sha256, receipt = _committed_recovery_fixture(
        tmp_path, submit, monkeypatch, released=released
    )
    if not released:
        submit.exclusive_json(
            submission / "CANCEL_REQUESTED.json",
            {
                "schema_version": 1,
                "status": "cancel_requested",
                "campaign_id": receipt["campaign_id"],
                "submission_sha256": receipt["submission_sha256"],
                "wave0_array_job_id": receipt["wave0_array_job_id"],
                "wave1_array_job_id": receipt["wave1_array_job_id"],
                "report_job_id": receipt["report_job_id"],
            },
        )
    path = submission / relative
    if mutation == "writable":
        path.chmod(0o644)
        pattern = "mode differs"
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        path.chmod(0o644)
        path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        path.chmod(0o444)
        pattern = "raw bytes differ"
    calls = []

    def runner(command, *_args):
        calls.append(list(command))
        raise AssertionError("scheduler must not be reached for invalid authority bytes")

    with pytest.raises(submit.SubmissionError, match=pattern):
        submit._recover_transaction_locked(
            repo,
            submission,
            scheduler_runner=runner,
            live_production_prerequisite=submit._UNENFORCED_RECOVERY_TEST_SEAM,
        )
    assert calls == []


def test_durable_ready_cancel_precedence_needs_no_receipt_or_scheduler_call(
    submit, tmp_path, monkeypatch
):
    repo, submission, _submission_sha256, receipt = _committed_recovery_fixture(
        tmp_path, submit, monkeypatch
    )
    (submission / "SUBMISSION_RECEIPT.json").unlink()
    submit.exclusive_json(
        submission / "CANCEL_REQUESTED.json",
        {
            "schema_version": 1,
            "status": "cancel_requested",
            "campaign_id": receipt["campaign_id"],
            "submission_sha256": receipt["submission_sha256"],
            "wave0_array_job_id": receipt["wave0_array_job_id"],
            "wave1_array_job_id": receipt["wave1_array_job_id"],
            "report_job_id": receipt["report_job_id"],
        },
    )
    calls = []

    def runner(command, *_args):
        calls.append(list(command))
        raise AssertionError("durable cancel precedence must not call the scheduler")

    result = submit._recover_transaction_locked(
        repo,
        submission,
        scheduler_runner=runner,
        live_production_prerequisite=submit._UNENFORCED_RECOVERY_TEST_SEAM,
    )
    assert result["status"] == "committed_two_wave_dag_cancel_requested"
    assert result["recovery"] == "durable_ready_cancel_precedence"
    assert result["scheduler_calls"] == 0
    assert calls == []


def test_recovery_rejects_one_live_scheduler_id_assigned_to_two_roles_before_mutation(
    submit, tmp_path, monkeypatch
):
    repo, submission, submission_sha256, _lock = _minimal_recovery_fixture(
        tmp_path, submit, monkeypatch
    )
    token = submission_sha256[:16]
    names = {
        role: f"exp23-launch8-{token}-{role}"
        for role in ("wave0", "wave1", "report")
    }
    comment = f"treewm-exp23:{submission_sha256}"
    user = submit.pwd.getpwuid(os.getuid()).pw_name
    calls = []

    def runner(command, _cwd, _environment, _inherited_fds):
        values = list(command)
        calls.append(values)
        assert Path(values[0]).name == "squeue"
        job_name = next(
            item.split("=", 1)[1] for item in values if item.startswith("--name=")
        )
        duplicate = job_name in {names["wave0"], names["wave1"]}
        stdout = (
            f"7000|{job_name}|{user}|PENDING|{comment}\n" if duplicate else ""
        )
        return subprocess.CompletedProcess(values, 0, stdout=stdout, stderr="")

    before = {
        str(path.relative_to(submission)): submit.file_sha256(path)
        for path in submission.rglob("*")
        if path.is_file()
    }
    with pytest.raises(submit.SubmissionError, match="multiple roles"):
        submit._recover_transaction_locked(
            repo,
            submission,
            scheduler_runner=runner,
            live_production_prerequisite=submit._UNENFORCED_RECOVERY_TEST_SEAM,
        )
    after = {
        str(path.relative_to(submission)): submit.file_sha256(path)
        for path in submission.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert calls and all(Path(row[0]).name == "squeue" for row in calls)
    assert not (submission / "CANCEL_REQUESTED.json").exists()
    assert not (submission / "journal/9000_RECOVERY_CANCELLED.json").exists()


def test_recovery_rejects_cross_artifact_duplicate_role_claim_before_scheduler(
    submit, tmp_path, monkeypatch
):
    repo, submission, submission_sha256, _lock = _minimal_recovery_fixture(
        tmp_path, submit, monkeypatch
    )
    submit.append_journal(
        submission,
        9999,
        "ABORTED",
        {
            "error": "fixture first abort",
            "known_job_ids": ["7000"],
            "job_ids_by_role": {
                "wave0": ["7000"], "wave1": [], "report": []
            },
            "cancellation_authority_job_ids": [],
            "cancellation_authority_job_ids_by_role": {
                "wave0": [], "wave1": [], "report": []
            },
            "submission_sha256": submission_sha256,
            "reconciliation_errors": [],
            "cancellation": None,
            "cancellation_error": None,
            "cancel_attempt_history": [],
            "scheduler_control_plane_observations": [],
        },
    )
    submit.append_journal(
        submission,
        9998,
        "OUTER_ABORTED",
        {
            "error": "fixture outer abort",
            "receipt_committed": False,
            "known_job_ids": ["7000"],
            "job_ids_by_role": {
                "wave0": [], "wave1": ["7000"], "report": []
            },
            "claim_token": "b" * 64,
            "submission_sha256": submission_sha256,
        },
    )
    calls = []

    def runner(command, *_args):
        calls.append(list(command))
        raise AssertionError("conflicting durable roles must fail before scheduler")

    before = {
        str(path.relative_to(submission)): submit.file_sha256(path)
        for path in submission.rglob("*")
        if path.is_file()
    }
    with pytest.raises(submit.SubmissionError, match="multiple roles"):
        submit._recover_transaction_locked(
            repo,
            submission,
            scheduler_runner=runner,
            live_production_prerequisite=submit._UNENFORCED_RECOVERY_TEST_SEAM,
        )
    after = {
        str(path.relative_to(submission)): submit.file_sha256(path)
        for path in submission.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert calls == []


def test_full_recovery_never_cancels_unverified_abort_id_and_revalidates_prior(
    submit, tmp_path, monkeypatch
):
    repo, submission, submission_sha256, _lock = _minimal_recovery_fixture(
        tmp_path, submit, monkeypatch
    )
    submit.append_journal(
        submission,
        9998,
        "OUTER_ABORTED",
        {
            "error": "fixture lost response",
            "receipt_committed": False,
            "known_job_ids": ["999999"],
            "job_ids_by_role": {
                "wave0": [],
                "wave1": [],
                "report": ["999999"],
            },
            "claim_token": "b" * 64,
            "submission_sha256": submission_sha256,
        },
    )
    calls = []

    def runner(command, _cwd, _environment, _inherited_fds):
        values = list(command)
        calls.append(values)
        assert Path(values[0]).name == "squeue"
        return subprocess.CompletedProcess(values, 0, stdout="", stderr="")

    result = submit._recover_transaction_locked(
        repo,
        submission,
        scheduler_runner=runner,
        live_production_prerequisite=submit._UNENFORCED_RECOVERY_TEST_SEAM,
    )
    assert result["durable_claimed_job_ids"] == ["999999"]
    assert result["live_verified_job_ids"] == []
    assert result["cancelled_live_job_ids"] == []
    assert result["scheduler_calls"] == 9
    assert len(calls) == 9 and not any(Path(row[0]).name == "scancel" for row in calls)

    terminal_path = submission / "journal/9000_RECOVERY_CANCELLED.json"
    terminal = submit.read_json(terminal_path)
    for mutate in (
        lambda value: value["durable_claimed_job_ids_by_role"]["report"].__setitem__(
            0, 999999
        ),
        lambda value: value.__setitem__("new_jobs_created", False),
    ):
        forged_terminal = copy.deepcopy(terminal)
        mutate(forged_terminal)
        terminal_path.chmod(0o600)
        terminal_path.write_text(
            json.dumps(forged_terminal, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        terminal_path.chmod(0o444)
        prior_calls = len(calls)
        with pytest.raises(submit.SubmissionError):
            submit._recover_transaction_locked(
                repo,
                submission,
                scheduler_runner=runner,
                live_production_prerequisite=submit._UNENFORCED_RECOVERY_TEST_SEAM,
            )
        assert len(calls) == prior_calls
    terminal_path.chmod(0o600)
    terminal_path.write_text(
        json.dumps(terminal, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    terminal_path.chmod(0o444)
    terminal["pre_cancel_census_rounds"][0]["job_ids_by_role"]["wave0"] = ["123"]
    terminal_path.chmod(0o600)
    terminal_path.write_text(
        json.dumps(terminal, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    terminal_path.chmod(0o444)
    prior_calls = len(calls)
    with pytest.raises(submit.SubmissionError, match="not derived"):
        submit._recover_transaction_locked(
            repo,
            submission,
            scheduler_runner=runner,
            live_production_prerequisite=submit._UNENFORCED_RECOVERY_TEST_SEAM,
        )
    assert len(calls) == prior_calls


def test_full_recovery_consumes_lost_cancel_response_with_residual_attempt(
    submit, tmp_path, monkeypatch
):
    repo, submission, submission_sha256, lock_binding = _minimal_recovery_fixture(
        tmp_path, submit, monkeypatch
    )
    context = {
        "schema_version": 1,
        "status": "scheduler_calling",
        "campaign_id": submit.CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "claim_token": "b" * 64,
        "role": "recovery_cancel",
        "transaction_lock": lock_binding,
    }
    submit.exclusive_json(
        submission / "journal/CALLING_RECOVERY_CANCEL_0000.json",
        {
            **context,
            "attempt_index": 0,
            "job_ids": ["7000"],
            "command": [str(tmp_path / "scancel"), "7000"],
        },
    )
    initial_history = submit._validated_recovery_cancel_history(
        submission / "journal",
        calling_prefix="CALLING_RECOVERY_CANCEL",
        result_prefix="RECOVERY_CANCEL_RESULT",
        context=context,
        scancel=str(tmp_path / "scancel"),
        expected_control_plane=scheduler_observation(submit),
        fallback={"fixture": True},
    )
    submit.append_journal(
        submission,
        9999,
        "ABORTED",
        {
            "error": "fixture hard death after accepted scancel",
            "known_job_ids": ["7000", "7999"],
            "job_ids_by_role": {
                "wave0": ["7000", "7999"],
                "wave1": [],
                "report": [],
            },
            "cancellation_authority_job_ids": ["7000"],
            "cancellation_authority_job_ids_by_role": {
                "wave0": ["7000"], "wave1": [], "report": []
            },
            "submission_sha256": submission_sha256,
            "reconciliation_errors": [],
            "cancellation": None,
            "cancellation_error": "lost scheduler response",
            "cancel_attempt_history": initial_history,
            "scheduler_control_plane_observations": [],
        },
    )
    active = True
    calls = []
    token = submission_sha256[:16]
    wave0_name = f"exp23-launch8-{token}-wave0"
    comment = f"treewm-exp23:{submission_sha256}"
    user = submit.pwd.getpwuid(os.getuid()).pw_name

    def runner(command, _cwd, _environment, _inherited_fds):
        nonlocal active
        values = list(command)
        calls.append(values)
        if Path(values[0]).name == "squeue":
            name = next(item.split("=", 1)[1] for item in values if item.startswith("--name="))
            stdout = (
                f"7000|{wave0_name}|{user}|PENDING|{comment}\n"
                if active and name == wave0_name
                else ""
            )
            return subprocess.CompletedProcess(values, 0, stdout=stdout, stderr="")
        assert Path(values[0]).name == "scancel" and values[-1] == "7000"
        active = False
        return subprocess.CompletedProcess(values, 0, stdout="", stderr="")

    result = submit._recover_transaction_locked(
        repo,
        submission,
        scheduler_runner=runner,
        live_production_prerequisite=submit._UNENFORCED_RECOVERY_TEST_SEAM,
    )
    assert result["status"] == "recovered_terminal_after_cancel_attempts"
    assert result["durable_claimed_job_ids"] == ["7000", "7999"]
    assert result["live_verified_job_ids"] == ["7000"]
    assert result["post_cancel_active_job_ids_by_role"] == {
        "wave0": [], "wave1": [], "report": []
    }
    assert [row["attempt_index"] for row in result["cancel_attempt_history"]] == [0, 1]
    assert result["cancel_attempt_history"][0]["result_sha256"] is None
    assert result["cancel_attempt_history"][1]["job_ids"] == ["7000"]
    assert [Path(row[0]).name for row in calls].count("scancel") == 1
    assert result["scheduler_calls"] == 19

    active = True
    prior_calls = len(calls)
    resumed = submit._recover_transaction_locked(
        repo,
        submission,
        scheduler_runner=runner,
        live_production_prerequisite=submit._UNENFORCED_RECOVERY_TEST_SEAM,
    )
    assert len(calls) - prior_calls == 19
    assert [Path(row[0]).name for row in calls[prior_calls:]].count("scancel") == 1
    chain = resumed["residual_reconciliation_chain"]
    assert len(chain) == 1
    assert chain[0]["previous_terminal_name"] == "9000_RECOVERY_CANCELLED.json"
    assert chain[0]["live_verified_job_ids"] == ["7000"]
    assert chain[0]["status"] == (
        "recovered_residual_terminal_after_residual_cancel"
    )

    prior_calls = len(calls)
    reused = submit._recover_transaction_locked(
        repo,
        submission,
        scheduler_runner=runner,
        live_production_prerequisite=submit._UNENFORCED_RECOVERY_TEST_SEAM,
    )
    assert len(calls) - prior_calls == 9
    assert len(reused["residual_reconciliation_chain"]) == 1

    residual_path = submission / "journal/RECOVERY_RECONCILED_0000.json"
    residual = submit.read_json(residual_path)
    for mutate, expected_error in (
        (
            lambda value: value["live_verified_job_ids_by_role"]["wave0"].__setitem__(
                0, 7000
            ),
            "residual recovery generation 0 live wave0 IDs differ",
        ),
        (
            lambda value: value["live_verified_job_ids_by_role"]["wave1"].append(
                "7000"
            ),
            (
                "residual recovery generation 0 live assigns one scheduler ID "
                "to multiple roles"
            ),
        ),
        (
            lambda value: value.__setitem__("new_jobs_created", False),
            "residual recovery generation 0 identity differs",
        ),
    ):
        forged_residual = copy.deepcopy(residual)
        mutate(forged_residual)
        residual_path.chmod(0o600)
        residual_path.write_text(
            json.dumps(forged_residual, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        residual_path.chmod(0o444)
        prior_calls = len(calls)
        with pytest.raises(submit.SubmissionError, match=expected_error):
            submit._recover_transaction_locked(
                repo,
                submission,
                scheduler_runner=runner,
                live_production_prerequisite=submit._UNENFORCED_RECOVERY_TEST_SEAM,
            )
        assert len(calls) == prior_calls


def test_recovery_latch_is_historical_not_delayed_visibility_cancel_authority(
    submit, tmp_path, monkeypatch
):
    repo, submission, submission_sha256, lock_binding = _minimal_recovery_fixture(
        tmp_path, submit, monkeypatch
    )
    empty_roles = {"wave0": [], "wave1": [], "report": []}
    latch = {
        "schema_version": 1,
        "status": "cancel_requested",
        "campaign_id": submit.CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "claim_token": "b" * 64,
        "wave0_array_job_id": None,
        "wave1_array_job_id": None,
        "report_job_id": None,
        "job_ids_by_role": empty_roles,
        "transaction_lock": lock_binding,
        "recovery": True,
        "scheduler_id_authority": "fresh_settled_exact_name_census_only",
    }
    submit.exclusive_json(submission / "CANCEL_REQUESTED.json", latch)
    latch_sha256 = submit.file_sha256(submission / "CANCEL_REQUESTED.json")
    token = submission_sha256[:16]
    wave0_name = f"exp23-launch8-{token}-wave0"
    comment = f"treewm-exp23:{submission_sha256}"
    expected_user = submit.pwd.getpwuid(os.getuid()).pw_name
    active = True
    calls: list[list[str]] = []

    def runner(command, _cwd, _environment, _inherited_fds):
        nonlocal active
        values = list(command)
        calls.append(values)
        if Path(values[0]).name == "squeue":
            name = next(
                item.split("=", 1)[1]
                for item in values
                if item.startswith("--name=")
            )
            stdout = (
                f"7000|{wave0_name}|{expected_user}|PENDING|{comment}\n"
                if active and name == wave0_name
                else ""
            )
            return subprocess.CompletedProcess(values, 0, stdout=stdout, stderr="")
        assert values == [str(tmp_path / "scancel"), "7000"]
        active = False
        return subprocess.CompletedProcess(values, 0, stdout="", stderr="")

    result = submit._recover_transaction_locked(
        repo,
        submission,
        scheduler_runner=runner,
        live_production_prerequisite=submit._UNENFORCED_RECOVERY_TEST_SEAM,
    )
    assert result["live_verified_job_ids"] == ["7000"]
    assert result["cancelled_live_job_ids"] == ["7000"]
    assert [row for row in calls if Path(row[0]).name == "scancel"] == [
        [str(tmp_path / "scancel"), "7000"]
    ]
    assert submit.file_sha256(submission / "CANCEL_REQUESTED.json") == latch_sha256
    assert submit.read_json(submission / "CANCEL_REQUESTED.json") == latch


def test_recovery_rejects_writable_historical_latch_before_scheduler_call(
    submit, tmp_path, monkeypatch
):
    repo, submission, submission_sha256, lock_binding = _minimal_recovery_fixture(
        tmp_path, submit, monkeypatch
    )
    submit.exclusive_json(
        submission / "CANCEL_REQUESTED.json",
        {
            "schema_version": 1,
            "status": "cancel_requested",
            "campaign_id": submit.CAMPAIGN_ID,
            "submission_sha256": submission_sha256,
            "claim_token": "b" * 64,
            "wave0_array_job_id": None,
            "wave1_array_job_id": None,
            "report_job_id": None,
            "job_ids_by_role": {"wave0": [], "wave1": [], "report": []},
            "transaction_lock": lock_binding,
            "recovery": True,
            "scheduler_id_authority": "fresh_settled_exact_name_census_only",
        },
    )
    (submission / "CANCEL_REQUESTED.json").chmod(0o644)
    calls = []

    def runner(command, *_args):
        calls.append(list(command))
        raise AssertionError("invalid latch must fail before every scheduler call")

    with pytest.raises(submit.SubmissionError, match="mode differs"):
        submit._recover_transaction_locked(
            repo,
            submission,
            scheduler_runner=runner,
            live_production_prerequisite=submit._UNENFORCED_RECOVERY_TEST_SEAM,
        )
    assert calls == []


def test_scheduler_supervisor_retains_lock_when_client_closes_unknown_fds(
    submit, tmp_path
):
    lock_path = tmp_path / "transaction.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    os.fchmod(descriptor, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    started = tmp_path / "started"
    release = tmp_path / "release"
    code = "\n".join(
        [
            "import os,sys,time",
            "for fd in range(3,256):",
            "    try: os.close(fd)",
            "    except OSError: pass",
            "open(sys.argv[1],'wb').write(b'started')",
            "while not os.path.exists(sys.argv[2]): time.sleep(.01)",
            "sys.stdout.write('child-out\\n')",
            "sys.stderr.write('child-err\\n')",
            "raise SystemExit(7)",
        ]
    )
    result = {}

    def invoke():
        result["completed"] = submit._default_scheduler_runner(
            [sys.executable, "-I", "-S", "-B", "-c", code, str(started), str(release)],
            tmp_path,
            {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            (descriptor,),
        )

    thread = threading.Thread(target=invoke)
    thread.start()
    deadline = time.monotonic() + 5
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started.exists()
    os.close(descriptor)
    contender = os.open(lock_path, os.O_RDWR)
    with pytest.raises(BlockingIOError):
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    release.write_bytes(b"release")
    thread.join(timeout=5)
    assert not thread.is_alive()
    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(contender)
    completed = result["completed"]
    assert completed.returncode == 7
    assert completed.stdout == "child-out\n"
    assert completed.stderr == "child-err\n"


def _seal_lineage_fixture(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.write_bytes(payload)
    path.chmod(0o444)
    return hashlib.sha256(payload).hexdigest()


def _lineage_marker(report, *, wave_index: int) -> dict:
    return {
        "schema_version": 1,
        "campaign_id": report.CAMPAIGN_ID,
        "submission_sha256": "c" * 64,
        "launch_sha256": "d" * 64,
        "cell_index": 3,
        "wave_index": wave_index,
        "array_job_id": "7000" if wave_index == 0 else "8000",
        "array_task_id": 3,
        "predecessor_array_job_id": "none" if wave_index == 0 else "7000",
        "submission_authorization_sha256": "e" * 64,
        "status": "worker_complete",
        "completed_updates": 25_000,
        "checkpoint_sha256": "1" * 64,
        "completion_sha256": "2" * 64,
        "final_eval_progress_sha256": "3" * 64,
        "completed_results_sha256": "4" * 64,
        "identity_sha256": "5" * 64,
        "final_metrics": {"eval/success_rate": 0.5},
    }


def _lineage_start(report, *, wave_index: int, input_kind, checkpoint, predecessor) -> dict:
    return {
        "schema_version": 1,
        "campaign_id": report.CAMPAIGN_ID,
        "submission_sha256": "c" * 64,
        "launch_sha256": "d" * 64,
        "cell_index": 3,
        "wave_index": wave_index,
        "array_job_id": "7000" if wave_index == 0 else "8000",
        "array_task_id": 3,
        "predecessor_array_job_id": "none" if wave_index == 0 else "7000",
        "submission_authorization_sha256": "e" * 64,
        "status": "wave_started",
        "input_kind": input_kind,
        "input_checkpoint_sha256": checkpoint,
        "predecessor_evidence_sha256": predecessor,
    }


def _lineage_terminal(marker: dict) -> dict:
    return {
        key: marker[key]
        for key in (
            "completed_updates",
            "checkpoint_sha256",
            "completion_sha256",
            "final_eval_progress_sha256",
            "completed_results_sha256",
            "identity_sha256",
            "final_metrics",
        )
    }


def test_report_binds_wave0_ready_to_wave1_resume_lineage(report, tmp_path):
    task = tmp_path / "task"
    marker = _lineage_marker(report, wave_index=1)
    _seal_lineage_fixture(
        task / "waves/0/START.json",
        _lineage_start(
            report, wave_index=0, input_kind="fresh", checkpoint=None, predecessor=None
        ),
    )
    ready = {
        **{
            key: value
            for key, value in _lineage_start(
                report, wave_index=0, input_kind="fresh", checkpoint=None, predecessor=None
            ).items()
            if key in report.ARTIFACT_BASE_KEYS
        },
        "status": "continuation_ready",
        "trainer_exit_code": 75,
        "checkpoint_kind": "train",
        "completed_updates": 17_518,
        "phase": "train",
        "pending_eval_step": None,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_file_identity": {
            "device": 1,
            "inode": 2,
            "size": 3,
            "mtime_ns": 4,
            "ctime_ns": 5,
        },
        "final_eval_progress_sha256": None,
    }
    ready_sha = _seal_lineage_fixture(task / "waves/0/CONTINUATION_READY.json", ready)
    start1 = _lineage_start(
        report,
        wave_index=1,
        input_kind="train",
        checkpoint="a" * 64,
        predecessor=ready_sha,
    )
    start1_path = task / "waves/1/START.json"
    _seal_lineage_fixture(start1_path, start1)
    _seal_lineage_fixture(task / "waves/1/WORKER_COMPLETE.json", marker)
    _seal_lineage_fixture(task / "WORKER_COMPLETE.json", marker)
    lineage = report.validate_worker_receipt(
        marker,
        index=3,
        launch={"launch_sha256": "d" * 64},
        submission_sha256="c" * 64,
        wave0_array_job_id="7000",
        wave1_array_job_id="8000",
        submission_authorization_sha256="e" * 64,
        task_root=task,
        terminal=_lineage_terminal(marker),
    )
    assert lineage["branch"] == "wave0_ready_wave1_resume"
    assert lineage["wave0_predecessor_evidence_sha256"] == ready_sha

    bool_marker = {**marker, "schema_version": True}
    with pytest.raises(report.ReportError, match="worker receipt schema differs"):
        report.validate_worker_receipt(
            bool_marker,
            index=3,
            launch={"launch_sha256": "d" * 64},
            submission_sha256="c" * 64,
            wave0_array_job_id="7000",
            wave1_array_job_id="8000",
            submission_authorization_sha256="e" * 64,
            task_root=task,
            terminal=_lineage_terminal(marker),
        )

    ready_path = task / "waves/0/CONTINUATION_READY.json"
    ready_path.chmod(0o600)
    bool_ready = {**ready, "wave_index": False}
    _seal_lineage_fixture(ready_path, bool_ready)
    with pytest.raises(report.ReportError, match="wave0 artifact base differs"):
        report.validate_worker_receipt(
            marker,
            index=3,
            launch={"launch_sha256": "d" * 64},
            submission_sha256="c" * 64,
            wave0_array_job_id="7000",
            wave1_array_job_id="8000",
            submission_authorization_sha256="e" * 64,
            task_root=task,
            terminal=_lineage_terminal(marker),
        )
    ready_path.chmod(0o600)
    _seal_lineage_fixture(ready_path, ready)

    start1_path.chmod(0o600)
    start1["predecessor_evidence_sha256"] = "f" * 64
    _seal_lineage_fixture(start1_path, start1)
    with pytest.raises(report.ReportError, match="checkpoint predecessor lineage"):
        report.validate_worker_receipt(
            marker,
            index=3,
            launch={"launch_sha256": "d" * 64},
            submission_sha256="c" * 64,
            wave0_array_job_id="7000",
            wave1_array_job_id="8000",
            submission_authorization_sha256="e" * 64,
            task_root=task,
            terminal=_lineage_terminal(marker),
        )


def test_report_binds_wave0_complete_to_wave1_noop_lineage(report, tmp_path):
    task = tmp_path / "task"
    marker = _lineage_marker(report, wave_index=0)
    _seal_lineage_fixture(
        task / "waves/0/START.json",
        _lineage_start(
            report, wave_index=0, input_kind="fresh", checkpoint=None, predecessor=None
        ),
    )
    complete_sha = _seal_lineage_fixture(task / "waves/0/WORKER_COMPLETE.json", marker)
    _seal_lineage_fixture(task / "WORKER_COMPLETE.json", marker)
    start1 = _lineage_start(
        report,
        wave_index=1,
        input_kind="complete",
        checkpoint=marker["checkpoint_sha256"],
        predecessor=complete_sha,
    )
    _seal_lineage_fixture(task / "waves/1/START.json", start1)
    noop = {
        **marker,
        "status": "wave_one_noop_after_wave_zero_complete",
        "wave_index": 1,
        "array_job_id": "8000",
        "predecessor_array_job_id": "7000",
    }
    _seal_lineage_fixture(task / "waves/1/WORKER_COMPLETE.json", noop)
    lineage = report.validate_worker_receipt(
        marker,
        index=3,
        launch={"launch_sha256": "d" * 64},
        submission_sha256="c" * 64,
        wave0_array_job_id="7000",
        wave1_array_job_id="8000",
        submission_authorization_sha256="e" * 64,
        task_root=task,
        terminal=_lineage_terminal(marker),
    )
    assert lineage["branch"] == "wave0_complete_wave1_noop"
    assert lineage["wave1_predecessor_evidence_sha256"] == complete_sha
    noop_path = task / "waves/1/WORKER_COMPLETE.json"
    noop_path.chmod(0o600)
    _seal_lineage_fixture(noop_path, {**noop, "wave_index": True})
    with pytest.raises(report.ReportError, match="wave-one completion no-op differs"):
        report.validate_worker_receipt(
            marker,
            index=3,
            launch={"launch_sha256": "d" * 64},
            submission_sha256="c" * 64,
            wave0_array_job_id="7000",
            wave1_array_job_id="8000",
            submission_authorization_sha256="e" * 64,
            task_root=task,
            terminal=_lineage_terminal(marker),
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
                "scheduler_control_plane": {"schema_version": 1},
                "canonical_boundary_error": None,
                "scheduler_attempts": [
                    {
                        "command": ["/usr/bin/scancel", *ids],
                        "mode": "authenticated_canonical",
                        "returncode": 0,
                        "stdout": "",
                        "stderr": "",
                        "control_plane": {"schema_version": 1},
                        "canonical_boundary_error": None,
                    }
                ],
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
    DENSE_TRAIN_METHOD_TAGS = ("dense_method",)
    METHOD_EXACT_TAGS = (
        "method",
        *DENSE_TRAIN_METHOD_TAGS,
        "data/validation_fixed_sample_count",
    )
    GRADIENT_NORM_TAGS = ("grad",)
    GRADIENT_CLIP_TAGS = ("clip",)
    TRAIN_PREFIX = "train/p/"
    PREFIX = "val/p/"
    PREFIX_COMMON_SUFFIXES = ("one",)


def exact_scalars():
    train = tuple(range(50, 25_001, 50))
    validation = tuple(range(1000, 25_001, 1000))
    return {
        **{
            tag: {step: 1.0 for step in train}
            for tag in ("gauge", "grad", "clip", "dense_method", "train/p/one")
        },
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


def test_reporter_binds_real_dense_method_tags_to_training_axis(report):
    gate = load("gate")
    assert gate.DENSE_TRAIN_METHOD_TAGS == (
        "expansion/gain_rank_correlation",
        "expansion/gain_pairwise_accuracy",
        "expansion/gain_eligible_decision_fraction",
        "expansion/gain_ordered_pair_count",
        "expansion/gain_pair_coverage_fraction",
        "tree/support_recall",
        "tree/support_precision",
    )
    train = tuple(range(50, 25_001, 50))
    validation = tuple(range(1000, 25_001, 1000))
    scalars = {
        **{
            tag: {step: 1.0 for step in train}
            for tag in (
                *gate.GAUGE_EXACT_TAGS,
                *gate.GRADIENT_NORM_TAGS,
                *gate.GRADIENT_CLIP_TAGS,
                *gate.DENSE_TRAIN_METHOD_TAGS,
                *(gate.TRAIN_PREFIX + suffix for suffix in gate.PREFIX_COMMON_SUFFIXES),
            )
        },
        **{
            tag: {step: 1.0 for step in validation}
            for tag in (
                *(
                    tag
                    for tag in gate.METHOD_EXACT_TAGS
                    if tag != "data/validation_fixed_sample_count"
                    and tag not in gate.DENSE_TRAIN_METHOD_TAGS
                ),
                *(gate.PREFIX + suffix for suffix in gate.PREFIX_COMMON_SUFFIXES),
            )
        },
        "data/validation_fixed_sample_count": {
            step: 5120.0 for step in (0, *validation)
        },
    }
    manifest = {
        "scientific_contract": {
            "training_telemetry_every_updates": 50,
            "validation_every_updates": 1000,
        }
    }
    report.validate_boundary_axes(scalars, gate, manifest)
    del scalars[gate.DENSE_TRAIN_METHOD_TAGS[0]][50]
    with pytest.raises(report.ReportError, match="full training telemetry.*axis differs"):
        report.validate_boundary_axes(scalars, gate, manifest)


def test_event_parser_excludes_distinct_periodic_and_terminal_eval_namespaces(
    report, tmp_path
):
    from torch.utils.tensorboard import SummaryWriter

    sampler = {"global_sample_size": 5120, "seed": 1701}
    text = "<pre>" + json.dumps(sampler, sort_keys=True, indent=2) + "</pre>"
    writer = SummaryWriter(str(tmp_path), filename_suffix=".generation")
    writer.add_text("meta/fixed_validation_sample", text, 0)
    writer.add_scalar("train/loss_total", 2.0, 50)
    writer.add_scalar("eval/success_rate", 0.2, 25_000)
    writer.add_scalar("eval/final/success_rate", 0.8, 25_000)
    writer.flush()
    writer.close()
    parsed = report.parse_event_files(tmp_path, sampler)
    assert "eval/success_rate" not in parsed["scalars"]
    assert "eval/final/success_rate" not in parsed["scalars"]
    assert parsed["excluded_eval_tags"] == [
        "eval/final/success_rate",
        "eval/success_rate",
    ]
    assert parsed["scalars"]["train/loss_total"] == {50: 2.0}


def test_event_parser_authenticates_wandb_symlink_leaves_only(report, tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    sampler = {"global_sample_size": 5120, "seed": 1701}
    text = "<pre>" + json.dumps(sampler, sort_keys=True, indent=2) + "</pre>"
    writer = SummaryWriter(str(tmp_path))
    writer.add_text("meta/fixed_validation_sample", text, 0)
    writer.add_scalar("train/loss_total", 2.0, 50)
    writer.flush()
    writer.close()

    run = tmp_path / "wandb" / "run-identity"
    logs = run / "logs"
    logs.mkdir(parents=True)
    (logs / "debug.log").write_text("observational\n")
    (tmp_path / "wandb" / "latest-run").symlink_to("run-identity")
    (tmp_path / "wandb" / "debug.log").symlink_to(
        "run-identity/logs/debug.log"
    )
    # Broken and absolute targets are observational link text only.  Mutating an
    # external target must not affect the authenticated scientific tree.
    (tmp_path / "wandb" / "debug-internal.log").symlink_to(
        "run-identity/logs/does-not-exist.log"
    )
    external = tmp_path.parent / f"{tmp_path.name}-external.log"
    external.write_text("before\n")
    (logs / "debug-core.log").symlink_to(external)
    parsed = report.parse_event_files(tmp_path, sampler)
    assert parsed["scalars"]["train/loss_total"] == {50: 2.0}
    external.write_text("after with different bytes\n")
    parsed = report.parse_event_files(tmp_path, sampler)
    assert parsed["scalars"]["train/loss_total"] == {50: 2.0}
    external.unlink()

    (tmp_path / "outside-link").symlink_to("wandb/run-identity")
    with pytest.raises(report.ReportError, match="contains symlink: outside-link"):
        report.parse_event_files(tmp_path, sampler)


def test_wandb_symlink_policy_rejects_root_swap_special_and_unreadable(
    report, tmp_path, monkeypatch
):
    root_link_case = tmp_path / "root-link-case"
    root_link_case.mkdir()
    (root_link_case / "target").mkdir()
    (root_link_case / "wandb").symlink_to("target")
    with pytest.raises(report.ReportError, match="contains symlink: wandb"):
        report._secure_tree_rows(
            root_link_case,
            "root-link case",
            hash_files=True,
            allow_wandb_symlink_leaves=True,
        )

    special_case = tmp_path / "special-case"
    wandb = special_case / "wandb"
    wandb.mkdir(parents=True)
    fifo = wandb / "hidden.fifo"
    os.mkfifo(fifo)
    with pytest.raises(report.ReportError, match="contains special file"):
        report._secure_tree_rows(
            special_case,
            "special case",
            hash_files=True,
            allow_wandb_symlink_leaves=True,
        )
    fifo.unlink()
    closed = wandb / "closed"
    closed.mkdir()
    closed.chmod(0)
    try:
        with pytest.raises(report.ReportError, match="directory is not traversable"):
            report._secure_tree_rows(
                special_case,
                "unreadable case",
                hash_files=True,
                allow_wandb_symlink_leaves=True,
            )
    finally:
        closed.chmod(0o700)

    swap_case = tmp_path / "swap-case"
    swap_wandb = swap_case / "wandb"
    swap_wandb.mkdir(parents=True)
    link = swap_wandb / "latest-run"
    link.symlink_to("first")
    real_readlink = report.os.readlink
    swapped = False

    def swapping_readlink(path, *, dir_fd=None):
        nonlocal swapped
        value = real_readlink(path, dir_fd=dir_fd)
        if path == b"" and not swapped:
            swapped = True
            link.unlink()
            link.symlink_to("second")
        return value

    monkeypatch.setattr(report.os, "readlink", swapping_readlink)
    with pytest.raises(report.ReportError, match="symlink changed"):
        report._secure_tree_rows(
            swap_case,
            "swap case",
            hash_files=True,
            allow_wandb_symlink_leaves=True,
        )


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


def test_report_publication_is_atomic_and_idempotent(report, tmp_path, monkeypatch):
    monkeypatch.setattr(
        report,
        "_validated_report_publication_prerequisite",
        lambda *_args, **_kwargs: {},
    )
    bundle = {"schema_version": 1, "cells": []}
    decision = {"status": "rejected", "gate_sha256": "a" * 64}
    provenance = {"schema_version": 1}
    first = report.publish_report(tmp_path, "b" * 64, bundle, decision, provenance)
    second = report.publish_report(tmp_path, "b" * 64, bundle, decision, provenance)
    assert first == second
    assert (tmp_path / "report" / "REPORT_COMMIT.json").is_file()
    assert not list(tmp_path.glob(".report.tmp.*"))


def test_report_failure_before_rename_publishes_nothing(report, tmp_path, monkeypatch):
    monkeypatch.setattr(
        report,
        "_validated_report_publication_prerequisite",
        lambda *_args, **_kwargs: {},
    )
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


def test_report_publication_requires_exact_successful_canary_prerequisite(
    report, submit, tmp_path
):
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    protocol = (PACKAGE / "protocol.sha256").read_text(encoding="ascii").strip()
    prerequisite = report._validated_production_authorization_prerequisite(
        manifest, protocol
    )
    assert prerequisite == submit._validated_production_authorization_prerequisite(
        manifest,
        allow_missing=False,
        package_protocol_sha256=protocol,
    )

    def sealed_submission(name, *, include_prerequisite):
        submission = tmp_path / name
        snapshot = submission / "source-snapshot/repo"
        package = snapshot / report.PACKAGE_RELATIVE
        package.mkdir(parents=True)
        inventory = {}
        for filename in (
            "manifest.json",
            "protocol.sha256",
            "canary2_acceptance_provenance.json",
        ):
            source = PACKAGE / filename
            target = package / filename
            target.write_bytes(source.read_bytes())
            target.chmod(0o444)
            inventory[(report.PACKAGE_RELATIVE / filename).as_posix()] = (
                report.file_sha256(target)
            )
        contract = {
            "schema_version": 1,
            "status": "sealed_for_submission",
            "campaign_id": report.CAMPAIGN_ID,
            "submission_root": str(submission),
            "snapshot_root": str(snapshot),
            "snapshot_inventory": inventory,
            "snapshot_inventory_sha256": report.stable_hash(inventory),
            "manifest_sha256": report.stable_hash(manifest),
            "package_protocol_sha256": protocol,
        }
        if include_prerequisite:
            contract["production_authorization_prerequisite"] = prerequisite
        submission_sha256 = report.seal_json(
            submission / "SUBMISSION_CONTRACT.json", contract
        )
        return submission, submission_sha256

    submission, submission_sha256 = sealed_submission(
        "authorized", include_prerequisite=True
    )
    provenance = {
        "schema_version": 1,
        "campaign_id": report.CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "production_authorization_prerequisite": prerequisite,
        "production_authorization_prerequisite_sha256": report.stable_hash(
            prerequisite
        ),
    }
    decision_body = {"status": "rejected"}
    decision = {
        **decision_body,
        "gate_sha256": report.stable_hash(decision_body),
    }
    commit = report.publish_report(
        submission,
        submission_sha256,
        {"schema_version": 1},
        decision,
        provenance,
    )
    assert commit["status"] == "rejected"

    blocked, blocked_sha256 = sealed_submission(
        "blocked", include_prerequisite=True
    )
    (blocked / "journal").mkdir()
    report.seal_json(
        blocked / "journal/PREREQUISITE_MISSING.json",
        {"schema_version": 1, "status": "cleanup_only"},
    )
    blocked_provenance = {
        **provenance,
        "submission_sha256": blocked_sha256,
    }
    with pytest.raises(report.ReportError, match="cancelled/ambiguous"):
        report.publish_report(
            blocked,
            blocked_sha256,
            {"schema_version": 1},
            decision,
            blocked_provenance,
        )
    assert not os.path.lexists(blocked / "report")
    assert not list(blocked.glob(".report.tmp.*"))

    missing, missing_sha256 = sealed_submission(
        "missing", include_prerequisite=False
    )
    with pytest.raises(report.ReportError, match="contract projection"):
        report.publish_report(
            missing,
            missing_sha256,
            {"schema_version": 1},
            decision,
            {
                "schema_version": 1,
                "campaign_id": report.CAMPAIGN_ID,
                "submission_sha256": missing_sha256,
            },
        )
    assert not os.path.lexists(missing / "report")


def test_report_rejects_broken_cancellation_latch(report, tmp_path):
    snapshot = tmp_path / "snapshot"
    submission = tmp_path / "submission"
    snapshot.mkdir()
    submission.mkdir()
    (submission / "CANCEL_REQUESTED.json").symlink_to(tmp_path / "missing-latch-target")
    with pytest.raises(report.ReportError, match="cancelled/ambiguous"):
        report.assemble_report(snapshot, submission, "a" * 64)


@pytest.mark.parametrize("status", ["accepted_engineering_pilot", "rejected"])
def test_report_and_cancel_terminal_states_are_mutually_exclusive(
    report, cancel, tmp_path, status, monkeypatch
):
    monkeypatch.setattr(
        report,
        "_validated_report_publication_prerequisite",
        lambda *_args, **_kwargs: {},
    )
    submission = tmp_path / "submission"
    submission.mkdir()
    body = {"status": status, "reason": "fixture"}
    decision = {**body, "gate_sha256": report.stable_hash(body)}
    report.publish_report(
        submission,
        "a" * 64,
        {"schema_version": 1},
        decision,
        {"schema_version": 1},
    )
    receipt = {
        "campaign_id": cancel.CAMPAIGN_ID,
        "submission_sha256": "a" * 64,
    }
    with cancel._ReportCancelLock(submission):
        commit = cancel._validated_published_report(submission, receipt)
        assert commit is not None and commit["status"] == status
        assert not os.path.lexists(submission / "CANCEL_REQUESTED.json")
    other = tmp_path / "cancel-first"
    other.mkdir()
    cancel.seal_json(other / "CANCEL_REQUESTED.json", {"status": "cancel_requested"})
    with pytest.raises(report.ReportError, match="cancelled/ambiguous"):
        report.publish_report(other, "a" * 64, {}, decision, {})
    assert not os.path.lexists(other / "report")


def test_cancel_rejects_symlink_or_forged_report_terminal(
    report, cancel, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        report,
        "_validated_report_publication_prerequisite",
        lambda *_args, **_kwargs: {},
    )
    receipt = {"campaign_id": cancel.CAMPAIGN_ID, "submission_sha256": "a" * 64}
    symlinked = tmp_path / "symlinked"
    symlinked.mkdir()
    (symlinked / "report").symlink_to(tmp_path / "missing-report")
    with pytest.raises(cancel.CancellationError, match="symlink"):
        cancel._validated_published_report(symlinked, receipt)

    forged = tmp_path / "forged"
    forged.mkdir()
    body = {"status": "rejected"}
    decision = {**body, "gate_sha256": report.stable_hash(body)}
    report.publish_report(forged, "a" * 64, {}, decision, {})
    commit_path = forged / "report" / "REPORT_COMMIT.json"
    commit = json.loads(commit_path.read_text())
    commit["status"] = "accepted"
    (forged / "report").chmod(0o755)
    commit_path.chmod(0o644)
    commit_path.write_text(json.dumps(commit, sort_keys=True, indent=2) + "\n")
    commit_path.chmod(0o444)
    (forged / "report").chmod(0o555)
    with pytest.raises(cancel.CancellationError, match="commit differs"):
        cancel._validated_published_report(forged, receipt)


def _cancellable_submission_root(tmp_path, submit, cancel):
    submission = tmp_path / "outputs" / "launch8" / "state" / "submission"
    submission.mkdir(parents=True)
    lock = submit._transaction_lock_path(submission)
    lock.touch(mode=0o600)
    lock.chmod(0o600)
    assert cancel._transaction_lock_path(submission) == lock
    return submission


def _cancel_contract(snapshot, submit, fallback):
    return {
        "snapshot_root": str(snapshot),
        "scheduler_fallback_config": fallback,
        "scheduler_preclaim": {
            "scheduler_control_plane": scheduler_observation(submit)
        },
    }


def _cancel_squeue_stdout(submit, receipt, active_ids):
    token = str(receipt["submission_sha256"])[:16]
    names = {
        str(receipt["wave0_array_job_id"]): f"exp23-launch8-{token}-wave0",
        str(receipt["wave1_array_job_id"]): f"exp23-launch8-{token}-wave1",
        str(receipt["report_job_id"]): f"exp23-launch8-{token}-report",
    }
    user = submit.pwd.getpwuid(os.getuid()).pw_name
    comment = f"treewm-exp23:{receipt['submission_sha256']}"
    return "".join(
        f"{job_id}|{names[job_id]}|{user}|RUNNING|{comment}\n"
        for job_id in active_ids
    )


def _install_explicit_cancel_fixture(
    tmp_path, submit, cancel, monkeypatch, *, token="f", job_ids=("300", "301", "302")
):
    submission = _cancellable_submission_root(tmp_path, submit, cancel)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    scancel = tmp_path / "scancel"
    squeue = tmp_path / "squeue"
    for executable in (scancel, squeue):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    receipt = {
        "campaign_id": cancel.CAMPAIGN_ID,
        "submission_sha256": token * 64,
        "submission_authorization_sha256": "e" * 64,
        "wave0_array_job_id": job_ids[0],
        "wave1_array_job_id": job_ids[1],
        "report_job_id": job_ids[2],
    }
    fallback = scheduler_fallback(submit)
    contract = _cancel_contract(snapshot, submit, fallback)
    manifest = {
        "execution": {
            "scancel": str(scancel),
            "squeue": str(squeue),
            "scheduler_control_plane": scheduler_contract(),
        }
    }
    state = {
        "active": list(job_ids),
        "commands": [],
        "invocations": [],
        "scancel_returncode": 0,
        "scancel_stdout": "cancelled\n",
        "scancel_stderr": "",
    }
    monkeypatch.setattr(
        cancel, "validate_receipt", lambda _root: (receipt, contract, manifest)
    )
    monkeypatch.setattr(
        cancel,
        "scheduler_fallback_config",
        lambda *_args: (fallback, scheduler_config_bytes()),
    )
    monkeypatch.setattr(
        cancel,
        "scheduler_control_plane_observation",
        lambda *_args: scheduler_observation(submit),
    )
    monkeypatch.setattr(cancel.time, "sleep", lambda _seconds: None)

    def run(command, **_kwargs):
        state["commands"].append(list(command))
        state["invocations"].append(
            {
                "command": list(command),
                "environment": dict(_kwargs["environment"]),
                "inherited_fds": tuple(_kwargs["inherited_fds"]),
            }
        )
        if command[0] == str(squeue):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_cancel_squeue_stdout(submit, receipt, state["active"]),
                stderr="",
            )
        return subprocess.CompletedProcess(
            command,
            state["scancel_returncode"],
            stdout=state["scancel_stdout"],
            stderr=state["scancel_stderr"],
        )

    monkeypatch.setattr(cancel, "_run_scheduler_client_with_lock_supervisor", run)
    return submission, receipt, contract, manifest, fallback, state


def test_committed_receipt_cancel_waits_for_submit_release_linearization(
    submit, cancel, tmp_path
):
    submission = _cancellable_submission_root(tmp_path, submit, cancel)
    owner_ready = threading.Event()
    allow_release = threading.Event()
    cancel_acquired = threading.Event()
    failures = []

    def submit_owner():
        try:
            with submit._TransactionLock(submission):
                (submission / "RECEIPT_VISIBLE").touch()
                owner_ready.set()
                assert allow_release.wait(5)
                (submission / "WAVE0_RELEASED").touch()
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    def cancel_owner():
        try:
            with cancel._CancellationTransactionLock(submission):
                cancel_acquired.set()
                assert (submission / "RECEIPT_VISIBLE").is_file()
                assert (submission / "WAVE0_RELEASED").is_file()
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    submitter = threading.Thread(target=submit_owner)
    canceller = threading.Thread(target=cancel_owner)
    submitter.start()
    assert owner_ready.wait(5)
    canceller.start()
    assert not cancel_acquired.wait(0.1)
    allow_release.set()
    submitter.join(5)
    canceller.join(5)
    assert not submitter.is_alive() and not canceller.is_alive() and not failures


@pytest.mark.parametrize(
    "active_ids,expected_status,expected_second_call",
    [
        ([], "cancel_reconciled_all_exact_jobs_terminal_or_absent", False),
        (["100"], "cancel_reconciled_active_exact_jobs_signalled", True),
    ],
)
def test_cancel_hard_kill_after_accepted_call_reconciles_before_retry(
    submit,
    cancel,
    tmp_path,
    monkeypatch,
    active_ids,
    expected_status,
    expected_second_call,
):
    submission = _cancellable_submission_root(tmp_path, submit, cancel)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    scancel = tmp_path / "scancel"
    scancel.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    scancel.chmod(0o755)
    squeue = tmp_path / "squeue"
    squeue.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    squeue.chmod(0o755)
    receipt = {
        "campaign_id": cancel.CAMPAIGN_ID,
        "submission_sha256": "c" * 64,
        "submission_authorization_sha256": "e" * 64,
        "wave0_array_job_id": "100",
        "wave1_array_job_id": "101",
        "report_job_id": "102",
    }
    manifest = {
        "execution": {
            "scancel": str(scancel),
            "squeue": str(squeue),
            "scheduler_control_plane": scheduler_contract(),
        }
    }
    fallback = scheduler_fallback(submit)
    contract = _cancel_contract(snapshot, submit, fallback)
    monkeypatch.setattr(cancel, "validate_receipt", lambda _root: (receipt, contract, manifest))
    monkeypatch.setattr(
        cancel,
        "scheduler_fallback_config",
        lambda *_args: (fallback, scheduler_config_bytes()),
    )
    monkeypatch.setattr(
        cancel,
        "scheduler_control_plane_observation",
        lambda *_args: scheduler_observation(submit),
    )
    calls = []

    def run(command, **_kwargs):
        calls.append(list(command))
        stdout = (
            _cancel_squeue_stdout(submit, receipt, ["100", "101", "102"])
            if command[0] == str(squeue)
            else "accepted\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(cancel, "_run_scheduler_client_with_lock_supervisor", run)
    real_seal = cancel.seal_json
    killed = False

    def kill_before_result(path, value):
        nonlocal killed
        if path.name == "CANCEL_RESULT.json" and not killed:
            killed = True
            raise OSError("injected SIGKILL boundary after accepted scancel")
        return real_seal(path, value)

    monkeypatch.setattr(cancel, "seal_json", kill_before_result)
    with pytest.raises(OSError, match="SIGKILL boundary"):
        cancel.explicit_cancel(submission)
    assert calls == [
        *[
            [
                str(squeue),
                "--noheader",
                "--jobs=100,101,102",
                "--format=%A|%j|%u|%T|%k",
            ]
            for _ in range(3)
        ],
        [str(scancel), "100", "101", "102"],
    ]
    reconciliation_calls = 0

    def reconcile(**kwargs):
        nonlocal reconciliation_calls
        reconciliation_calls += 1
        selected = list(active_ids) if reconciliation_calls <= 3 else []
        return selected, {
            "kind": "intent_without_result_exact_id_reconciliation",
            "command": [
                str(squeue),
                "--noheader",
                "--jobs=100,101,102",
                "--format=%A|%j|%u|%T|%k",
            ],
            "returncode": 0,
            "stdout": _cancel_squeue_stdout(submit, receipt, selected),
            "stderr": "",
            "active_job_ids": selected,
            "scheduler_mode": "canonical_root_admin_config",
            "scheduler_control_plane_before": scheduler_observation(submit),
            "scheduler_control_plane_after": scheduler_observation(submit),
            "canonical_boundary_error": None,
            "reconciled_call_records": [
                dict(item) for item in kwargs.get("reconciled_call_records", ())
            ],
        }

    monkeypatch.setattr(
        cancel,
        "_reconcile_exact_cancel_ids",
        reconcile,
    )
    result = cancel.explicit_cancel(submission)
    assert result["status"] == expected_status
    assert result["reconciled_active_job_ids"] == active_ids
    assert result["executed_cancel_command"] == (
        [str(scancel), *active_ids] if active_ids else None
    )
    assert len(calls) == 4 + int(expected_second_call)
    before = list(calls)
    reused = cancel.explicit_cancel(submission)
    assert reused["reused_durable_cancel_result"] is True
    assert calls == before
    assert reconciliation_calls == 6


def test_cancel_intent_reconciliation_queries_only_exact_three_ids(
    submit, cancel, tmp_path, monkeypatch
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    squeue = tmp_path / "squeue"
    squeue.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    squeue.chmod(0o755)
    receipt = {
        "submission_sha256": "c" * 64,
        "wave0_array_job_id": "100",
        "wave1_array_job_id": "101",
        "report_job_id": "102",
    }
    manifest = {
        "execution": {
            "squeue": str(squeue),
            "scheduler_control_plane": scheduler_contract(),
        }
    }
    monkeypatch.setattr(
        cancel,
        "scheduler_control_plane_observation",
        lambda *_args: scheduler_observation(submit),
    )
    user = submit.pwd.getpwuid(os.getuid()).pw_name
    comment = "treewm-exp23:" + "c" * 64

    def run(command, **kwargs):
        assert command == [
            str(squeue),
            "--noheader",
            "--jobs=100,101,102",
            "--format=%A|%j|%u|%T|%k",
        ]
        assert kwargs["inherited_fds"] == ()
        stdout = (
            f"100|exp23-launch8-{'c' * 16}-wave0|{user}|RUNNING|{comment}\n"
            f"102|exp23-launch8-{'c' * 16}-report|{user}|PENDING|{comment}\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(cancel, "_run_scheduler_client_with_lock_supervisor", run)
    fallback = scheduler_fallback(submit)
    lease_path = tmp_path / "cancel-lease.lock"
    lease_path.touch(mode=0o600)
    lease_descriptor = os.open(lease_path, os.O_RDWR)
    try:
        active, evidence = cancel._reconcile_exact_cancel_ids(
            snapshot_root=snapshot,
            receipt=receipt,
            manifest=manifest,
            fallback_binding=fallback,
            fallback_payload=scheduler_config_bytes(),
            expected_control_plane=scheduler_observation(submit),
            transaction_lock_descriptor=lease_descriptor,
        )
    finally:
        os.close(lease_descriptor)
    assert active == ["100", "102"]
    assert evidence["active_job_ids"] == active


def test_durable_cancel_precedes_every_missing_wave0_release_path(submit):
    live = inspect.getsource(submit._submit_campaign_impl)
    receipt = live.index('"SUBMISSION_RECEIPT.json"')
    cancel_check = live.index("_validated_committed_cancel_latch", receipt)
    release = live.index("_release_authorized_wave0", cancel_check)
    assert receipt < cancel_check < release

    recovery = inspect.getsource(submit._recover_transaction_locked)
    finish = recovery.index("def finish_committed_receipt")
    cancel_check = recovery.index("_validated_committed_cancel_latch", finish)
    cancel_return = recovery.index('"forbidden_by_durable_cancel"', cancel_check)
    ensure_release = recovery.index("_ensure_authorized_wave0_released", cancel_return)
    second_check = recovery.index("cancel_latch is None", cancel_return)
    assert cancel_check < cancel_return < second_check < ensure_release


def test_cancel_latch_precedes_scheduler_and_result_is_durable(
    submit, cancel, tmp_path, monkeypatch
):
    submission = _cancellable_submission_root(tmp_path, submit, cancel)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    scancel = tmp_path / "scancel"
    scancel.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    scancel.chmod(0o755)
    squeue = tmp_path / "squeue"
    squeue.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    squeue.chmod(0o755)
    receipt = {
            "campaign_id": "treewm-executable-prefix-repair-pilot-v1-launch8",
        "submission_sha256": "c" * 64,
        "submission_authorization_sha256": "e" * 64,
        "wave0_array_job_id": "100",
        "wave1_array_job_id": "101",
        "report_job_id": "102",
        "snapshot_root": str(snapshot),
    }
    control_plane = scheduler_contract()
    manifest = {
        "execution": {
            "scancel": str(scancel),
            "squeue": str(squeue),
            "scheduler_control_plane": control_plane,
        }
    }
    contract = _cancel_contract(snapshot, submit, scheduler_fallback(submit))
    monkeypatch.setattr(cancel, "validate_receipt", lambda _root: (receipt, contract, manifest))
    monkeypatch.setattr(
        cancel,
        "scheduler_fallback_config",
        lambda *_args: (scheduler_fallback(submit), scheduler_config_bytes()),
    )
    monkeypatch.setattr(
        cancel,
        "scheduler_control_plane_observation",
        lambda *_args: scheduler_observation(submit),
    )
    monkeypatch.setenv("SLURM_CONF", "/tmp/hostile-slurm.conf")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/hostile-libraries")

    def run(command, **kwargs):
        assert (submission / "CANCEL_REQUESTED.json").is_file()
        if command[0] == str(scancel):
            assert list((submission / "cancellation").glob("CANCEL_CALL.*.json"))
        assert kwargs["environment"] == {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
                "SLURM_CONF": control_plane["slurm_conf"],
            }
        assert kwargs["inherited_fds"] == ()
        assert isinstance(kwargs["transaction_lock_descriptor"], int)
        stdout = (
            _cancel_squeue_stdout(submit, receipt, ["100", "101", "102"])
            if command[0] == str(squeue)
            else "ok\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(cancel, "_run_scheduler_client_with_lock_supervisor", run)
    result = cancel.explicit_cancel(submission)
    assert result["job_ids"] == ["100", "101", "102"]
    result_files = list((submission / "cancellation").glob("CANCEL_RESULT.json"))
    assert len(result_files) == 1
    sealed = json.loads(result_files[0].read_text())
    assert sealed["returncode"] == 0
    assert sealed["executed_cancel_command"][-3:] == ["100", "101", "102"]
    assert (submission / "cancellation" / "CANCEL_COMMIT.json").is_file()
    assert cancel.scheduler_environment(control_plane) == submit._scheduler_environment(
        control_plane
    )


@pytest.mark.parametrize("bad_mode,bad_schema", [(0o644, 1), (0o444, True)])
def test_cancel_latch_rejects_writable_or_bool_schema_existing_artifact(
    cancel, tmp_path, bad_mode, bad_schema
):
    submission = tmp_path / f"case-{bad_mode}-{bad_schema}"
    submission.mkdir()
    receipt = {
        "campaign_id": cancel.CAMPAIGN_ID,
        "submission_sha256": "c" * 64,
        "wave0_array_job_id": "100",
        "wave1_array_job_id": "101",
        "report_job_id": "102",
    }
    value = {
        "schema_version": bad_schema,
        "status": "cancel_requested",
        "campaign_id": receipt["campaign_id"],
        "submission_sha256": receipt["submission_sha256"],
        "wave0_array_job_id": "100",
        "wave1_array_job_id": "101",
        "report_job_id": "102",
    }
    path = submission / "CANCEL_REQUESTED.json"
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    path.chmod(bad_mode)
    with pytest.raises(cancel.CancellationError, match="existing cancellation latch differs"):
        cancel.seal_latch(submission, receipt)


def test_explicit_cancel_uses_retained_original_config_after_canonical_drift(
    submit, cancel, tmp_path, monkeypatch
):
    submission = _cancellable_submission_root(tmp_path, submit, cancel)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    scancel = tmp_path / "scancel"
    scancel.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    scancel.chmod(0o755)
    squeue = tmp_path / "squeue"
    squeue.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    squeue.chmod(0o755)
    receipt = {
            "campaign_id": "treewm-executable-prefix-repair-pilot-v1-launch8",
        "submission_sha256": "c" * 64,
        "submission_authorization_sha256": "e" * 64,
        "wave0_array_job_id": "100",
        "wave1_array_job_id": "101",
        "report_job_id": "102",
        "snapshot_root": str(snapshot),
    }
    manifest = {
        "execution": {
            "scancel": str(scancel),
            "squeue": str(squeue),
            "scheduler_control_plane": scheduler_contract(),
        }
    }
    fallback = scheduler_fallback(submit)
    monkeypatch.setattr(
        cancel,
        "validate_receipt",
        lambda _root: (
            receipt,
            _cancel_contract(snapshot, submit, fallback),
            manifest,
        ),
    )
    monkeypatch.setattr(
        cancel,
        "scheduler_fallback_config",
        lambda *_args: (fallback, scheduler_config_bytes()),
    )
    monkeypatch.setattr(
        cancel,
        "scheduler_control_plane_observation",
        lambda *_args: (_ for _ in ()).throw(cancel.CancellationError("critical drift")),
    )

    def run(command, **kwargs):
        if command[0] == str(scancel):
            assert list(command)[-3:] == ["100", "101", "102"]
        assert kwargs["environment"]["SLURM_CONF"].startswith("/proc/self/fd/")
        assert len(kwargs["inherited_fds"]) == 1
        descriptor = kwargs["inherited_fds"][0]
        assert os.pread(descriptor, len(scheduler_config_bytes()), 0) == scheduler_config_bytes()
        stdout = (
            _cancel_squeue_stdout(submit, receipt, ["100", "101", "102"])
            if command[0] == str(squeue)
            else "ok\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(cancel, "_run_scheduler_client_with_lock_supervisor", run)
    result = cancel.explicit_cancel(submission)
    assert result["scheduler_mode"] == "sealed_original_config_fallback"
    assert "critical drift" in result["canonical_boundary_error"]
    assert result["job_ids"] == ["100", "101", "102"]


def test_explicit_cancel_retries_exact_ids_when_canonical_failure_closes_on_drift(
    submit, cancel, tmp_path, monkeypatch
):
    submission = _cancellable_submission_root(tmp_path, submit, cancel)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    scancel = tmp_path / "scancel"
    scancel.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    scancel.chmod(0o755)
    squeue = tmp_path / "squeue"
    squeue.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    squeue.chmod(0o755)
    receipt = {
            "campaign_id": "treewm-executable-prefix-repair-pilot-v1-launch8",
        "submission_sha256": "d" * 64,
        "submission_authorization_sha256": "e" * 64,
        "wave0_array_job_id": "200",
        "wave1_array_job_id": "201",
        "report_job_id": "202",
        "snapshot_root": str(snapshot),
    }
    manifest = {
        "execution": {
            "scancel": str(scancel),
            "squeue": str(squeue),
            "scheduler_control_plane": scheduler_contract(),
        }
    }
    fallback = scheduler_fallback(submit)
    monkeypatch.setattr(
        cancel,
        "validate_receipt",
        lambda _root: (
            receipt,
            _cancel_contract(snapshot, submit, fallback),
            manifest,
        ),
    )
    monkeypatch.setattr(
        cancel,
        "scheduler_fallback_config",
        lambda *_args: (fallback, scheduler_config_bytes()),
    )
    observation_count = 0

    def observe(*_args):
        nonlocal observation_count
        observation_count += 1
        if observation_count >= 8:
            raise cancel.CancellationError("critical drift after canonical scancel")
        return scheduler_observation(submit)

    calls = 0

    def run(command, **kwargs):
        nonlocal calls
        calls += 1
        if command[0] == str(squeue):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_cancel_squeue_stdout(
                    submit, receipt, ["200", "201", "202"]
                ),
                stderr="",
            )
        assert list(command)[-3:] == ["200", "201", "202"]
        if kwargs["environment"]["SLURM_CONF"] == scheduler_contract()["slurm_conf"]:
            assert kwargs["environment"]["SLURM_CONF"] == scheduler_contract()["slurm_conf"]
            assert kwargs["inherited_fds"] == ()
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="config raced")
        assert kwargs["environment"]["SLURM_CONF"].startswith("/proc/self/fd/")
        assert len(kwargs["inherited_fds"]) == 1
        return subprocess.CompletedProcess(command, 0, stdout="cancelled\n", stderr="")

    monkeypatch.setattr(cancel, "scheduler_control_plane_observation", observe)
    monkeypatch.setattr(cancel, "_run_scheduler_client_with_lock_supervisor", run)
    result = cancel.explicit_cancel(submission)
    assert result["scheduler_calls"] == 8
    assert result["scheduler_mode"] == "sealed_original_config_fallback_after_unknown_response"
    assert result["returncode"] == 0
    assert len(result["scheduler_attempts"]) == 8
    assert len(list((submission / "cancellation").glob("CANCEL_CALL.*.json"))) == 2


@pytest.mark.parametrize("active_ids", [[], ["300"]])
def test_explicit_cancel_first_mutation_authority_is_fresh_exact_census_only(
    submit, cancel, tmp_path, monkeypatch, active_ids
):
    submission, _receipt, _contract, manifest, _fallback, state = (
        _install_explicit_cancel_fixture(tmp_path, submit, cancel, monkeypatch)
    )
    state["active"] = list(active_ids)
    result = cancel.explicit_cancel(submission)
    scancel_commands = [
        row for row in state["commands"] if Path(row[0]).name == "scancel"
    ]
    assert scancel_commands == (
        [[manifest["execution"]["scancel"], *active_ids]] if active_ids else []
    )
    assert result["reconciled_active_job_ids"] == active_ids
    assert result["executed_cancel_command"] == (
        scancel_commands[0] if active_ids else None
    )
    assert "301" not in sum(scancel_commands, [])
    assert "302" not in sum(scancel_commands, [])


def test_explicit_cancel_preclaim_drift_never_uses_canonical_mutation(
    submit, cancel, tmp_path, monkeypatch
):
    submission, _receipt, _contract, _manifest, _fallback, state = (
        _install_explicit_cancel_fixture(tmp_path, submit, cancel, monkeypatch)
    )
    changed = json.loads(json.dumps(scheduler_observation(submit)))
    changed["cli_filter_policy"]["tree_sha256"] = "b" * 64
    monkeypatch.setattr(
        cancel, "scheduler_control_plane_observation", lambda *_args: changed
    )
    result = cancel.explicit_cancel(submission)
    scancel_invocations = [
        row for row in state["invocations"] if Path(row["command"][0]).name == "scancel"
    ]
    assert len(scancel_invocations) == 1
    assert scancel_invocations[0]["environment"]["SLURM_CONF"].startswith(
        "/proc/self/fd/"
    )
    assert len(scancel_invocations[0]["inherited_fds"]) == 1
    assert result["scheduler_mode"] == "sealed_original_config_fallback"
    assert "committed cancellation preclaim" in result["canonical_boundary_error"]
    result_path = submission / "cancellation" / "CANCEL_RESULT.json"
    value = json.loads(result_path.read_text(encoding="utf-8"))
    value["scheduler_attempts"][0]["canonical_boundary_error"] = ""
    result_path.chmod(0o644)
    result_path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    result_path.chmod(0o444)
    before = len(state["commands"])
    with pytest.raises(cancel.CancellationError, match="fallback binding differs"):
        cancel.explicit_cancel(submission)
    assert len(state["commands"]) == before


@pytest.mark.parametrize(
    "mutator,error",
    [
        (
            lambda value: value.update(
                {"scheduler_attempts": [{}], "scheduler_calls": 1}
            ),
            "cancellation call attempt 0 differs",
        ),
        (
            lambda value: value.update({"scheduler_calls": 1.0}),
            "durable cancellation result differs",
        ),
        (
            lambda value: value.update({"stdout": "forged terminal output\n"}),
            "terminal call summary differs",
        ),
    ],
)
def test_explicit_cancel_rejects_forged_prior_result_before_fresh_scheduler_call(
    submit, cancel, tmp_path, monkeypatch, mutator, error
):
    submission, _receipt, _contract, _manifest, _fallback, state = (
        _install_explicit_cancel_fixture(tmp_path, submit, cancel, monkeypatch)
    )
    cancel.explicit_cancel(submission)
    before = len(state["commands"])
    result_path = submission / "cancellation" / "CANCEL_RESULT.json"
    value = json.loads(result_path.read_text(encoding="utf-8"))
    mutator(value)
    result_path.chmod(0o644)
    result_path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    result_path.chmod(0o444)
    with pytest.raises(cancel.CancellationError, match=error):
        cancel.explicit_cancel(submission)
    assert len(state["commands"]) == before


def test_explicit_cancel_rejects_rehashed_call_intent_mutation(
    submit, cancel, tmp_path, monkeypatch
):
    submission, _receipt, _contract, _manifest, _fallback, state = (
        _install_explicit_cancel_fixture(tmp_path, submit, cancel, monkeypatch)
    )
    cancel.explicit_cancel(submission)
    before = len(state["commands"])
    call_path = next((submission / "cancellation").glob("CANCEL_CALL.*.json"))
    value = json.loads(call_path.read_text(encoding="utf-8"))
    value["command"] = [value["command"][0], "301"]
    call_path.chmod(0o644)
    call_path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    call_path.chmod(0o444)
    with pytest.raises(cancel.CancellationError, match="detached from its intent"):
        cancel.explicit_cancel(submission)
    assert len(state["commands"]) == before


@pytest.mark.parametrize("artifact", ["CANCEL_REQUESTED.json", "CANCEL_INTENT.json", "CALL"])
def test_explicit_cancel_binds_raw_intent_and_call_bytes(
    submit, cancel, tmp_path, monkeypatch, artifact
):
    submission, _receipt, _contract, _manifest, _fallback, state = (
        _install_explicit_cancel_fixture(tmp_path, submit, cancel, monkeypatch)
    )
    cancel.explicit_cancel(submission)
    evidence = submission / "cancellation"
    path = (
        next(evidence.glob("CANCEL_CALL.*.json"))
        if artifact == "CALL"
        else (
            submission / artifact
            if artifact == "CANCEL_REQUESTED.json"
            else evidence / artifact
        )
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    path.chmod(0o644)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    before = len(state["commands"])
    with pytest.raises(
        cancel.CancellationError,
        match=(
            "durable cancellation result differs"
            if artifact in {"CANCEL_REQUESTED.json", "CANCEL_INTENT.json"}
            else "detached from its intent"
        ),
    ):
        cancel.explicit_cancel(submission)
    assert len(state["commands"]) == before


@pytest.mark.parametrize("artifact", ["CANCEL_INTENT.json", "CANCEL_COMMIT.json"])
def test_explicit_cancel_rejects_boolean_schema_in_semantic_artifacts(
    submit, cancel, tmp_path, monkeypatch, artifact
):
    submission, receipt, _contract, manifest, _fallback, state = (
        _install_explicit_cancel_fixture(tmp_path, submit, cancel, monkeypatch)
    )
    evidence = submission / "cancellation"
    if artifact == "CANCEL_COMMIT.json":
        cancel.explicit_cancel(submission)
    else:
        evidence.mkdir(mode=0o700)
        cancel.seal_json(
            evidence / artifact,
            {
                "schema_version": True,
                "status": "exact_cancel_intent",
                "campaign_id": receipt["campaign_id"],
                "submission_sha256": receipt["submission_sha256"],
                "submission_authorization_sha256": receipt[
                    "submission_authorization_sha256"
                ],
                "job_ids": ["300", "301", "302"],
                "command": [manifest["execution"]["scancel"], "300", "301", "302"],
            },
        )
    path = evidence / artifact
    value = json.loads(path.read_text(encoding="utf-8"))
    value["schema_version"] = True
    path.chmod(0o644)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    path.chmod(0o444)
    before = len(state["commands"])
    with pytest.raises(cancel.CancellationError, match="cancellation (intent|commit) differs"):
        cancel.explicit_cancel(submission)
    assert len(state["commands"]) == before


def test_explicit_cancel_rejects_coherently_rehashed_nested_boolean_control_plane(
    submit, cancel, tmp_path, monkeypatch
):
    submission, _receipt, _contract, _manifest, _fallback, state = (
        _install_explicit_cancel_fixture(tmp_path, submit, cancel, monkeypatch)
    )
    cancel.explicit_cancel(submission)
    evidence = submission / "cancellation"

    def rewrite(path, value):
        path.chmod(0o644)
        path.write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        path.chmod(0o444)

    call_path = next(evidence.glob("CANCEL_CALL.*.json"))
    call = json.loads(call_path.read_text(encoding="utf-8"))
    call["scheduler_control_plane"]["schema_version"] = True
    rewrite(call_path, call)

    result_path = evidence / "CANCEL_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    call_attempt = next(
        row for row in result["scheduler_attempts"] if row["kind"] == "exact_cancel_call"
    )
    call_attempt["call_intent_sha256"] = cancel.file_sha256(call_path)
    call_attempt["scheduler_control_plane_before"]["schema_version"] = True
    call_attempt["scheduler_control_plane_after"]["schema_version"] = True
    result["scheduler_control_plane"]["schema_version"] = True
    rewrite(result_path, result)

    commit_path = evidence / "CANCEL_COMMIT.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["result_sha256"] = cancel.file_sha256(result_path)
    rewrite(commit_path, commit)

    before = len(state["commands"])
    with pytest.raises(cancel.CancellationError, match="canonical binding differs"):
        cancel.explicit_cancel(submission)
    assert len(state["commands"]) == before


def test_explicit_cancel_rejects_coherently_rehashed_alternate_mutation_authority(
    submit, cancel, tmp_path, monkeypatch
):
    submission, _receipt, _contract, manifest, _fallback, state = (
        _install_explicit_cancel_fixture(tmp_path, submit, cancel, monkeypatch)
    )
    cancel.explicit_cancel(submission)
    evidence = submission / "cancellation"
    call_path = next(evidence.glob("CANCEL_CALL.*.json"))
    call = json.loads(call_path.read_text(encoding="utf-8"))
    call["command"] = [manifest["execution"]["scancel"], "301"]
    call_path.chmod(0o644)
    call_path.write_text(
        json.dumps(call, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    call_path.chmod(0o444)
    result_path = evidence / "CANCEL_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    call_attempt = next(
        row for row in result["scheduler_attempts"] if row["kind"] == "exact_cancel_call"
    )
    call_attempt["command"] = list(call["command"])
    call_attempt["call_intent_sha256"] = cancel.file_sha256(call_path)
    result["executed_cancel_command"] = list(call["command"])
    result["reconciled_active_job_ids"] = ["301"]
    result["terminal_or_absent_job_ids"] = ["300", "302"]
    result_path.chmod(0o644)
    result_path.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    result_path.chmod(0o444)
    commit_path = evidence / "CANCEL_COMMIT.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["result_sha256"] = cancel.file_sha256(result_path)
    commit_path.chmod(0o644)
    commit_path.write_text(
        json.dumps(commit, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    commit_path.chmod(0o444)
    before = len(state["commands"])
    with pytest.raises(cancel.CancellationError, match="settled exact census"):
        cancel.explicit_cancel(submission)
    assert len(state["commands"]) == before


def test_explicit_cancel_prior_reappearance_appends_then_converges(
    submit, cancel, tmp_path, monkeypatch
):
    submission, _receipt, _contract, _manifest, _fallback, state = (
        _install_explicit_cancel_fixture(tmp_path, submit, cancel, monkeypatch)
    )
    first = cancel.explicit_cancel(submission)
    immutable_sha = first["cancel_result_sha256"]
    state["active"] = ["300"]
    pending = cancel.explicit_cancel(submission)
    assert pending["residual_continuation_chain"][-1]["status"] == (
        "cancel_residual_signalled_pending_terminal"
    )
    assert pending["residual_continuation_chain"][-1]["active_job_ids"] == ["300"]
    state["active"] = []
    terminal = cancel.explicit_cancel(submission)
    assert [row["generation"] for row in terminal["residual_continuation_chain"]] == [0, 1]
    assert terminal["residual_continuation_chain"][-1]["status"] == (
        "cancel_residual_reconciled_terminal"
    )
    assert cancel.file_sha256(
        submission / "cancellation" / "CANCEL_RESULT.json"
    ) == immutable_sha
    before_files = sorted(
        path.name
        for path in (submission / "cancellation").glob("CANCEL_CONTINUATION.*.json")
    )
    stable = cancel.explicit_cancel(submission)
    assert stable["fresh_active_job_ids"] == []
    assert sorted(
        path.name
        for path in (submission / "cancellation").glob("CANCEL_CONTINUATION.*.json")
    ) == before_files


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.update({"generation": 0.0}),
        lambda value: value.update({"scheduler_calls": 7.0}),
        lambda value: value["pre_cancel_census_rounds"][0].update({"round": 0.0}),
        lambda value: value["cancel_call"].update({"stdout": 0}),
        lambda value: value["cancel_call"].update({"canonical_boundary_error": ""}),
    ],
)
def test_explicit_cancel_continuation_rejects_nonexact_scalars(
    submit, cancel, tmp_path, monkeypatch, mutator
):
    submission, _receipt, _contract, _manifest, _fallback, state = (
        _install_explicit_cancel_fixture(tmp_path, submit, cancel, monkeypatch)
    )
    cancel.explicit_cancel(submission)
    state["active"] = ["300"]
    cancel.explicit_cancel(submission)
    continuation = submission / "cancellation" / "CANCEL_CONTINUATION.0000.json"
    value = json.loads(continuation.read_text(encoding="utf-8"))
    mutator(value)
    continuation.chmod(0o644)
    continuation.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    continuation.chmod(0o444)
    before = len(state["commands"])
    with pytest.raises(cancel.CancellationError, match="cancel continuation 0"):
        cancel.explicit_cancel(submission)
    assert len(state["commands"]) == before


def test_explicit_cancel_continuation_rejects_call_reconciled_and_executed_twice(
    submit, cancel, tmp_path, monkeypatch
):
    submission, _receipt, _contract, _manifest, _fallback, state = (
        _install_explicit_cancel_fixture(tmp_path, submit, cancel, monkeypatch)
    )
    cancel.explicit_cancel(submission)
    state["active"] = ["300"]
    cancel.explicit_cancel(submission)
    continuation = submission / "cancellation" / "CANCEL_CONTINUATION.0000.json"
    value = json.loads(continuation.read_text(encoding="utf-8"))
    call = value["cancel_call"]
    reconciled = {
        "name": call["call_intent_name"],
        "sha256": call["call_intent_sha256"],
        "call_token": call["call_token"],
    }
    value["reconciled_call_records"] = [reconciled]
    value["pre_cancel_census_rounds"][0]["evidence"][
        "reconciled_call_records"
    ] = [reconciled]
    continuation.chmod(0o644)
    continuation.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    continuation.chmod(0o444)
    before = len(state["commands"])
    with pytest.raises(cancel.CancellationError, match="residual call differs"):
        cancel.explicit_cancel(submission)
    assert len(state["commands"]) == before


def test_explicit_cancel_consumes_orphan_residual_call_after_hard_kill(
    submit, cancel, tmp_path, monkeypatch
):
    submission, _receipt, _contract, _manifest, _fallback, state = (
        _install_explicit_cancel_fixture(tmp_path, submit, cancel, monkeypatch)
    )
    cancel.explicit_cancel(submission)
    state["active"] = ["300"]
    real_seal = cancel.seal_json

    def kill_before_continuation(path, value):
        if path.name == "CANCEL_CONTINUATION.0000.json":
            raise OSError("injected hard kill after residual scancel")
        return real_seal(path, value)

    monkeypatch.setattr(cancel, "seal_json", kill_before_continuation)
    with pytest.raises(OSError, match="hard kill"):
        cancel.explicit_cancel(submission)
    orphan_calls = sorted(
        path.name for path in (submission / "cancellation").glob("CANCEL_CALL.*.json")
    )
    assert len(orphan_calls) == 2
    monkeypatch.setattr(cancel, "seal_json", real_seal)
    state["active"] = []
    recovered = cancel.explicit_cancel(submission)
    generation = recovered["residual_continuation_chain"][-1]
    assert generation["cancel_call"] is None
    assert [row["name"] for row in generation["reconciled_call_records"]] == [
        orphan_calls[-1]
    ]
    assert len(
        [row for row in state["commands"] if Path(row[0]).name == "scancel"]
    ) == 2


def test_explicit_cancel_consumes_orphan_residual_call_before_client_return(
    submit, cancel, tmp_path, monkeypatch
):
    submission, _receipt, _contract, _manifest, _fallback, state = (
        _install_explicit_cancel_fixture(tmp_path, submit, cancel, monkeypatch)
    )
    cancel.explicit_cancel(submission)
    state["active"] = ["300"]
    real_run = cancel._run_scheduler_client_with_lock_supervisor
    killed = False

    def kill_residual(command, **kwargs):
        nonlocal killed
        if Path(command[0]).name == "scancel" and command[1:] == ["300"] and not killed:
            killed = True
            raise OSError("injected hard kill with residual CALL durable")
        return real_run(command, **kwargs)

    monkeypatch.setattr(cancel, "_run_scheduler_client_with_lock_supervisor", kill_residual)
    with pytest.raises(OSError, match="residual CALL durable"):
        cancel.explicit_cancel(submission)
    assert len(list((submission / "cancellation").glob("CANCEL_CALL.*.json"))) == 2
    monkeypatch.setattr(cancel, "_run_scheduler_client_with_lock_supervisor", real_run)
    state["active"] = []
    recovered = cancel.explicit_cancel(submission)
    assert recovered["residual_continuation_chain"][-1]["cancel_call"] is None
    assert recovered["residual_continuation_chain"][-1][
        "reconciled_call_records"
    ]


def test_explicit_cancel_nonzero_response_leaves_only_reconcilable_call_intent(
    submit, cancel, tmp_path, monkeypatch
):
    submission, _receipt, _contract, _manifest, _fallback, state = (
        _install_explicit_cancel_fixture(tmp_path, submit, cancel, monkeypatch)
    )
    state["scancel_returncode"] = 1
    state["scancel_stderr"] = "rejected\n"
    with pytest.raises(cancel.CancellationError, match="durable CALL intent"):
        cancel.explicit_cancel(submission)
    evidence = submission / "cancellation"
    assert list(evidence.glob("CANCEL_CALL.*.json"))
    assert not list(evidence.glob("CANCEL_FAILURE.*.json"))
    assert not (evidence / "CANCEL_RESULT.json").exists()
    state["active"] = []
    state["scancel_returncode"] = 0
    recovered = cancel.explicit_cancel(submission)
    assert recovered["status"] == "cancel_reconciled_all_exact_jobs_terminal_or_absent"
    assert recovered["executed_cancel_command"] is None


def test_explicit_cancel_rc0_with_unverifiable_postcondition_reconciles_not_replays(
    submit, cancel, tmp_path, monkeypatch
):
    submission, receipt, _contract, manifest, _fallback, state = (
        _install_explicit_cancel_fixture(tmp_path, submit, cancel, monkeypatch)
    )
    observation_count = 0

    def observe(*_args):
        nonlocal observation_count
        observation_count += 1
        if observation_count == 8:
            raise cancel.CancellationError("post-scancel control plane is unavailable")
        return scheduler_observation(submit)

    def run(command, **kwargs):
        state["commands"].append(list(command))
        state["invocations"].append(
            {
                "command": list(command),
                "environment": dict(kwargs["environment"]),
                "inherited_fds": tuple(kwargs["inherited_fds"]),
            }
        )
        if command[0] == manifest["execution"]["squeue"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_cancel_squeue_stdout(submit, receipt, state["active"]),
                stderr="",
            )
        state["active"] = []
        return subprocess.CompletedProcess(command, 0, stdout="accepted\n", stderr="")

    monkeypatch.setattr(cancel, "scheduler_control_plane_observation", observe)
    monkeypatch.setattr(cancel, "_run_scheduler_client_with_lock_supervisor", run)
    result = cancel.explicit_cancel(submission)
    scancel_commands = [
        row for row in state["commands"] if Path(row[0]).name == "scancel"
    ]
    assert scancel_commands == [
        [manifest["execution"]["scancel"], "300", "301", "302"]
    ]
    assert result["status"] == "cancel_reconciled_all_exact_jobs_terminal_or_absent"
    assert result["executed_cancel_command"] is None
    assert result["scheduler_calls"] == 7
    assert result["canonical_boundary_error"] is None
    assert result["scheduler_attempts"][3]["postcondition_error"]
    assert result["scheduler_attempts"][4]["reconciled_call_records"]
    result_path = submission / "cancellation" / "CANCEL_RESULT.json"
    valid = json.loads(result_path.read_text(encoding="utf-8"))
    forged = copy.deepcopy(valid)
    reconciled = forged["scheduler_attempts"][4]["reconciled_call_records"].pop()
    forged["scheduler_attempts"][0]["reconciled_call_records"] = [reconciled]
    result_path.chmod(0o644)
    result_path.write_text(
        json.dumps(forged, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    result_path.chmod(0o444)
    before = len(state["commands"])
    with pytest.raises(
        cancel.CancellationError,
        match="initial census reconciles a call that occurs later",
    ):
        cancel.explicit_cancel(submission)
    assert len(state["commands"]) == before
    result_path.chmod(0o644)
    result_path.write_text(
        json.dumps(valid, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    result_path.chmod(0o444)
    forged = copy.deepcopy(valid)
    reconciled = forged["scheduler_attempts"][4]["reconciled_call_records"]
    reconciled.append(copy.deepcopy(reconciled[0]))
    result_path.chmod(0o644)
    result_path.write_text(
        json.dumps(forged, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    result_path.chmod(0o444)
    before = len(state["commands"])
    with pytest.raises(cancel.CancellationError, match="reconciled call binding differs"):
        cancel.explicit_cancel(submission)
    assert len(state["commands"]) == before
