from __future__ import annotations

import ast
import base64
import copy
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest


PACKAGE = Path(__file__).resolve().parents[1]
ACTUAL_SUBMISSION_CONTRACT_IDENTITY = {
    "raw_sha256": (
        "bbeaa71f8f37f22cbe74c16c68b733742e8a4366838812832180257d145f5418"
    ),
    "git_provenance": {
        "branch": "main",
        "head": "33122e15d0aaf3661893a4c853fd5ac49173c685",
        "object_format": "sha1",
        "origin_main": "33122e15d0aaf3661893a4c853fd5ac49173c685",
        "remote_origin": "git@github.com:ChristianLin0420/treewm.git",
        "worktree_status": "clean",
        "worktree_status_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
    },
    "git_provenance_canonical_sha256": (
        "cea33bbdfbcd4849ea6acaff94d342a6403daba102e815aad35d16779b24e135"
    ),
}


def _actual_git_provenance():
    return copy.deepcopy(ACTUAL_SUBMISSION_CONTRACT_IDENTITY["git_provenance"])


def _write_source_archive_fixture(
    submission: Path, report_program: bytes, *, write_authorization: bool = True
) -> tuple[Path, str, int]:
    """Build the exact spoolable V2 source archive around fixture report bytes."""

    submission.mkdir(mode=0o700, parents=True, exist_ok=True)
    prefix = (PACKAGE / "report_repair.slurm").read_bytes()
    marker = b"__TREEWM_EXP23_SOURCE_ARCHIVE_V2_PAYLOAD__\n"
    end = b"\n__TREEWM_EXP23_SOURCE_ARCHIVE_V2_END__\n"
    assert prefix.endswith(marker) and prefix.count(marker) == 1
    protocol = "b" * 64
    raw_files = {
        "protocol.sha256": f"{protocol}\n".encode("ascii"),
        "report.py": report_program,
        "report_repair.py": b"fixture controller\n",
        "report_repair.slurm": prefix,
    }
    projection = {
        name: {
            "mode": 0o444,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in sorted(raw_files.items())
    }
    authority = {
        "schema_version": 2,
        "repair_source_commit": "c" * 40,
        "repair_package_protocol_sha256": protocol,
        "repair_source_files": projection,
        "repair_source_files_sha256": hashlib.sha256(
            json.dumps(
                projection,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
        "repair_source_installation_method": "direct_final_source_archive_o_excl",
    }
    envelope = {
        "archive_kind": "treewm_exp23_report_repair_source",
        "schema_version": 2,
        "authority": authority,
        "files": {
            name: {
                **projection[name],
                "data_base64": base64.b64encode(payload).decode("ascii"),
            }
            for name, payload in sorted(raw_files.items())
        },
    }
    body = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    payload = prefix + body + end
    archive = submission / "REPORT_REPAIR_0002_SOURCE_ARCHIVE.bin"
    archive.write_bytes(payload)
    archive.chmod(0o444)
    if write_authorization:
        authorization = submission / "REPORT_REPAIR_0002_AUTHORIZED.json"
        authorization.write_bytes(b'{"schema_version":1}\n')
        authorization.chmod(0o444)
    return archive, hashlib.sha256(payload).hexdigest(), len(payload)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def report():
    return _load("exp23_repair_test_report", "report.py")


@pytest.fixture
def repair():
    return _load("exp23_repair_test_controller", "report_repair.py")


@pytest.fixture(autouse=True)
def _strict_fixture_umask():
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.mark.parametrize("mask", [0o077, 0o022])
def test_report_seal_json_is_exact_under_process_umask(tmp_path, mask):
    target = tmp_path / f"sealed-{mask:o}.json"
    program = f"""
import importlib.util, json, os, pathlib, stat, sys
path = pathlib.Path({str(PACKAGE / 'report.py')!r})
spec = importlib.util.spec_from_file_location('subprocess_report', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
os.umask({mask})
target = pathlib.Path({str(target)!r})
digest = module.seal_json(target, {{'schema_version': 1, 'value': 'repair'}})
info = target.lstat()
print(json.dumps({{'digest': digest, 'mode': stat.S_IMODE(info.st_mode), 'size': info.st_size, 'uid': info.st_uid, 'nlink': info.st_nlink}}))
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", program],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    assert completed.returncode == 0, completed.stderr.decode()
    result = json.loads(completed.stdout)
    payload = target.read_bytes()
    assert result == {
        "digest": hashlib.sha256(payload).hexdigest(),
        "mode": 0o444,
        "size": len(payload),
        "uid": os.getuid(),
        "nlink": 1,
    }


def test_report_seal_json_rejects_nonidentical_existing_artifact(report, tmp_path):
    target = tmp_path / "artifact.json"
    report.seal_json(target, {"schema_version": 1, "value": 1})
    with pytest.raises(report.ReportError, match="differs"):
        report.seal_json(target, {"schema_version": 1, "value": 2})
    assert stat.S_IMODE(target.lstat().st_mode) == 0o444


def test_report_publication_uses_noreplace_and_exact_four_file_commit(report, tmp_path):
    body = {"status": "rejected", "reason": "repair fixture"}
    decision = {**body, "gate_sha256": report.stable_hash(body)}
    provenance = {
        "schema_version": 2,
        "publication_authority": {"status": "fixture"},
    }
    commit = report._legacy_staged_publish_report_locked(
        tmp_path,
        "a" * 64,
        {"schema_version": 1},
        decision,
        provenance,
    )
    assert len(commit) == 14
    assert set(commit) == {
        "schema_version",
        "status",
        "scientific_rejection",
        "campaign_id",
        "submission_sha256",
        "report_bundle",
        "report_bundle_sha256",
        "report_bundle_file_sha256",
        "gate_decision",
        "gate_sha256",
        "gate_decision_file_sha256",
        "provenance",
        "provenance_sha256",
        "provenance_file_sha256",
    }
    report_root = tmp_path / "report"
    assert stat.S_IMODE(report_root.lstat().st_mode) == 0o555
    assert len(list(report_root.iterdir())) == 4
    assert all(stat.S_IMODE(path.lstat().st_mode) == 0o444 for path in report_root.iterdir())


def test_normal_publish_requires_exact_active_original_report_job(report, tmp_path, monkeypatch):
    monkeypatch.setattr(
        report, "_validated_report_publication_prerequisite", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        report,
        "_validated_report_publication_receipt",
        lambda *_args, **_kwargs: {"report_job_id": "12345"},
    )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(report.ReportError, match="exact committed report Slurm job"):
        report.publish_report(
            tmp_path,
            "a" * 64,
            {},
            {"status": "rejected", "gate_sha256": "b" * 64},
            {},
        )
    assert not os.path.lexists(tmp_path / "report")


def test_repair_slurm_rejects_absent_slurm_job_id_before_publication(tmp_path):
    completed = subprocess.run(
        [
            "/bin/bash",
            str(PACKAGE / "report_repair.slurm"),
            str(tmp_path / "snapshot"),
            str(tmp_path / "submission"),
            "a" * 64,
            "2",
            str(tmp_path / "submission/REPORT_REPAIR_0002_SOURCE_ARCHIVE.bin"),
            "b" * 64,
            "1",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    assert completed.returncode == 2
    assert b"requires an exact active scalar SLURM_JOB_ID" in completed.stderr


@pytest.mark.parametrize("restart_count", ["1", "-1", "garbage", ""])
def test_repair_slurm_rejects_nonzero_or_malformed_restart_identity_before_publication(
    tmp_path, restart_count
):
    completed = subprocess.run(
        [
            "/bin/bash",
            str(PACKAGE / "report_repair.slurm"),
            str(tmp_path / "snapshot"),
            str(tmp_path / "submission"),
            "a" * 64,
            "2",
            str(tmp_path / "submission/REPORT_REPAIR_0002_SOURCE_ARCHIVE.bin"),
            "b" * 64,
            "1",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "SLURM_JOB_ID": "12345",
            "SLURM_RESTART_COUNT": restart_count,
        },
    )
    assert completed.returncode == 2
    assert b"cannot be requeued" in completed.stderr


@pytest.mark.parametrize("expected_hash_matches", [True, False])
@pytest.mark.parametrize("restart_count", [None, "0"])
def test_repair_slurm_verifies_sealed_source_when_executed_from_spool_copy(
    tmp_path, expected_hash_matches, restart_count
):
    submission = tmp_path / "submission"
    snapshot = tmp_path / "snapshot"
    spool = tmp_path / "slurm-spool"
    snapshot.mkdir()
    spool.mkdir()
    archive, archive_sha256, archive_size = _write_source_archive_fixture(
        submission, b"print(__file__)\n"
    )
    copied_script = spool / "slurm_script"
    # sbatch receives the retained archive descriptor and copies that exact
    # framed archive into its spool.  The runtime wrapper must execute/read the
    # spool inode, not reopen the canonical source path.
    copied_script.write_bytes(archive.read_bytes())
    copied_script.chmod(0o700)

    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "SLURM_JOB_ID": "444444",
    }
    if restart_count is not None:
        environment["SLURM_RESTART_COUNT"] = restart_count
    completed = subprocess.run(
        [
            "/bin/bash",
            str(copied_script),
            str(snapshot),
            str(submission),
            "a" * 64,
            "2",
            str(archive),
            (
                archive_sha256
                if expected_hash_matches
                else "b" * 64
            ),
            str(archive_size),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    if expected_hash_matches:
        assert completed.returncode == 0, completed.stderr.decode()
        assert completed.stdout.decode("ascii").strip() == f"{archive}::report.py"
    else:
        assert completed.returncode != 0
        assert completed.stdout == b""
        assert b"sealed repair source archive identity/hash differs" in completed.stderr


@pytest.mark.parametrize(
    ("declared_size", "message"),
    [
        (8 * 1024 * 1024, b"identity/hash differs"),
        (8 * 1024 * 1024 + 1, b"size is out of bounds"),
    ],
)
def test_repair_slurm_source_archive_size_cap_is_exact(
    tmp_path, declared_size, message
):
    submission = tmp_path / "submission"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    archive, archive_sha256, _archive_size = _write_source_archive_fixture(
        submission, b"print('size boundary')\n"
    )
    completed = subprocess.run(
        [
            "/bin/bash",
            str(PACKAGE / "report_repair.slurm"),
            str(snapshot),
            str(submission),
            "a" * 64,
            "2",
            str(archive),
            archive_sha256,
            str(declared_size),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "SLURM_JOB_ID": "444444",
        },
    )
    assert completed.returncode != 0
    assert message in completed.stderr


class _FakeLocks:
    def bindings(self):
        return (
            {"path": "/tmp/transaction", "device": 1, "inode": 2, "uid": os.getuid(), "mode": 0o600},
            {"path": "/tmp/report-cancel", "device": 1, "inode": 3, "uid": os.getuid(), "mode": 0o600},
        )


class _FixtureRetainedBinding:
    """Narrow adapter for unit-testing retained-only semantic helpers."""

    def retained_regular(self, path):
        if not path.is_file() or path.is_symlink():
            return None
        return path.read_bytes(), path.lstat()

    def retained_directory(self, path):
        return path.lstat(), frozenset(item.name for item in path.iterdir())


class _WaitFixtureBinding:
    """Retain the initial authority and admit only one later RELEASED file."""

    def __init__(self, submission_root):
        self.submission_root = submission_root
        authorization = submission_root / "REPORT_REPAIR_0002_AUTHORIZED.json"
        self._rows = {
            authorization.absolute(): (authorization.read_bytes(), authorization.lstat())
        }

    def retained_regular(self, path):
        return self._rows.get(path.absolute())

    def admit_release_evidence(self, name):
        assert name == "REPORT_REPAIR_0002_RELEASED.json"
        path = (self.submission_root / name).absolute()
        if path in self._rows:
            return True
        if not path.is_file() or path.is_symlink():
            return False
        self._rows[path] = (path.read_bytes(), path.lstat())
        return True

    def release_wait_is_open(self):
        return not any(
            "TERMINAL" in path.name or "CANCEL" in path.name
            for path in self.submission_root.iterdir()
        )


def _seed_retained_authority_fixture(module, submission):
    """Populate immutable roots that every real retained transition requires."""

    submission.mkdir(mode=0o700, parents=True, exist_ok=True)
    submission.chmod(0o700)
    for name in (
        "SUBMISSION_CONTRACT.json",
        "SUBMISSION_RECEIPT.json",
        "SUBMISSION_AUTHORIZATION.json",
    ):
        path = submission / name
        if not path.exists():
            path.write_bytes(
                _sealed_json_payload({"schema_version": 1, "fixture": name})
            )
            path.chmod(0o444)

    snapshot = submission / "source-snapshot"
    snapshot.mkdir(mode=0o700, exist_ok=True)
    snapshot.chmod(0o700)
    snapshot_repo = snapshot / "repo"
    snapshot_repo.mkdir(mode=0o700, exist_ok=True)
    snapshot_repo.chmod(0o700)
    snapshot_file = snapshot_repo / "retained-authority.fixture"
    if not snapshot_file.exists():
        snapshot_file.write_bytes(b"retained source snapshot authority\n")
        snapshot_file.chmod(0o444)
    snapshot_repo.chmod(0o555)
    snapshot.chmod(0o555)

    tasks = submission / "tasks"
    tasks.mkdir(mode=0o700, exist_ok=True)
    for index in range(20):
        cell = tasks / f"cell-{index:02d}"
        cell.mkdir(mode=0o700, exist_ok=True)
        launch_artifact = cell / "LAUNCH.json"
        if not launch_artifact.exists():
            launch_artifact.write_bytes(
                _sealed_json_payload(
                    {
                        "schema_version": 1,
                        "campaign_id": module.CAMPAIGN_ID,
                        "submission_sha256": module.EXPECTED_SUBMISSION_SHA256,
                        "cell_index": index,
                    }
                )
            )
            launch_artifact.chmod(0o444)
        receipt = cell / "WORKER_COMPLETE.json"
        if not receipt.exists():
            receipt.write_bytes(
                _sealed_json_payload(
                    {
                        "schema_version": 1,
                        "campaign_id": module.CAMPAIGN_ID,
                        "submission_sha256": module.EXPECTED_SUBMISSION_SHA256,
                        "cell_index": index,
                        "status": "worker_complete",
                    }
                )
            )
            receipt.chmod(0o444)
        waves = cell / "waves"
        waves.mkdir(mode=0o700, exist_ok=True)
        for wave_index, terminal_name in (
            (0, "CONTINUATION_READY.json"),
            (1, "WORKER_COMPLETE.json"),
        ):
            wave = waves / str(wave_index)
            wave.mkdir(mode=0o700, exist_ok=True)
            for name in (
                "START.json",
                "WORKER_SIGNAL_READY.json",
                terminal_name,
            ):
                artifact = wave / name
                if not artifact.exists():
                    artifact.write_bytes(
                        _sealed_json_payload(
                            {
                                "schema_version": 1,
                                "campaign_id": module.CAMPAIGN_ID,
                                "submission_sha256": module.EXPECTED_SUBMISSION_SHA256,
                                "cell_index": index,
                                "wave_index": wave_index,
                                "name": name,
                            }
                        )
                    )
                    artifact.chmod(0o444)

    launches = submission / "launches"
    launches.mkdir(mode=0o700, exist_ok=True)
    runs = submission / "fixture-runs"
    runs.mkdir(mode=0o700, exist_ok=True)
    for index in range(20):
        run_directory = runs / f"cell-{index:02d}"
        run_directory.mkdir(mode=0o700, exist_ok=True)
        launch = launches / f"cell-{index:02d}.json"
        if not launch.exists():
            launch.write_bytes(
                _sealed_json_payload(
                    {
                        "schema_version": 1,
                        "campaign_id": module.CAMPAIGN_ID,
                        "submission_sha256": module.EXPECTED_SUBMISSION_SHA256,
                        "cell_index": index,
                        "cell": {"run_directory": str(run_directory)},
                    }
                )
            )
            launch.chmod(0o444)

    logs = submission / "logs"
    logs.mkdir(mode=0o700, exist_ok=True)
    attempt1_log = logs / f"report-repair-0001-{module.EXPECTED_ATTEMPT1_JOB_ID}.out"
    if not attempt1_log.exists():
        attempt1_log.write_bytes(b"repair publication cannot be requeued\n")
        attempt1_log.chmod(0o600)


def _phase_fixture_value(repair, name):
    """Return one structurally exact attempt-2 append value for graph tests."""

    value = {key: None for key in repair._journal_artifact_keyset(name)}
    value.update(
        {
            "schema_version": 1,
            "status": repair._journal_artifact_status(name),
            "campaign_id": repair.CAMPAIGN_ID,
            "submission_sha256": repair.EXPECTED_SUBMISSION_SHA256,
            "attempt": repair.ATTEMPT,
        }
    )
    if name == "CALLING_REPORT_REPAIR_0002_SUBMIT.json":
        value["scheduler_source_archive_input"] = {
            "schema_version": 1,
            "transport": "inherited_proc_fd",
            "descriptor": 198,
            "argument": "/proc/self/fd/198",
            "source_archive": None,
            "sha256": None,
            "size": None,
            "file_identity": None,
        }
    return value


def _seed_direct_authorized_phase(repair, submission):
    """Install the immutable graph/namespace assumed by low-level unit calls."""

    _seed_retained_authority_fixture(repair, submission)
    journal = submission / "journal"
    journal.mkdir(parents=True, mode=0o700, exist_ok=True)
    journal.chmod(0o700)
    historical_names = {
        *repair.EXPECTED_ATTEMPT1_CHAIN_SHA256,
        "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
    }
    attempt2_names = {
        "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json",
        "CALLING_REPORT_REPAIR_0002_SUBMIT.json",
        "REPORT_REPAIR_0002_SUBMITTED.json",
        "REPORT_REPAIR_0002_AUTHORIZED.json",
    }
    for name in sorted(historical_names):
        path = journal / name
        if not os.path.lexists(path):
            path.write_bytes(b"fixture\n")
            path.chmod(0o444)
    for name in sorted(attempt2_names):
        value = _phase_fixture_value(repair, name)
        path = submission / name
        if not os.path.lexists(path):
            path.write_bytes(_sealed_json_payload(value))
            path.chmod(0o444)
    repair_parent = submission / "report-repair"
    repair_parent.mkdir(mode=0o700, exist_ok=True)
    repair_parent.chmod(0o700)
    attempt1_root = repair_parent / "attempt-0001"
    attempt1_root.mkdir(mode=0o700, exist_ok=True)
    attempt1_root.chmod(0o700)
    attempt1_source = attempt1_root / "source"
    attempt1_source.mkdir(mode=0o700, exist_ok=True)
    attempt1_source.chmod(0o555)
    archive = submission / repair.SOURCE_ARCHIVE_NAME
    if not archive.exists():
        archive, _digest, _size = _write_source_archive_fixture(
            submission,
            b"print('fixture report')\n",
            write_authorization=False,
        )
    return repair._load_sealed_repair_source(archive)


def _direct_release(repair, submission, *args, **kwargs):
    sealed_source = _seed_direct_authorized_phase(repair, submission)
    rebound_args = list(args)
    rebound_args[2] = {**dict(rebound_args[2]), **sealed_source}
    original_revalidate = repair._revalidated_sealed_json

    def fixture_revalidate(path, expected, digest, label):
        if (
            path.name == "REPORT_REPAIR_0002_AUTHORIZED.json"
            and path.read_bytes()
            == _sealed_json_payload(
                _phase_fixture_value(
                    repair, "REPORT_REPAIR_0002_AUTHORIZED.json"
                )
            )
        ):
            return dict(expected)
        return original_revalidate(path, expected, digest, label)

    repair._revalidated_sealed_json = fixture_revalidate
    try:
        return getattr(repair, "_release_authorized_job")(
            submission, *rebound_args, **kwargs
        )
    finally:
        repair._revalidated_sealed_json = original_revalidate


def _direct_cleanup(repair, submission, *args, **kwargs):
    _seed_direct_authorized_phase(repair, submission)
    locks = args[5]
    with repair._retained_transition_scope(
        submission, locks, source_must_be_installed=True
    ):
        return getattr(repair, "_cleanup_repair_rows")(
            submission, *args, **kwargs
        )


def _report_payload_fixture(report, installation_method=None):
    body = {"status": "rejected", "reason": "attempt2 publication fixture"}
    decision = {**body, "gate_sha256": report.stable_hash(body)}
    method = installation_method or report.DIRECT_FINAL_INSTALL_METHOD
    provenance = {
        "schema_version": 2,
        "publication_authority": {
            "schema_version": 2,
            "status": "authorized_terminal_report_repair",
            "attempt": 2,
            "report_publication_installation_method": method,
        },
    }
    return {"schema_version": 1}, decision, provenance


def _complete_report_staging(report, tmp_path, submission, method=None):
    bundle, decision, provenance = _report_payload_fixture(report, method)
    builder = tmp_path / f"report-builder-{len(list(tmp_path.glob('report-builder-*')))}"
    builder.mkdir()
    commit = report._legacy_staged_publish_report_locked(
        builder,
        "a" * 64,
        bundle,
        decision,
        provenance,
    )
    staging = submission / ".report.tmp.killpoint"
    shutil.copytree(builder / "report", staging)
    return staging, commit, bundle, decision, provenance


def _report_tree_validator(expected_payloads):
    calls = []

    def validate(root):
        assert stat.S_IMODE(root.lstat().st_mode) == 0o555
        assert {path.name for path in root.iterdir()} == set(expected_payloads)
        for name, payload in expected_payloads.items():
            path = root / name
            assert stat.S_IMODE(path.lstat().st_mode) == 0o444
            assert path.read_bytes() == payload
        calls.append(root)

    return validate, calls


@pytest.mark.parametrize(
    "method",
    [
        "renameat2_noreplace",
        f"locked_same_parent_rename_after_errno_{errno.EINVAL}",
    ],
)
def legacy_repaired_report_installation_method_is_exact_and_transitively_bound(
    report, tmp_path, monkeypatch, method
):
    bundle, decision, provenance = _report_payload_fixture(report, method)
    real_rename = os.rename
    primary_calls = []
    fallback_calls = []

    def primary(parent_fd, source_name, target_name):
        primary_calls.append((source_name, target_name, parent_fd))
        real_rename(
            source_name,
            target_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )

    def fallback(source_name, target_name, **kwargs):
        fallback_calls.append((source_name, target_name, dict(kwargs)))
        return real_rename(source_name, target_name, **kwargs)

    if method == report.INSTALL_METHOD_PRIMARY:
        monkeypatch.setattr(report, "_renameat2_noreplace", primary)
        monkeypatch.setattr(
            report.os,
            "rename",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("primary report install entered fallback")
            ),
        )
    else:
        monkeypatch.setattr(
            report,
            "_renameat2_noreplace",
            lambda *_args: (_ for _ in ()).throw(
                AssertionError("fallback report install entered renameat2")
            ),
        )
        monkeypatch.setattr(report.os, "rename", fallback)

    commit = report._publish_report_locked(
        tmp_path,
        "a" * 64,
        bundle,
        decision,
        provenance,
        repair_installation_method=method,
        repair_locks=_FakeLocks(),
    )
    if method == report.INSTALL_METHOD_PRIMARY:
        assert len(primary_calls) == 1 and fallback_calls == []
    else:
        assert primary_calls == [] and len(fallback_calls) == 1
        source_name, target_name, kwargs = fallback_calls[0]
        assert source_name.startswith(".report.tmp.") and "/" not in source_name
        assert target_name == "report"
        assert kwargs["src_dir_fd"] == kwargs["dst_dir_fd"]
        assert type(kwargs["src_dir_fd"]) is int
    stored = json.loads((tmp_path / "report" / commit["provenance"]).read_text())
    assert (
        stored["publication_authority"][
            "report_publication_installation_method"
        ]
        == method
    )
    assert commit["provenance_sha256"] == report.stable_hash(stored)
    assert not list(tmp_path.glob(".report.tmp.*"))


@pytest.mark.parametrize("injection", ["target", "unknown"])
def legacy_repaired_report_fallback_rechecks_namespace_before_plain_rename(
    report, tmp_path, monkeypatch, injection
):
    submission = tmp_path / "submission"
    submission.mkdir()
    staging, _commit, _bundle, _decision, _provenance = _complete_report_staging(
        report, tmp_path, submission
    )
    expected_payloads = {path.name: path.read_bytes() for path in staging.iterdir()}
    validate, calls = _report_tree_validator(expected_payloads)
    target = submission / "report"
    unexpected = submission / "unexpected"

    def inject_on_second_validation(root):
        validate(root)
        if len(calls) == 2:
            if injection == "target":
                target.mkdir()
            else:
                unexpected.write_text("preserve", encoding="ascii")

    monkeypatch.setattr(
        report.os,
        "rename",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("namespace injection reached plain rename")
        ),
    )
    with pytest.raises((FileExistsError, report.ReportError)):
        report._install_repaired_report_directory(
            staging,
            target,
            installation_method=(
                f"{report.INSTALL_METHOD_FALLBACK_PREFIX}{errno.EINVAL}"
            ),
            locks=_FakeLocks(),
            baseline_names=set(),
            validate_tree=inject_on_second_validation,
        )
    assert os.path.lexists(target if injection == "target" else unexpected)
    assert os.path.lexists(staging)


def legacy_repaired_report_fallback_rejects_exact_content_on_a_different_inode(
    report, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    submission.mkdir()
    staging, _commit, _bundle, _decision, _provenance = _complete_report_staging(
        report, tmp_path, submission
    )
    expected_payloads = {path.name: path.read_bytes() for path in staging.iterdir()}
    validate, calls = _report_tree_validator(expected_payloads)
    target = submission / "report"
    moved = submission / ".moved-original-report"
    real_rename = os.rename

    def swap_after_postinstall_validation(root):
        validate(root)
        if len(calls) == 3:
            real_rename(root, moved)
            shutil.copytree(moved, root)
            moved.chmod(0o700)
            shutil.rmtree(moved)

    with pytest.raises(report.ReportError, match="namespace/identity"):
        report._install_repaired_report_directory(
            staging,
            target,
            installation_method=(
                f"{report.INSTALL_METHOD_FALLBACK_PREFIX}{errno.EINVAL}"
            ),
            locks=_FakeLocks(),
            baseline_names=set(),
            validate_tree=swap_after_postinstall_validation,
        )
    assert not os.path.lexists(staging)
    assert target.is_dir() and not os.path.lexists(moved)


@pytest.mark.parametrize("drift", ["parent_mode", "lock_binding"])
def legacy_repaired_report_fallback_rejects_parent_or_lock_drift_before_rename(
    report, tmp_path, monkeypatch, drift
):
    submission = tmp_path / "submission"
    submission.mkdir()
    staging, _commit, _bundle, _decision, _provenance = _complete_report_staging(
        report, tmp_path, submission
    )
    expected_payloads = {path.name: path.read_bytes() for path in staging.iterdir()}
    validate, calls = _report_tree_validator(expected_payloads)

    class MutableLocks(_FakeLocks):
        changed = False

        def bindings(self):
            transaction, report_cancel = super().bindings()
            if self.changed:
                transaction = {**transaction, "inode": 999}
            return transaction, report_cancel

    locks = MutableLocks()

    def drift_on_second_validation(root):
        validate(root)
        if len(calls) == 2:
            if drift == "parent_mode":
                submission.chmod(0o711)
            else:
                locks.changed = True

    monkeypatch.setattr(
        report.os,
        "rename",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("authority drift reached plain rename")
        ),
    )
    with pytest.raises(report.ReportError, match="authority changed"):
        report._install_repaired_report_directory(
            staging,
            submission / "report",
            installation_method=(
                f"{report.INSTALL_METHOD_FALLBACK_PREFIX}{errno.EINVAL}"
            ),
            locks=locks,
            baseline_names=set(),
            validate_tree=drift_on_second_validation,
        )
    assert os.path.lexists(staging)
    assert not os.path.lexists(submission / "report")


@pytest.mark.parametrize(
    "crash_state",
    [
        "first_file_partial",
        "commit_write_partial",
        "sealed_complete",
        "directory_reopened",
        "commit_invalidated",
        "during_chmod_pass",
        "after_first_unlink",
    ],
)
def legacy_repaired_report_staging_crash_prefixes_resume_idempotently(
    report, tmp_path, crash_state
):
    submission = tmp_path / "submission"
    submission.mkdir()
    method = f"{report.INSTALL_METHOD_FALLBACK_PREFIX}{errno.EINVAL}"
    staging, expected_commit, bundle, decision, provenance = _complete_report_staging(
        report, tmp_path, submission, method
    )
    if crash_state != "sealed_complete":
        staging.chmod(0o700)
    if crash_state == "first_file_partial":
        names = sorted(path.name for path in staging.iterdir())
        victim_name = next(name for name in names if name != "REPORT_COMMIT.json")
        for path in list(staging.iterdir()):
            path.chmod(0o600)
            path.unlink()
        victim = staging / victim_name
        victim.write_bytes(b"partial")
        victim.chmod(0o600)
    elif crash_state == "commit_write_partial":
        commit_path = staging / "REPORT_COMMIT.json"
        commit_path.chmod(0o600)
        commit_path.write_bytes(b"{")
    if crash_state in {
        "commit_invalidated",
        "during_chmod_pass",
        "after_first_unlink",
    }:
        (staging / "REPORT_COMMIT.json").chmod(0o600)
    noncommit = sorted(
        path for path in staging.iterdir() if path.name != "REPORT_COMMIT.json"
    )
    if crash_state in {"during_chmod_pass", "after_first_unlink"}:
        noncommit[0].chmod(0o600)
    if crash_state == "after_first_unlink":
        for path in noncommit[1:]:
            path.chmod(0o600)
        noncommit[0].unlink()

    commit = report._publish_report_locked(
        submission,
        "a" * 64,
        bundle,
        decision,
        provenance,
        repair_installation_method=method,
        repair_locks=_FakeLocks(),
    )
    assert commit == expected_commit
    assert not os.path.lexists(staging)
    assert stat.S_IMODE((submission / "report").lstat().st_mode) == 0o555
    assert (
        report._publish_report_locked(
            submission,
            "a" * 64,
            bundle,
            decision,
            provenance,
            repair_installation_method=method,
            repair_locks=_FakeLocks(),
        )
        == commit
    )


def legacy_repaired_report_postrename_fsync_failure_recovers_without_second_install(
    report, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    submission.mkdir()
    method = f"{report.INSTALL_METHOD_FALLBACK_PREFIX}{errno.EINVAL}"
    staging, expected_commit, bundle, decision, provenance = _complete_report_staging(
        report, tmp_path, submission, method
    )
    expected_payloads = {path.name: path.read_bytes() for path in staging.iterdir()}
    validate, _calls = _report_tree_validator(expected_payloads)
    real_fsync = report.os.fsync

    def fail_parent_fsync(_descriptor):
        raise OSError(errno.EIO, os.strerror(errno.EIO))

    monkeypatch.setattr(report.os, "fsync", fail_parent_fsync)
    with pytest.raises(OSError, match="Input/output error"):
        report._install_repaired_report_directory(
            staging,
            submission / "report",
            installation_method=method,
            locks=_FakeLocks(),
            baseline_names=set(),
            validate_tree=validate,
        )
    assert not os.path.lexists(staging)
    assert (submission / "report").is_dir()

    monkeypatch.setattr(report.os, "fsync", real_fsync)
    assert (
        report._publish_report_locked(
            submission,
            "a" * 64,
            bundle,
            decision,
            provenance,
            repair_installation_method=method,
            repair_locks=_FakeLocks(),
        )
        == expected_commit
    )


def legacy_repaired_report_completed_commit_with_missing_coverage_is_rejected(
    report, tmp_path
):
    submission = tmp_path / "submission"
    submission.mkdir()
    staging, _commit, bundle, decision, provenance = _complete_report_staging(
        report, tmp_path, submission
    )
    staging.chmod(0o700)
    victim = next(
        path for path in staging.iterdir() if path.name != "REPORT_COMMIT.json"
    )
    victim.unlink()
    with pytest.raises(report.ReportError, match="completed report staging coverage"):
        report._publish_report_locked(
            submission,
            "a" * 64,
            bundle,
            decision,
            provenance,
            repair_installation_method=report.INSTALL_METHOD_PRIMARY,
            repair_locks=_FakeLocks(),
        )
    assert os.path.lexists(staging)
    assert not os.path.lexists(submission / "report")


def legacy_repaired_report_refuses_install_probe_residue_before_staging(
    report, tmp_path
):
    bundle, decision, provenance = _report_payload_fixture(report)
    residue = tmp_path / ".report.install-probe-source"
    residue.mkdir(mode=0o700)
    with pytest.raises(report.ReportError, match="baseline namespace"):
        report._publish_report_locked(
            tmp_path,
            "a" * 64,
            bundle,
            decision,
            provenance,
            repair_installation_method=report.INSTALL_METHOD_PRIMARY,
            repair_locks=_FakeLocks(),
        )
    assert residue.is_dir()
    assert not os.path.lexists(tmp_path / "report")
    assert not list(tmp_path.glob(".report.tmp.*"))


@pytest.mark.parametrize("forgery", ["unknown", "hardlink", "symlink", "fifo", "mode"])
def legacy_repaired_report_partial_staging_rejects_unsafe_entries(
    report, tmp_path, forgery
):
    submission = tmp_path / "submission"
    submission.mkdir()
    bundle, decision, provenance = _report_payload_fixture(report)
    staging = submission / ".report.tmp.hostile"
    staging.mkdir(mode=0o700)
    bundle_name = f"REPORT_BUNDLE.{report.stable_hash(bundle)}.json"
    target = staging / bundle_name
    if forgery == "unknown":
        (staging / "unexpected").write_text("forged", encoding="ascii")
    elif forgery == "hardlink":
        external = tmp_path / "external"
        external.write_text("forged", encoding="ascii")
        os.link(external, target)
    elif forgery == "symlink":
        target.symlink_to(tmp_path / "external")
    elif forgery == "fifo":
        os.mkfifo(target, 0o600)
    else:
        target.write_text("forged", encoding="ascii")
        target.chmod(0o640)
    with pytest.raises(report.ReportError, match="report staging cleanup"):
        report._publish_report_locked(
            submission,
            "a" * 64,
            bundle,
            decision,
            provenance,
            repair_installation_method=report.INSTALL_METHOD_PRIMARY,
            repair_locks=_FakeLocks(),
        )
    assert os.path.lexists(staging)
    assert not os.path.lexists(submission / "report")


@pytest.mark.parametrize(
    "state",
    ["wrong_0444", "distinct_0444", "distinct_0600", "linked_wrong_0444"],
)
def legacy_report_staging_inner_seal_is_classified_before_any_mutation(
    report, tmp_path, state
):
    submission = tmp_path / "submission"
    submission.mkdir(mode=0o700)
    staging = submission / ".report.tmp.1.1"
    staging.mkdir(mode=0o700)
    target_name = "artifact.json"
    target = staging / target_name
    stage = staging / f".{target_name}.seal.tmp"
    expected = b'{"value":1}\n'
    staged_payload = b"wrong\n" if "wrong" in state else expected
    stage.write_bytes(staged_payload)
    stage.chmod(0o600 if state == "distinct_0600" else 0o444)
    if state.startswith("distinct"):
        target.write_bytes(expected)
        target.chmod(0o444)
    elif state == "linked_wrong_0444":
        os.link(stage, target)
    before = {
        path.name: (
            path.read_bytes(),
            path.lstat().st_dev,
            path.lstat().st_ino,
            path.lstat().st_nlink,
            stat.S_IMODE(path.lstat().st_mode),
        )
        for path in staging.iterdir()
    }
    with pytest.raises(report.ReportError, match="artifact-seal"):
        report._remove_report_staging(
            staging,
            expected_payloads={target_name: expected},
            validate_complete=lambda _path: None,
        )
    after = {
        path.name: (
            path.read_bytes(),
            path.lstat().st_dev,
            path.lstat().st_ino,
            path.lstat().st_nlink,
            stat.S_IMODE(path.lstat().st_mode),
        )
        for path in staging.iterdir()
    }
    assert after == before


@pytest.mark.parametrize("state", ["partial_0600", "detached_0444", "linked_0444"])
def legacy_report_staging_inner_seal_crash_prefix_is_restartable(
    report, tmp_path, state
):
    submission = tmp_path / "submission"
    submission.mkdir(mode=0o700)
    staging = submission / ".report.tmp.1.1"
    staging.mkdir(mode=0o700)
    target_name = "artifact.json"
    target = staging / target_name
    stage = staging / f".{target_name}.seal.tmp"
    expected = b'{"value":1}\n'
    stage.write_bytes(b"partial" if state == "partial_0600" else expected)
    stage.chmod(0o600 if state == "partial_0600" else 0o444)
    if state == "linked_0444":
        os.link(stage, target)
    report._remove_report_staging(
        staging,
        expected_payloads={target_name: expected},
        validate_complete=lambda _path: None,
    )
    assert not os.path.lexists(staging)


def legacy_report_staging_inner_seal_inode_swap_is_rejected_untouched(
    report, tmp_path
):
    submission = tmp_path / "submission"
    submission.mkdir(mode=0o700)
    staging = submission / ".report.tmp.1.1"
    staging.mkdir(mode=0o700)
    target_name = "artifact.json"
    stage = staging / f".{target_name}.seal.tmp"
    displaced = staging / ".artifact.json.seal.displaced"
    stage.write_bytes(b"partial-original")
    stage.chmod(0o600)
    checks = 0

    def swap_after_read_only_classification():
        nonlocal checks
        checks += 1
        if checks == 2:
            stage.rename(displaced)
            stage.write_bytes(b"partial-replacement")
            stage.chmod(0o600)

    with pytest.raises(
        report.ReportError, match="artifact-seal changed|namespace changed"
    ):
        report._remove_report_staging(
            staging,
            expected_payloads={target_name: b'{"value":1}\n'},
            validate_complete=lambda _path: None,
            phase_check=swap_after_read_only_classification,
        )
    assert stage.read_bytes() == b"partial-replacement"
    assert displaced.read_bytes() == b"partial-original"


@pytest.mark.parametrize("crash_state", ["source_created", "target_installed"])
def legacy_directory_install_probe_crash_residue_is_cleaned_before_retry(
    repair, tmp_path, monkeypatch, crash_state
):
    parent = tmp_path / "attempt"
    parent.mkdir(mode=0o700)
    real_rename = os.rename

    def crash_probe(parent_fd, source_name, target_name):
        if crash_state == "target_installed":
            real_rename(
                source_name,
                target_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        raise RuntimeError("probe crash")

    monkeypatch.setattr(repair, "_renameat2_noreplace", crash_probe)
    with pytest.raises(RuntimeError, match="probe crash"):
        repair._directory_install_method_probe(
            parent,
            "source",
            _FakeLocks(),
            expected_baseline=set(),
        )
    expected_residue = (
        ".source.install-probe-target"
        if crash_state == "target_installed"
        else ".source.install-probe-source"
    )
    assert {path.name for path in parent.iterdir()} == {expected_residue}

    def primary(parent_fd, source_name, target_name):
        real_rename(
            source_name,
            target_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )

    monkeypatch.setattr(repair, "_renameat2_noreplace", primary)
    assert (
        repair._directory_install_method_probe(
            parent,
            "source",
            _FakeLocks(),
            expected_baseline=set(),
        )
        == repair.SOURCE_INSTALL_METHOD_PRIMARY
    )
    assert list(parent.iterdir()) == []


@pytest.mark.parametrize("residue", ["both", "file", "symlink", "nonempty", "unknown"])
def legacy_directory_install_probe_rejects_ambiguous_or_hostile_residue(
    repair, tmp_path, residue
):
    parent = tmp_path / "attempt"
    parent.mkdir(mode=0o700)
    source = parent / ".source.install-probe-source"
    target = parent / ".source.install-probe-target"
    if residue == "both":
        source.mkdir(mode=0o700)
        target.mkdir(mode=0o700)
    elif residue == "file":
        source.write_text("hostile", encoding="ascii")
    elif residue == "symlink":
        source.symlink_to(tmp_path / "absent")
    elif residue == "nonempty":
        source.mkdir(mode=0o700)
        (source / "payload").write_text("hostile", encoding="ascii")
    else:
        (parent / "unknown").write_text("hostile", encoding="ascii")

    with pytest.raises(repair.RepairError):
        repair._directory_install_method_probe(
            parent,
            "source",
            _FakeLocks(),
            expected_baseline=set(),
        )
    assert list(parent.iterdir())


@pytest.mark.parametrize("unsupported_errno", [errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP])
def legacy_directory_install_probe_selects_only_narrow_capability_fallback(
    repair, tmp_path, monkeypatch, unsupported_errno
):
    parent = tmp_path / "attempt"
    parent.mkdir(mode=0o700)

    def unsupported(*_args):
        raise OSError(unsupported_errno, os.strerror(unsupported_errno))

    monkeypatch.setattr(repair, "_renameat2_noreplace", unsupported)
    assert repair._directory_install_method_probe(
        parent,
        "source",
        _FakeLocks(),
        expected_baseline=set(),
    ) == f"{repair.SOURCE_INSTALL_METHOD_FALLBACK_PREFIX}{unsupported_errno}"
    assert list(parent.iterdir()) == []


def _stream(payload: bytes):
    return {
        "encoding": "base64",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "data": base64.b64encode(payload).decode("ascii"),
    }


def _command_result(repair, stdout=b"", stderr=b"", returncode=0):
    return repair.CommandResult(returncode, stdout, stderr)


def _squeue_result(repair, rows):
    payload = "".join(
        "|".join(
            [
                row["job_id"],
                row["job_name"],
                row["owner"],
                row["state"],
                row["comment"],
                row["reason"],
            ]
        )
        + "\n"
        for row in rows
    ).encode()
    return _command_result(repair, payload)


def _repair_sacct_result(
    repair,
    *,
    job_id="444444",
    state="RUNNING",
    exit_code="0:0",
    start="2026-08-29T10:00:02",
    end="",
    reason="None",
):
    environment = repair._scheduler_environment("/tmp/slurm.conf")
    parsed = {
        "JobIDRaw": job_id,
        "JobName": repair._repair_name(repair.EXPECTED_SUBMISSION_SHA256),
        "User": environment["USER"],
        "State": state,
        "ExitCode": exit_code,
        "ElapsedRaw": "61",
        "AllocNodes": "1",
        "NodeList": "cpu-fixture",
        "Submit": "2026-08-29T10:00:00",
        "Eligible": "2026-08-29T10:00:01",
        "Start": start,
        "End": end,
        "Comment": repair._repair_comment(repair.EXPECTED_SUBMISSION_SHA256),
        "Reason": reason,
    }
    payload = (
        "|".join(parsed[field] for field in repair.REPAIR_SACCT_FIELDS) + "\n"
    ).encode()
    return _command_result(repair, payload)


def _wait_authorization(report, submission, submission_sha="a" * 64, job_id="444444"):
    value = {key: None for key in report.REPORT_REPAIR_AUTHORIZATION_KEYS}
    value.update(
        {
            "schema_version": 1,
            "status": "authorized_terminal_report_repair",
            "campaign_id": report.CAMPAIGN_ID,
            "submission_sha256": submission_sha,
            "attempt": 2,
            "repair_report_job_id": job_id,
            "worker_handoff": dict(report.REPAIR_WORKER_HANDOFF),
            "publication_allowed": True,
            "scheduler_submission_allowed": False,
        }
    )
    path = submission / "REPORT_REPAIR_0002_AUTHORIZED.json"
    digest = report.seal_json(path, value)
    return value, digest


def _repair_row(repair, job_id="444444", state="PENDING", reason="JobHeldUser"):
    owner = repair._scheduler_environment("/tmp/slurm.conf")["USER"]
    return {
        "job_id": job_id,
        "job_name": repair._repair_name(repair.EXPECTED_SUBMISSION_SHA256),
        "owner": owner,
        "state": state,
        "comment": repair._repair_comment(repair.EXPECTED_SUBMISSION_SHA256),
        "reason": reason,
    }


def _census(repair, rows):
    environment = repair._scheduler_environment("/tmp/slurm.conf")
    argv = [
        "/usr/local/bin/squeue",
        "--noheader",
        f"--user={environment['USER']}",
        f"--format={repair.SQUEUE_FORMAT}",
    ]
    raw_result = _squeue_result(repair, rows)
    raw = repair._command_evidence(argv, environment, raw_result)
    rounds = [
        {"round": index, "raw": raw, "relevant_rows": rows} for index in range(3)
    ]
    return {
        "schema_version": 1,
        "rounds": rounds,
        "settled_rows": rows,
        "captured_at_utc": "2026-08-29T00:00:00Z",
    }


def _scheduler_contract():
    return {"scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}}


def _held_job_control_result(
    repair, submission, job_id="444444", *, scheduler_command=None
):
    environment = repair._scheduler_environment("/tmp/slurm.conf")
    source = submission / repair.SOURCE_ARCHIVE_NAME
    calling_path = submission / "CALLING_REPORT_REPAIR_0002_SUBMIT.json"
    scheduler_command = str(source) if scheduler_command is None else scheduler_command
    if (
        scheduler_command == str(source)
        and calling_path.is_file()
        and not calling_path.is_symlink()
    ):
        calling = json.loads(calling_path.read_text(encoding="utf-8"))
        scheduler_input = calling.get("scheduler_source_archive_input")
        if isinstance(scheduler_input, dict) and isinstance(
            scheduler_input.get("argument"), str
        ):
            scheduler_command = scheduler_input["argument"]
    output = submission / f"logs/report-repair-0002-{job_id}.out"
    fields = {
        "JobId": job_id,
        "JobName": repair._repair_name(repair.EXPECTED_SUBMISSION_SHA256),
        "UserId": f"{environment['USER']}({os.getuid()})",
        "JobState": "PENDING",
        "Reason": "JobHeldUser",
        "Requeue": "0",
        "Restarts": "0",
        "BatchFlag": "1",
        "TimeLimit": "04:00:00",
        "Comment": repair._repair_comment(repair.EXPECTED_SUBMISSION_SHA256),
        "Partition": "cpu",
        "Account": "edgeai_tao-ptm_image-foundation-model-clip",
        "QOS": "normal",
        "NumNodes": "1",
        "NumCPUs": "12",
        "NumTasks": "1",
        "CPUs/Task": "12",
        "MinMemoryNode": "64G",
        "Command": scheduler_command,
        "WorkDir": str(submission / "source-snapshot/repo"),
        "StdOut": str(output),
        "StdErr": str(output),
        "StdIn": "/dev/null",
    }
    return _command_result(
        repair,
        (" ".join(f"{key}={value}" for key, value in fields.items()) + "\n").encode(),
    )


def _with_job_control(repair, submission, delegate):
    def wrapped(argv, cwd, environment):
        if list(argv[:4]) == ["/usr/local/bin/scontrol", "show", "job", "-dd"]:
            assert len(argv) == 5
            return _held_job_control_result(repair, submission, str(argv[4]))
        return delegate(argv, cwd, environment)

    return wrapped


def _attempt1_terminal_sacct_payload(repair):
    environment = repair._scheduler_environment("/tmp/slurm.conf")
    parsed = {
        "JobIDRaw": repair.EXPECTED_ATTEMPT1_JOB_ID,
        "JobName": repair.EXPECTED_ATTEMPT1_JOB_NAME,
        "User": environment["USER"],
        "State": "FAILED",
        "ExitCode": "2:0",
        "ElapsedRaw": "5",
        "AllocNodes": "1",
        "NodeList": "cpu-00049",
        "Submit": "2026-08-29T13:22:38",
        "Eligible": "2026-08-29T13:22:41",
        "Start": "2026-08-29T13:22:43",
        "End": "2026-08-29T13:22:48",
        "Comment": repair.EXPECTED_ATTEMPT1_COMMENT,
        "Reason": "None",
    }
    return (
        "|".join(parsed[field] for field in repair.REPAIR_SACCT_FIELDS) + "\n"
    ).encode("utf-8")


def _synthetic_attempt1_terminal_and_predecessor(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    (submission / "logs").mkdir()
    _seed_retained_authority_fixture(repair, submission)
    for name in repair.EXPECTED_ATTEMPT1_CHAIN_SHA256:
        path = submission / "journal" / name
        path.write_bytes(b"fixture\n")
        path.chmod(0o444)
    repair_parent = submission / "report-repair"
    repair_parent.mkdir(mode=0o700)
    attempt1 = repair_parent / "attempt-0001"
    attempt1.mkdir(mode=0o700)
    source_root = attempt1 / "source"
    source_root.mkdir(mode=0o700)
    source_root.chmod(0o555)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    census = _census(repair, [])
    accounting_payload = _attempt1_terminal_sacct_payload(repair)
    assert (
        hashlib.sha256(accounting_payload).hexdigest()
        == repair.EXPECTED_ATTEMPT1_TERMINAL_SACCT_STDOUT_SHA256
    )

    def accounting_runner(argv, _cwd, _environment):
        assert list(argv) == repair._attempt1_accounting_argv()
        return _command_result(repair, accounting_payload)

    accounting = repair._attempt1_terminal_scheduler_observation(
        submission, contract, accounting_runner, _FakeLocks()
    )
    terminal, terminal_sha = repair._seal_attempt1_worker_failure_terminal(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        census,
        accounting,
        _FakeLocks(),
    )
    source = {
        "root": str(submission / "report-repair/attempt-0001/source"),
        "authority": "report-repair/attempt-0001/source/SOURCE_AUTHORITY.json",
        "authority_sha256": "e" * 64,
        "schema_version": 1,
        "repair_source_commit": repair.EXPECTED_ATTEMPT1_SOURCE_COMMIT,
        "repair_package_protocol_sha256": repair.EXPECTED_ATTEMPT1_SOURCE_PROTOCOL,
        "repair_source_files": copy.deepcopy(repair.EXPECTED_ATTEMPT1_SOURCE_FILES),
        "repair_source_files_sha256": repair.EXPECTED_ATTEMPT1_SOURCE_FILES_SHA256,
    }
    chain = {
        "schema_version": 1,
        "files": dict(repair.EXPECTED_ATTEMPT1_CHAIN_SHA256),
        "source": source,
    }
    monkeypatch.setattr(
        repair, "_validated_attempt1_chain", lambda _submission: copy.deepcopy(chain)
    )
    environment_payload = b"SLURM_EXPORT_ENV=NONE\nFIXTURE=attempt1\n"
    batch_payload = (
        b"#!/bin/bash\n"
        b'[[ -n "${SLURM_RESTART_COUNT+x}" && '
        b'"$SLURM_RESTART_COUNT" == "0" ]]\n'
    )
    monkeypatch.setattr(
        repair,
        "EXPECTED_ATTEMPT1_ENV_STDOUT_SHA256",
        hashlib.sha256(environment_payload).hexdigest(),
    )
    monkeypatch.setattr(
        repair, "EXPECTED_ATTEMPT1_ENV_STDOUT_SIZE", len(environment_payload)
    )
    monkeypatch.setattr(
        repair,
        "EXPECTED_ATTEMPT1_BATCH_STDOUT_SHA256",
        hashlib.sha256(batch_payload).hexdigest(),
    )
    monkeypatch.setattr(
        repair, "EXPECTED_ATTEMPT1_BATCH_STDOUT_SIZE", len(batch_payload)
    )
    log_path = (
        submission
        / "logs"
        / f"report-repair-0001-{repair.EXPECTED_ATTEMPT1_JOB_ID}.out"
    )
    log_path.write_bytes(repair.EXPECTED_ATTEMPT1_LOG_BYTES)
    log_path.chmod(0o600)

    def retained_runner(argv, _cwd, _environment):
        assert tuple(argv[:3]) == (
            "/usr/local/bin/sacct",
            "-j",
            repair.EXPECTED_ATTEMPT1_JOB_ID,
        )
        if argv[-1] == "--env-vars":
            return _command_result(repair, environment_payload)
        assert argv[-1] == "--batch-script"
        return _command_result(repair, batch_payload)

    receipt_map = {"schema_version": 1, "files": {}}
    predecessor, predecessor_sha = repair._seal_attempt2_predecessor(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        receipt_map,
        terminal,
        terminal_sha,
        retained_runner,
        _FakeLocks(),
    )
    return {
        "submission": submission,
        "contract": contract,
        "receipt_map": receipt_map,
        "terminal": terminal,
        "terminal_sha256": terminal_sha,
        "predecessor": predecessor,
        "predecessor_sha256": predecessor_sha,
    }


def test_attempt1_terminal_and_attempt2_predecessor_are_append_only_and_exact(
    repair, tmp_path, monkeypatch
):
    fixture = _synthetic_attempt1_terminal_and_predecessor(
        repair, tmp_path, monkeypatch
    )
    submission = fixture["submission"]
    terminal_path = (
        submission / "journal/REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json"
    )
    predecessor_path = (
        submission / "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
    )
    assert stat.S_IMODE(terminal_path.lstat().st_mode) == 0o444
    assert stat.S_IMODE(predecessor_path.lstat().st_mode) == 0o444
    assert fixture["predecessor"]["terminal_worker_failure_evidence_sha256"] == fixture[
        "terminal_sha256"
    ]
    assert fixture["predecessor"]["restart_failure_classification"] == {
        "schema_version": 1,
        "failed_guard": "restart_count_presence_and_zero_required",
        "retained_environment_variable_present": False,
        "runtime_absence_directly_recorded": False,
        "runtime_absence_inferred_from_exact_guard_and_log": True,
        "durable_attempt1_scontrol_observation_available": False,
        "successor_first_start_allowed_representations": ["absent", "0"],
        "successor_requires_fresh_scheduler_requeue": 0,
        "successor_requires_fresh_scheduler_restarts": 0,
    }
    assert repair._validated_attempt1_worker_failure_terminal(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        fixture["contract"],
    ) == (fixture["terminal"], fixture["terminal_sha256"])
    assert repair._validated_attempt2_predecessor(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        fixture["contract"],
        fixture["receipt_map"],
        _FakeLocks(),
    ) == (fixture["predecessor"], fixture["predecessor_sha256"])


def test_controller_and_worker_repair_schema_keysets_are_identical(report, repair):
    pairs = (
        (repair.ATTEMPT1_TERMINAL_KEYS, report.REPORT_REPAIR_ATTEMPT1_TERMINAL_KEYS),
        (repair.ATTEMPT2_PREDECESSOR_KEYS, report.REPORT_REPAIR_PREDECESSOR_KEYS),
        (repair.FAILURE_KEYS, report.REPORT_REPAIR_FAILURE_KEYS),
        (repair.SUBMIT_CALLING_KEYS, report.REPORT_REPAIR_SUBMIT_CALLING_KEYS),
        (repair.SUBMITTED_KEYS, report.REPORT_REPAIR_SUBMITTED_KEYS),
        (repair.AUTHORIZATION_KEYS, report.REPORT_REPAIR_AUTHORIZATION_KEYS),
        (repair.RELEASED_KEYS, report.REPORT_REPAIR_RELEASE_KEYS),
        (repair.RELEASE_CALLING_KEYS, report.REPORT_REPAIR_RELEASE_CALLING_KEYS),
        (repair.RELEASE_RESULT_KEYS, report.REPORT_REPAIR_RELEASE_RESULT_KEYS),
        (repair.COMPLETED_KEYS, report.REPORT_REPAIR_COMPLETED_KEYS),
    )
    assert all(controller == worker for controller, worker in pairs)
    assert (
        repair.ATTEMPT1_PUBLIC_ENVIRONMENT_EVIDENCE
        == report.ATTEMPT1_PUBLIC_ENVIRONMENT_EVIDENCE
    )
    assert (
        repair.SOURCE_AUTHORITY_V1_KEYS
        == report.REPORT_REPAIR_SOURCE_AUTHORITY_V1_KEYS
    )
    assert (
        repair.SOURCE_AUTHORITY_V2_KEYS
        == report.REPORT_REPAIR_SOURCE_AUTHORITY_V2_KEYS
    )
    assert (
        repair.ATTEMPT1_SOURCE_EVIDENCE_KEYS
        == report.REPORT_REPAIR_ATTEMPT1_SOURCE_EVIDENCE_KEYS
    )


@pytest.mark.parametrize(
    "mutation",
    ["extra", "release_chain", "census_environment", "accounting_environment"],
)
def test_worker_exactly_validates_attempt1_terminal_before_publication(
    report, repair, tmp_path, monkeypatch, mutation
):
    fixture = _synthetic_attempt1_terminal_and_predecessor(
        repair, tmp_path, monkeypatch
    )
    submission = fixture["submission"]
    terminal_path = (
        submission / "journal/REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json"
    )
    if mutation == "extra":
        terminal = copy.deepcopy(fixture["terminal"])
        terminal["unexpected"] = True
    elif mutation == "release_chain":
        terminal = copy.deepcopy(fixture["terminal"])
        terminal["release_attempts"][0]["result_sha256"] = "0" * 64
        terminal["release_attempts_sha256"] = report.stable_hash(
            terminal["release_attempts"]
        )
    elif mutation == "census_environment":
        terminal = copy.deepcopy(fixture["terminal"])
        terminal["post_release_census"]["rounds"][0]["raw"]["environment"][
            "UNEXPECTED"
        ] = "value"
        terminal["post_release_census_sha256"] = report.stable_hash(
            terminal["post_release_census"]
        )
    else:
        terminal = copy.deepcopy(fixture["terminal"])
        terminal["terminal_scheduler_observation"]["raw"]["environment"][
            "UNEXPECTED"
        ] = "value"
        terminal["terminal_scheduler_observation_sha256"] = report.stable_hash(
            terminal["terminal_scheduler_observation"]
        )
    terminal_path.chmod(0o600)
    terminal_path.unlink()
    digest = report.seal_json(terminal_path, terminal)
    with pytest.raises(report.ReportError):
        report._validated_attempt1_worker_failure_terminal(
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            fixture["contract"],
            relative_name="journal/REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
            expected_raw_sha256=digest,
            phase_binding=_FixtureRetainedBinding(),
        )


def test_worker_accepts_exact_controller_attempt1_terminal(
    report, repair, tmp_path, monkeypatch
):
    fixture = _synthetic_attempt1_terminal_and_predecessor(
        repair, tmp_path, monkeypatch
    )
    terminal, environment = report._validated_attempt1_worker_failure_terminal(
        fixture["submission"],
        repair.EXPECTED_SUBMISSION_SHA256,
        fixture["contract"],
        relative_name="journal/REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
        expected_raw_sha256=fixture["terminal_sha256"],
        phase_binding=_FixtureRetainedBinding(),
    )
    assert terminal == fixture["terminal"]
    assert environment == repair._scheduler_environment("/tmp/slurm.conf")


@pytest.mark.parametrize(
    "mutation",
    [
        "terminal_hash",
        "environment",
        "batch",
        "log",
        "restart_classification",
        "chain",
        "lock",
        "extra",
    ],
)
def test_attempt2_predecessor_rejects_coherent_field_forgery(
    repair, tmp_path, monkeypatch, mutation
):
    fixture = _synthetic_attempt1_terminal_and_predecessor(
        repair, tmp_path, monkeypatch
    )
    submission = fixture["submission"]
    path = submission / "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
    value = copy.deepcopy(fixture["predecessor"])
    if mutation == "terminal_hash":
        value["terminal_worker_failure_evidence_sha256"] = "0" * 64
    elif mutation == "environment":
        value["retained_environment_evidence"]["stdout"]["data"] = ""
        value["retained_environment_evidence_sha256"] = repair.stable_hash(
            value["retained_environment_evidence"]
        )
    elif mutation == "batch":
        value["retained_batch_script_evidence"]["stdout"]["data"] = ""
        value["retained_batch_script_evidence_sha256"] = repair.stable_hash(
            value["retained_batch_script_evidence"]
        )
    elif mutation == "log":
        value["failure_log"]["data"] = ""
        value["failure_log_sha256"] = repair.stable_hash(value["failure_log"])
    elif mutation == "restart_classification":
        value["restart_failure_classification"][
            "successor_first_start_allowed_representations"
        ] = ["0"]
    elif mutation == "chain":
        value["predecessor_chain"][
            "REPORT_REPAIR_0001_RELEASED.json"
        ] = "0" * 64
        value["predecessor_chain_sha256"] = repair.stable_hash(
            value["predecessor_chain"]
        )
    elif mutation == "lock":
        value["transaction_lock"]["inode"] = 999
    else:
        value["unexpected"] = True
    path.chmod(0o600)
    path.unlink()
    repair.seal_json(path, value)
    with pytest.raises(repair.RepairError):
        repair._validated_attempt2_predecessor(
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            fixture["contract"],
            fixture["receipt_map"],
            _FakeLocks(),
        )


@pytest.mark.parametrize("restart_count", [None, "0"])
def test_repair_worker_waits_beyond_sixty_seconds_for_authenticated_release(
    report, tmp_path, monkeypatch, restart_count
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    _authorization, digest = _wait_authorization(report, submission)
    monkeypatch.setenv("SLURM_JOB_ID", "444444")
    if restart_count is None:
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    else:
        monkeypatch.setenv("SLURM_RESTART_COUNT", restart_count)
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    clock = {"value": 0.0}

    def monotonic():
        return clock["value"]

    def sleep(seconds):
        clock["value"] += float(seconds)
        if clock["value"] >= 61.0 and not os.path.lexists(
            submission / "REPORT_REPAIR_0002_RELEASED.json"
        ):
            report.seal_json(
                submission / "REPORT_REPAIR_0002_RELEASED.json",
                {"schema_version": 1, "status": "fixture_release"},
            )

    report._wait_for_repair_release_evidence(
        submission,
        "a" * 64,
        attempt=2,
        authorization_sha256=digest,
        phase_binding=_WaitFixtureBinding(submission),
        monotonic=monotonic,
        sleep=sleep,
    )
    assert 61.0 <= clock["value"] < report.REPAIR_RELEASE_EVIDENCE_WAIT_SECONDS


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        (None, True),
        ("0", True),
        ("", False),
        ("00", False),
        ("1", False),
        ("-1", False),
        ("garbage", False),
        (0, False),
        (False, False),
    ],
)
def test_report_repair_first_start_restart_policy_is_exact(
    report, value, accepted
):
    assert report._repair_first_start_restart_count_is_valid(value) is accepted


def test_repair_worker_wait_timeout_and_terminal_prefix_are_no_publication(
    report, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    _authorization, digest = _wait_authorization(report, submission)
    monkeypatch.setenv("SLURM_JOB_ID", "444444")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    times = iter([0.0, 0.0, float(report.REPAIR_RELEASE_EVIDENCE_WAIT_SECONDS)])
    with pytest.raises(report.ReportError, match="wait exhausted"):
        report._wait_for_repair_release_evidence(
            submission,
            "a" * 64,
            attempt=2,
            authorization_sha256=digest,
            phase_binding=_WaitFixtureBinding(submission),
            monotonic=lambda: next(times),
            sleep=lambda _seconds: None,
        )
    assert not os.path.lexists(submission / "report")

    (submission / "REPORT_REPAIR_0002_TERMINAL_WORKER_FAILURE.json").write_bytes(
        b"terminal"
    )
    with pytest.raises(report.ReportError, match="cleanup/terminal authority"):
        report._wait_for_repair_release_evidence(
            submission,
            "a" * 64,
            attempt=2,
            authorization_sha256=digest,
            phase_binding=_WaitFixtureBinding(submission),
            monotonic=lambda: 0.0,
            sleep=lambda _seconds: None,
        )
    assert not os.path.lexists(submission / "report")


def test_cleanup_marker_precedes_scancel_and_crash_retry_is_gap_free(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    row = _repair_row(repair)
    pre = _census(repair, [row])
    post = _census(repair, [])
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    calls = []

    def crash_after_calling(argv, _cwd, _environment):
        calls.append(list(argv))
        raise RuntimeError("killpoint after cleanup calling")

    with pytest.raises(RuntimeError, match="killpoint"):
        _direct_cleanup(repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            pre,
            "fixture_ambiguity",
            _with_job_control(repair, submission, crash_after_calling),
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    assert (submission / "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json").is_file()
    assert (submission / "CALLING_REPORT_REPAIR_0002_SCANCEL_0000_0000.json").is_file()
    assert not (submission / "REPORT_REPAIR_0002_SCANCEL_RESULT_0000_0000.json").exists()
    assert calls == [["/usr/local/bin/scancel", row["job_id"]]]

    queued = [post, post, post]

    def no_mutation_runner(argv, _cwd, _environment):
        assert argv[0] == "/usr/local/bin/squeue"
        current = queued.pop(0)
        return _squeue_result(repair, current["settled_rows"])

    observed = repair._scheduler_census(
        submission,
        contract,
        no_mutation_runner,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    terminal = _direct_cleanup(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        observed,
        "ignored_on_resume",
        no_mutation_runner,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"
    result = json.loads(
        (submission / "REPORT_REPAIR_0002_SCANCEL_RESULT_0000_0000.json").read_text()
    )
    assert result["mode"] == "lost_response_reconciled_cancel_effect"
    assert len(terminal["cancel_attempts"]) == 1


def test_repair_census_rejects_parsed_rows_detached_from_raw(report):
    submission_sha = "a" * 64
    owner = "fixture"
    argv = [
        "/usr/local/bin/squeue",
        "--noheader",
        f"--user={owner}",
        "--format=%A|%j|%u|%T|%k|%r",
    ]
    expected_environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "SLURM_CONF": "/tmp/slurm.conf",
        "USER": owner,
        "LOGNAME": owner,
    }
    raw = {
        "argv": argv,
        "environment": expected_environment,
        "returncode": 0,
        "stdout": _stream(b""),
        "stderr": _stream(b""),
    }
    forged = {
        "job_id": "123",
        "job_name": f"exp23-launch8-{submission_sha[:16]}-report-repair-0002",
        "owner": owner,
        "state": "PENDING",
        "comment": f"treewm-exp23-report-repair:{submission_sha}:0001",
        "reason": "JobHeldUser",
    }
    census = {
        "schema_version": 1,
        "rounds": [
            {"round": index, "raw": raw, "relevant_rows": [forged]}
            for index in range(3)
        ],
        "settled_rows": [forged],
        "captured_at_utc": "2026-08-29T00:00:00Z",
    }
    with pytest.raises(report.ReportError, match="parsed/raw"):
        report._validated_repair_census(
            census,
            submission_sha256=submission_sha,
            expected_environment=expected_environment,
            label="fixture",
        )


def test_controller_census_requires_exactly_three_rounds(repair):
    census = _census(repair, [_repair_row(repair)])
    census["rounds"].append({**census["rounds"][-1], "round": 3})
    with pytest.raises(repair.RepairError, match="census envelope"):
        repair._validated_scheduler_census(census, _scheduler_contract())


def test_actual_submission_contract_git_provenance_projection_is_exact(repair):
    fixture = copy.deepcopy(ACTUAL_SUBMISSION_CONTRACT_IDENTITY)
    assert fixture["raw_sha256"] == repair.EXPECTED_SUBMISSION_SHA256
    assert repair.stable_hash(fixture["git_provenance"]) == fixture[
        "git_provenance_canonical_sha256"
    ]
    assert fixture["git_provenance"] == repair.EXPECTED_ORIGINAL_GIT_PROVENANCE
    assert repair._validated_original_git_provenance(
        fixture["git_provenance"]
    ) == fixture["git_provenance"]


def test_original_submission_validator_invokes_exact_git_provenance_guard(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "run/state/submission"
    submission.mkdir(parents=True)
    contract = {
        "campaign_id": repair.CAMPAIGN_ID,
        "submission_root": str(submission),
        "snapshot_inventory_sha256": repair.EXPECTED_SNAPSHOT_INVENTORY_SHA256,
        "package_protocol_sha256": repair.EXPECTED_ORIGINAL_PROTOCOL,
        "git_provenance": _actual_git_provenance(),
    }
    contract_info = SimpleNamespace(
        st_mode=stat.S_IFREG | 0o444,
        st_uid=os.getuid(),
        st_nlink=1,
    )
    monkeypatch.setattr(
        repair, "CANONICAL_PRODUCTION_SUBMISSION_ROOT", submission
    )
    monkeypatch.setattr(
        repair,
        "read_json",
        lambda _path, _label: (
            contract,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract_info,
        ),
    )
    observed = []

    class GuardObserved(RuntimeError):
        pass

    def observe(value):
        observed.append(copy.deepcopy(value))
        raise GuardObserved("exact provenance guard reached")

    monkeypatch.setattr(repair, "_validated_original_git_provenance", observe)
    with pytest.raises(GuardObserved, match="guard reached"):
        repair._validate_original_submission(
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            report_program=tmp_path / "report.py",
        )
    assert observed == [ACTUAL_SUBMISSION_CONTRACT_IDENTITY["git_provenance"]]


@pytest.mark.parametrize(
    "mutation", ["commit_substitution", "missing", "extra", "head_origin_mismatch"]
)
def test_original_submission_git_provenance_rejects_shape_and_alias_forgery(
    repair, mutation
):
    value = copy.deepcopy(ACTUAL_SUBMISSION_CONTRACT_IDENTITY["git_provenance"])
    if mutation == "commit_substitution":
        value["commit"] = value.pop("head")
    elif mutation == "missing":
        value.pop("branch")
    elif mutation == "extra":
        value["commit"] = repair.EXPECTED_ORIGINAL_SOURCE_COMMIT
    else:
        value["origin_main"] = "2" * 40
    with pytest.raises(repair.RepairError, match="git provenance"):
        repair._validated_original_git_provenance(value)


@pytest.mark.parametrize(
    "field",
    [
        "branch",
        "head",
        "object_format",
        "origin_main",
        "remote_origin",
        "worktree_status",
        "worktree_status_sha256",
    ],
)
def test_original_submission_git_provenance_rejects_wrong_field_types(
    repair, field
):
    value = copy.deepcopy(ACTUAL_SUBMISSION_CONTRACT_IDENTITY["git_provenance"])
    value[field] = True
    with pytest.raises(repair.RepairError, match="git provenance"):
        repair._validated_original_git_provenance(value)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("branch", "release"),
        ("head", "2" * 40),
        ("object_format", "sha256"),
        ("origin_main", "2" * 40),
        ("remote_origin", "https://github.com/ChristianLin0420/treewm.git"),
        ("worktree_status", "dirty"),
        ("worktree_status_sha256", "2" * 64),
    ],
)
def test_original_submission_git_provenance_rejects_value_drift(
    repair, field, replacement
):
    value = copy.deepcopy(ACTUAL_SUBMISSION_CONTRACT_IDENTITY["git_provenance"])
    value[field] = replacement
    with pytest.raises(repair.RepairError, match="git provenance"):
        repair._validated_original_git_provenance(value)


def legacy_source_snapshot_first_file_crash_is_removed(repair, tmp_path, monkeypatch):
    submission = tmp_path / "submission"
    submission.mkdir()
    files = {}
    for name in repair.SOURCE_NAMES:
        payload = (PACKAGE / name).read_bytes()
        files[name] = {
            "mode": 0o444,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    source = {
        "schema_version": 1,
        "repair_source_commit": "1" * 40,
        "repair_package_protocol_sha256": "2" * 64,
        "repair_source_files": files,
        "repair_source_files_sha256": repair.stable_hash(files),
    }
    real = repair._write_sealed_file
    calls = 0

    def fail_second(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("first-file crash fixture")
        return real(path, payload)

    monkeypatch.setattr(repair, "_write_sealed_file", fail_second)
    with pytest.raises(OSError, match="first-file"):
        repair._seal_repair_source_snapshot(submission, source, _FakeLocks())
    attempt = repair._repair_root(submission)
    assert not (attempt / "source").exists()
    assert not list(attempt.glob(".source.tmp.*"))


def legacy_source_snapshot_noreplace_race_preserves_competing_target(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    real_noreplace = repair._rename_directory_noreplace

    def race(staging, target, source_authority, locks, installation_method):
        target.mkdir(mode=0o700)
        (target / "competitor").write_text("preserve me", encoding="utf-8")
        return real_noreplace(
            staging, target, source_authority, locks, installation_method
        )

    monkeypatch.setattr(repair, "_rename_directory_noreplace", race)
    with pytest.raises(FileExistsError):
        repair._seal_repair_source_snapshot(submission, source, _FakeLocks())
    target = repair._repair_source_root(submission)
    assert (target / "competitor").read_text(encoding="utf-8") == "preserve me"
    assert not list(repair._repair_root(submission).glob(".source.tmp.*"))


def legacy_source_snapshot_atomically_carries_checkout_authority(repair, tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    source_root = repair._seal_repair_source_snapshot(
        submission, source, _FakeLocks()
    )
    assert repair._source_base_matches(repair._load_sealed_repair_source(source_root), source)
    assert {path.name for path in source_root.iterdir()} == {
        *repair.SOURCE_NAMES,
        repair.SOURCE_AUTHORITY_NAME,
    }
    authority = source_root / repair.SOURCE_AUTHORITY_NAME
    assert stat.S_IMODE(authority.lstat().st_mode) == 0o444
    sealed = json.loads(authority.read_text())
    assert repair._source_base_matches(sealed, source)
    assert repair._valid_installation_method(
        sealed["repair_source_installation_method"]
    )


class _ContextLocks(_FakeLocks):
    def __init__(self, _submission_root=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, _kind, _value, _traceback):
        return None


def _repair_source(repair):
    files = {}
    for name in repair.SOURCE_NAMES:
        payload = (PACKAGE / name).read_bytes()
        files[name] = {
            "mode": 0o444,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return {
        "schema_version": 1,
        "repair_source_commit": "1" * 40,
        "repair_package_protocol_sha256": (
            PACKAGE / "protocol.sha256"
        ).read_text(encoding="ascii").strip(),
        "repair_source_files": files,
        "repair_source_files_sha256": repair.stable_hash(files),
    }


def _repair_archive_source(repair, submission):
    _payload, evidence = repair._source_archive_payload(
        _repair_source(repair), submission
    )
    return evidence


def _pre_rename_source_staging(repair, submission, source):
    repair_parent = submission / "report-repair"
    attempt_root = repair._repair_root(submission)
    repair_parent.mkdir(mode=0o700, exist_ok=True)
    attempt_root.mkdir(mode=0o700)
    staging = attempt_root / ".source.tmp.1.1"
    staging.mkdir(mode=0o700)
    for name in repair.SOURCE_NAMES:
        repair._write_sealed_file(staging / name, (PACKAGE / name).read_bytes())
    sealed_source = repair._source_with_installation_method(
        source, repair.SOURCE_INSTALL_METHOD_PRIMARY
    )
    authority_payload = (
        json.dumps(sealed_source, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    repair._write_sealed_file(
        staging / repair.SOURCE_AUTHORITY_NAME, authority_payload
    )
    staging.chmod(0o555)
    return staging


def _empty_source_staging(repair, submission):
    repair_parent = submission / "report-repair"
    attempt_root = repair._repair_root(submission)
    repair_parent.mkdir(mode=0o700)
    attempt_root.mkdir(mode=0o700)
    staging = attempt_root / ".source.tmp.1.1"
    staging.mkdir(mode=0o700)
    return staging


def _staging_source_authority(repair, staging):
    return repair._load_sealed_repair_source(staging)


def legacy_source_snapshot_recovers_complete_0555_pre_rename_staging(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)

    source_root = repair._seal_repair_source_snapshot(
        submission, source, _FakeLocks()
    )

    assert not os.path.lexists(staging)
    assert repair._source_base_matches(repair._load_sealed_repair_source(source_root), source)
    assert stat.S_IMODE(source_root.lstat().st_mode) == 0o555


@pytest.mark.parametrize(
    "fallback_errno", [errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP]
)
def legacy_source_snapshot_locked_same_parent_fallback_for_capability_errno(
    repair, tmp_path, monkeypatch, fallback_errno
):
    submission = tmp_path / "submission"
    submission.mkdir()
    # This is the exact harmless live prefix left by a failed pre-scheduler
    # installation attempt: both owned state directories exist and are empty.
    (submission / "report-repair").mkdir(mode=0o700)
    (submission / "report-repair/attempt-0002").mkdir(mode=0o700)
    source = _repair_source(repair)
    real_rename = os.rename
    fallback_calls = []
    methods = []
    real_install = repair._rename_directory_noreplace

    def unsupported(_parent_descriptor, source_name, target_name):
        raise OSError(
            fallback_errno,
            os.strerror(fallback_errno),
            f"{source_name} -> {target_name}",
        )

    def observed_rename(source_name, target_name, **kwargs):
        fallback_calls.append((source_name, target_name, dict(kwargs)))
        return real_rename(source_name, target_name, **kwargs)

    def observed_install(*args):
        method = real_install(*args)
        methods.append(method)
        return method

    monkeypatch.setattr(repair, "_renameat2_noreplace", unsupported)
    monkeypatch.setattr(repair.os, "rename", observed_rename)
    monkeypatch.setattr(repair, "_rename_directory_noreplace", observed_install)

    source_root = repair._seal_repair_source_snapshot(
        submission, source, _FakeLocks()
    )

    assert methods == [
        f"locked_same_parent_rename_after_errno_{fallback_errno}"
    ]
    assert len(fallback_calls) == 1
    source_name, target_name, kwargs = fallback_calls[0]
    assert source_name.startswith(".source.tmp.")
    assert "/" not in source_name
    assert target_name == "source"
    assert kwargs["src_dir_fd"] == kwargs["dst_dir_fd"]
    assert type(kwargs["src_dir_fd"]) is int
    assert repair._source_base_matches(repair._load_sealed_repair_source(source_root), source)
    assert not list(repair._repair_root(submission).glob(".source.tmp.*"))
    assert (
        repair._seal_repair_source_snapshot(
            submission, source, _FakeLocks()
        )
        == source_root
    )


def legacy_source_snapshot_primary_noreplace_does_not_enter_fallback(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    target = repair._repair_source_root(submission)
    real_rename = os.rename

    def primary(parent_descriptor, source_name, target_name):
        real_rename(
            source_name,
            target_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )

    monkeypatch.setattr(repair, "_renameat2_noreplace", primary)
    monkeypatch.setattr(
        repair.os,
        "rename",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("primary success must not enter fallback")
        ),
    )

    method = repair._rename_directory_noreplace(
        staging,
        target,
        _staging_source_authority(repair, staging),
        _FakeLocks(),
        repair.SOURCE_INSTALL_METHOD_PRIMARY,
    )

    assert method == "renameat2_noreplace"
    assert not os.path.lexists(staging)
    assert repair._source_base_matches(repair._load_sealed_repair_source(target), source)


@pytest.mark.parametrize(
    "target_kind", ["directory", "file", "symlink", "dangling", "fifo"]
)
def legacy_source_snapshot_fallback_never_replaces_injected_target(
    repair, tmp_path, monkeypatch, target_kind
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    target = repair._repair_source_root(submission)
    external = tmp_path / "external"
    external.mkdir()

    if target_kind == "directory":
        target.mkdir(mode=0o700)
        (target / "competitor").write_text("preserve", encoding="utf-8")
    elif target_kind == "file":
        target.write_text("preserve", encoding="utf-8")
    elif target_kind == "symlink":
        target.symlink_to(external, target_is_directory=True)
    elif target_kind == "dangling":
        target.symlink_to(tmp_path / "absent", target_is_directory=True)
    else:
        os.mkfifo(target, 0o600)

    def forbidden_fallback(*_args, **_kwargs):
        raise AssertionError("fallback must not replace an injected target")

    monkeypatch.setattr(repair.os, "rename", forbidden_fallback)

    with pytest.raises(FileExistsError):
        repair._rename_directory_noreplace(
            staging,
            target,
            _staging_source_authority(repair, staging),
            _FakeLocks(),
            f"{repair.SOURCE_INSTALL_METHOD_FALLBACK_PREFIX}{errno.EINVAL}",
        )

    assert os.path.lexists(target)
    assert os.path.lexists(staging)


@pytest.mark.parametrize(
    "unsupported_errno", [errno.EEXIST, errno.EPERM, errno.EIO, errno.EXDEV]
)
def legacy_source_snapshot_noncapability_errno_never_uses_fallback(
    repair, tmp_path, monkeypatch, unsupported_errno
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    target = repair._repair_source_root(submission)

    def rejected(_parent_fd, _source_name, _target_name):
        raise OSError(unsupported_errno, os.strerror(unsupported_errno))

    def forbidden_fallback(*_args, **_kwargs):
        raise AssertionError("unsupported errno must not use fallback")

    monkeypatch.setattr(repair, "_renameat2_noreplace", rejected)
    monkeypatch.setattr(repair.os, "rename", forbidden_fallback)

    with pytest.raises(OSError) as raised:
        repair._rename_directory_noreplace(
            staging,
            target,
            _staging_source_authority(repair, staging),
            _FakeLocks(),
            repair.SOURCE_INSTALL_METHOD_PRIMARY,
        )
    assert raised.value.errno == unsupported_errno
    assert os.path.lexists(staging)
    assert not os.path.lexists(target)


def legacy_source_snapshot_fallback_rejects_injected_attempt_namespace(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    target = repair._repair_source_root(submission)
    unexpected = staging.parent / "unexpected"

    unexpected.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(
        repair.os,
        "rename",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("injected namespace must block fallback")
        ),
    )

    with pytest.raises(repair.RepairError, match="namespace"):
        repair._rename_directory_noreplace(
            staging,
            target,
            _staging_source_authority(repair, staging),
            _FakeLocks(),
            f"{repair.SOURCE_INSTALL_METHOD_FALLBACK_PREFIX}{errno.EINVAL}",
        )
    assert unexpected.read_text(encoding="utf-8") == "preserve"
    assert os.path.lexists(staging)
    assert not os.path.lexists(target)


@pytest.mark.parametrize("injection", ["target", "unknown"])
def legacy_source_snapshot_fallback_rechecks_namespace_after_validation(
    repair, tmp_path, monkeypatch, injection
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    target = repair._repair_source_root(submission)
    unexpected = staging.parent / "unexpected"
    real_validate = repair._validate_sealed_repair_source
    validations = 0

    def unsupported(*_args):
        raise OSError(errno.EINVAL, os.strerror(errno.EINVAL))

    def inject_on_second_validation(root, authority):
        nonlocal validations
        real_validate(root, authority)
        validations += 1
        if validations == 2:
            if injection == "target":
                target.mkdir(mode=0o700)
            else:
                unexpected.write_text("preserve", encoding="utf-8")

    monkeypatch.setattr(repair, "_renameat2_noreplace", unsupported)
    monkeypatch.setattr(
        repair, "_validate_sealed_repair_source", inject_on_second_validation
    )
    monkeypatch.setattr(
        repair.os,
        "rename",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("post-validation injection must block fallback")
        ),
    )

    with pytest.raises((FileExistsError, repair.RepairError)):
        repair._rename_directory_noreplace(
            staging,
            target,
            _staging_source_authority(repair, staging),
            _FakeLocks(),
            f"{repair.SOURCE_INSTALL_METHOD_FALLBACK_PREFIX}{errno.EINVAL}",
        )
    assert os.path.lexists(target if injection == "target" else unexpected)
    assert os.path.lexists(staging)


def legacy_source_snapshot_postvalidation_identity_drift_is_rejected(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    target = repair._repair_source_root(submission)
    source_authority = _staging_source_authority(repair, staging)
    real_validate = repair._validate_sealed_repair_source
    validations = 0

    def unsupported(*_args):
        raise OSError(errno.EINVAL, os.strerror(errno.EINVAL))

    def drift_after_postinstall_validation(root, authority):
        nonlocal validations
        real_validate(root, authority)
        validations += 1
        if validations == 3:
            root.chmod(0o700)

    monkeypatch.setattr(repair, "_renameat2_noreplace", unsupported)
    monkeypatch.setattr(
        repair,
        "_validate_sealed_repair_source",
        drift_after_postinstall_validation,
    )

    with pytest.raises(repair.RepairError, match="identity/namespace"):
        repair._rename_directory_noreplace(
            staging,
            target,
            source_authority,
            _FakeLocks(),
            f"{repair.SOURCE_INSTALL_METHOD_FALLBACK_PREFIX}{errno.EINVAL}",
        )
    assert not os.path.lexists(staging)
    assert stat.S_IMODE(target.lstat().st_mode) == 0o700


def legacy_source_snapshot_fallback_rejects_lock_binding_drift(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    target = repair._repair_source_root(submission)

    class DriftingLocks(_FakeLocks):
        def __init__(self):
            self.calls = 0

        def bindings(self):
            self.calls += 1
            transaction, report = super().bindings()
            if self.calls >= 4:
                transaction = {**transaction, "inode": 999}
            return transaction, report

    def unsupported(*_args):
        raise OSError(errno.EINVAL, os.strerror(errno.EINVAL))

    monkeypatch.setattr(repair, "_renameat2_noreplace", unsupported)
    monkeypatch.setattr(
        repair.os,
        "rename",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("lock drift must block fallback")
        ),
    )

    with pytest.raises(repair.RepairError, match="authority binding"):
        repair._rename_directory_noreplace(
            staging,
            target,
            _staging_source_authority(repair, staging),
            DriftingLocks(),
            f"{repair.SOURCE_INSTALL_METHOD_FALLBACK_PREFIX}{errno.EINVAL}",
        )
    assert os.path.lexists(staging)
    assert not os.path.lexists(target)


def legacy_source_snapshot_fallback_rejects_named_parent_inode_drift(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    target = repair._repair_source_root(submission)
    attempt_root = staging.parent
    moved_attempt = tmp_path / "moved-attempt"
    real_rename = os.rename
    real_validate = repair._validate_sealed_repair_source
    source_authority = _staging_source_authority(repair, staging)
    injected = False

    def replace_parent_during_validation(root, authority):
        nonlocal injected
        real_validate(root, authority)
        if not injected:
            injected = True
            real_rename(attempt_root, moved_attempt)
            attempt_root.mkdir(mode=0o700)

    monkeypatch.setattr(
        repair, "_validate_sealed_repair_source", replace_parent_during_validation
    )
    monkeypatch.setattr(
        repair.os,
        "rename",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("parent drift must block fallback")
        ),
    )

    with pytest.raises(repair.RepairError, match="authority binding"):
        repair._rename_directory_noreplace(
            staging,
            target,
            source_authority,
            _FakeLocks(),
            f"{repair.SOURCE_INSTALL_METHOD_FALLBACK_PREFIX}{errno.EINVAL}",
        )
    assert os.path.lexists(moved_attempt / staging.name)
    assert not os.path.lexists(target)


def legacy_source_snapshot_postrename_lock_drift_is_rejected(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    target = repair._repair_source_root(submission)
    real_rename = os.rename

    class DriftingLocks(_FakeLocks):
        def __init__(self):
            self.calls = 0

        def bindings(self):
            self.calls += 1
            transaction, report = super().bindings()
            if self.calls >= 4:
                report = {**report, "inode": 999}
            return transaction, report

    def primary(parent_descriptor, source_name, target_name):
        real_rename(
            source_name,
            target_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )

    monkeypatch.setattr(repair, "_renameat2_noreplace", primary)

    with pytest.raises(repair.RepairError, match="authority binding"):
        repair._rename_directory_noreplace(
            staging,
            target,
            _staging_source_authority(repair, staging),
            DriftingLocks(),
            repair.SOURCE_INSTALL_METHOD_PRIMARY,
        )
    assert not os.path.lexists(staging)
    assert repair._source_base_matches(repair._load_sealed_repair_source(target), source)


@pytest.mark.parametrize("fake_mode", ["new_inode", "recreate_staging"])
def legacy_source_snapshot_fallback_rejects_nonatomic_or_split_poststate(
    repair, tmp_path, monkeypatch, fake_mode
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    target = repair._repair_source_root(submission)
    real_rename = os.rename

    def unsupported(*_args):
        raise OSError(errno.EINVAL, os.strerror(errno.EINVAL))

    def fake_rename(source_name, target_name, *, src_dir_fd, dst_dir_fd):
        assert src_dir_fd == dst_dir_fd
        if fake_mode == "new_inode":
            shutil.copytree(staging, target, copy_function=shutil.copy2)
            staging.chmod(0o700)
            shutil.rmtree(staging)
        else:
            real_rename(
                source_name,
                target_name,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
            )
            staging.mkdir(mode=0o555)

    monkeypatch.setattr(repair, "_renameat2_noreplace", unsupported)
    monkeypatch.setattr(repair.os, "rename", fake_rename)

    with pytest.raises(repair.RepairError, match="identity/namespace|remains"):
        repair._rename_directory_noreplace(
            staging,
            target,
            _staging_source_authority(repair, staging),
            _FakeLocks(),
            f"{repair.SOURCE_INSTALL_METHOD_FALLBACK_PREFIX}{errno.EINVAL}",
        )
    if fake_mode == "new_inode":
        assert not os.path.lexists(staging)
        assert os.path.lexists(target)
    else:
        assert os.path.lexists(staging)


def legacy_source_snapshot_install_parent_fsync_failure_propagates(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    target = repair._repair_source_root(submission)
    real_rename = os.rename

    def unsupported(*_args):
        raise OSError(errno.EINVAL, os.strerror(errno.EINVAL))

    monkeypatch.setattr(repair, "_renameat2_noreplace", unsupported)
    monkeypatch.setattr(repair.os, "rename", real_rename)
    monkeypatch.setattr(
        repair.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError("fsync killpoint")),
    )

    with pytest.raises(OSError, match="fsync killpoint"):
        repair._rename_directory_noreplace(
            staging,
            target,
            _staging_source_authority(repair, staging),
            _FakeLocks(),
            f"{repair.SOURCE_INSTALL_METHOD_FALLBACK_PREFIX}{errno.EINVAL}",
        )
    assert not os.path.lexists(staging)
    assert repair._source_base_matches(repair._load_sealed_repair_source(target), source)


@pytest.mark.parametrize("path_case", ["cross_parent", "wrong_target"])
def legacy_source_snapshot_install_rejects_wrong_path_shape(
    repair, tmp_path, path_case
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    if path_case == "cross_parent":
        other = tmp_path / "other"
        other.mkdir(mode=0o700)
        target = other / "source"
        match = "parents"
    else:
        target = staging.parent / "not-source"
        match = "names"

    with pytest.raises(repair.RepairError, match=match):
        repair._rename_directory_noreplace(
            staging,
            target,
            _staging_source_authority(repair, staging),
            _FakeLocks(),
            repair.SOURCE_INSTALL_METHOD_PRIMARY,
        )
    assert os.path.lexists(staging)
    assert not os.path.lexists(target)


@pytest.mark.parametrize(
    "forgery",
    [
        "extra",
        "missing",
        "wrong",
        "authority_v1",
        "authority_extra",
        "authority_method",
        "authority_nonpretty",
    ],
)
def legacy_source_snapshot_rejects_forged_0555_pre_rename_staging(
    repair, tmp_path, forgery
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    staging.chmod(0o700)
    if forgery == "extra":
        repair._write_sealed_file(staging / "unexpected", b"forged\n")
    elif forgery == "missing":
        (staging / repair.SOURCE_NAMES[0]).unlink()
    elif forgery == "wrong":
        target = staging / repair.SOURCE_NAMES[0]
        target.unlink()
        repair._write_sealed_file(target, b"forged\n")
    elif forgery == "authority_v1":
        target = staging / repair.SOURCE_AUTHORITY_NAME
        target.unlink()
        repair._write_sealed_file(target, b'{"schema_version":1}\n')
    else:
        target = staging / repair.SOURCE_AUTHORITY_NAME
        value = json.loads(target.read_text(encoding="utf-8"))
        target.unlink()
        if forgery == "authority_extra":
            value["unexpected"] = True
        elif forgery == "authority_method":
            value["repair_source_installation_method"] = "unbound_replace"
        else:
            assert forgery == "authority_nonpretty"
        payload = (
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            if forgery == "authority_nonpretty"
            else json.dumps(
                value, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        repair._write_sealed_file(target, payload)
    staging.chmod(0o555)

    with pytest.raises(repair.RepairError, match="repair source staging|sealed repair"):
        repair._seal_repair_source_snapshot(submission, source, _FakeLocks())

    assert os.path.lexists(staging)
    assert stat.S_IMODE(staging.lstat().st_mode) == 0o555
    assert not os.path.lexists(repair._repair_source_root(submission))


@pytest.mark.parametrize("partial_name", ["source", "authority"])
def legacy_source_snapshot_removes_0600_partial_pre_seal_staging(
    repair, tmp_path, partial_name
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _empty_source_staging(repair, submission)
    if partial_name == "source":
        target = staging / repair.SOURCE_NAMES[0]
    else:
        for name in repair.SOURCE_NAMES:
            repair._write_sealed_file(staging / name, (PACKAGE / name).read_bytes())
        target = staging / repair.SOURCE_AUTHORITY_NAME
    target.write_bytes(b"partial")
    target.chmod(0o600)

    source_root = repair._seal_repair_source_snapshot(
        submission, source, _FakeLocks()
    )

    assert not os.path.lexists(staging)
    assert repair._source_base_matches(repair._load_sealed_repair_source(source_root), source)


@pytest.mark.parametrize(
    "killpoint", ["authority_invalidated", "during_chmod_pass", "after_first_unlink"]
)
def legacy_source_snapshot_cleanup_two_phase_killpoints_are_restartable(
    repair, tmp_path, killpoint
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    staging.chmod(0o700)
    authority = staging / repair.SOURCE_AUTHORITY_NAME
    authority.chmod(0o600)
    if killpoint in {"during_chmod_pass", "after_first_unlink"}:
        (staging / repair.SOURCE_NAMES[0]).chmod(0o600)
    if killpoint == "after_first_unlink":
        for name in repair.SOURCE_NAMES[1:]:
            (staging / name).chmod(0o600)
        (staging / repair.SOURCE_NAMES[0]).unlink()

    source_root = repair._seal_repair_source_snapshot(
        submission, source, _FakeLocks()
    )

    assert not os.path.lexists(staging)
    assert repair._source_base_matches(repair._load_sealed_repair_source(source_root), source)
    assert (
        repair._seal_repair_source_snapshot(submission, source, _FakeLocks())
        == source_root
    )


def legacy_source_snapshot_rejects_missing_0700_coverage_with_completed_authority(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    staging.chmod(0o700)
    (staging / repair.SOURCE_NAMES[0]).unlink()

    with pytest.raises(repair.RepairError, match="authority coverage"):
        repair._seal_repair_source_snapshot(submission, source, _FakeLocks())

    assert stat.S_IMODE(
        (staging / repair.SOURCE_AUTHORITY_NAME).lstat().st_mode
    ) == 0o444
    assert not os.path.lexists(repair._repair_source_root(submission))


def legacy_source_snapshot_rejects_partial_entry_with_completed_authority_untouched(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    staging.chmod(0o700)
    partial = staging / repair.SOURCE_NAMES[0]
    partial.chmod(0o600)
    before = {
        path.name: (path.read_bytes(), stat.S_IMODE(path.lstat().st_mode))
        for path in staging.iterdir()
    }

    with pytest.raises(repair.RepairError, match="partial source entries"):
        repair._seal_repair_source_snapshot(submission, source, _FakeLocks())

    after = {
        path.name: (path.read_bytes(), stat.S_IMODE(path.lstat().st_mode))
        for path in staging.iterdir()
    }
    assert after == before
    assert not os.path.lexists(repair._repair_source_root(submission))


@pytest.mark.parametrize(
    "forgery", ["unknown", "hardlink", "symlink", "special", "mode"]
)
def legacy_source_snapshot_rejects_unsafe_0700_partial_staging(
    repair, tmp_path, forgery
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _empty_source_staging(repair, submission)
    target = staging / repair.SOURCE_NAMES[0]
    if forgery == "unknown":
        target = staging / "unexpected"
        repair._write_sealed_file(target, b"forged\n")
    elif forgery == "hardlink":
        external = tmp_path / "external"
        external.write_bytes(b"forged\n")
        os.link(external, target)
    elif forgery == "symlink":
        target.symlink_to(tmp_path / "external")
    elif forgery == "special":
        os.mkfifo(target, 0o600)
    else:
        target.write_bytes(b"forged\n")
        target.chmod(0o640)

    with pytest.raises(repair.RepairError, match="repair source staging"):
        repair._seal_repair_source_snapshot(submission, source, _FakeLocks())

    assert os.path.lexists(staging)
    assert stat.S_IMODE(staging.lstat().st_mode) == 0o700
    assert not os.path.lexists(repair._repair_source_root(submission))


def _configure_minimal_controller(repair, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    submission = tmp_path / "submission"
    repo.mkdir()
    (submission / "journal").mkdir(parents=True)
    submission.chmod(0o700)
    (submission / "logs").mkdir()
    contract = {
        "snapshot_root": str(submission / "source-snapshot/repo"),
        "snapshot_inventory_sha256": "3" * 64,
        "package_protocol_sha256": repair.EXPECTED_ORIGINAL_PROTOCOL,
        "git_provenance": _actual_git_provenance(),
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"},
    }
    repair.seal_json(submission / "SUBMISSION_CONTRACT.json", contract)
    _seed_retained_authority_fixture(repair, submission)
    repair_parent = submission / "report-repair"
    repair_parent.mkdir(mode=0o700)
    attempt1 = repair_parent / "attempt-0001"
    attempt1.mkdir(mode=0o700)
    source1 = attempt1 / "source"
    source1.mkdir(mode=0o700)
    source1.chmod(0o555)
    receipt = {"report_job_id": repair.EXPECTED_ORIGINAL_REPORT_JOB_ID}
    receipt_map = {"schema_version": 1, "files": {}}
    source = _repair_source(repair)
    failure = {"schema_version": 1, "status": "fixture_failure"}
    # Controller state-machine tests below isolate attempt-2 transitions.  A
    # dedicated hostile block exercises the real immutable attempt-1 chain;
    # here we install its mandatory predecessor namespace and use exact stable
    # validator seams so older release/cleanup tests do not manufacture live
    # accounting or mutate the predecessor generation.
    predecessor_files = {}
    for name in repair.EXPECTED_ATTEMPT1_CHAIN_SHA256:
        value = failure if name == "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json" else {
            "schema_version": 1,
            "status": f"fixture_{name}",
        }
        digest = repair.seal_json(submission / "journal" / name, value)
        predecessor_files[name] = digest
    terminal_value = {
        "schema_version": 1,
        "status": "report_repair_terminal_worker_failure",
        "attempt": 1,
    }
    terminal_sha = repair.seal_json(
        submission / "journal/REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
        terminal_value,
    )
    predecessor_value = _phase_fixture_value(
        repair, "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
    )
    predecessor_sha = repair.seal_json(
        submission / "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json",
        predecessor_value,
    )
    monkeypatch.setattr(repair, "REPOSITORY_ROOT", repo)
    monkeypatch.setattr(
        repair, "CANONICAL_PRODUCTION_SUBMISSION_ROOT", submission
    )
    monkeypatch.setattr(repair, "_RepairLocks", _ContextLocks)
    monkeypatch.setattr(
        repair,
        "_validate_original_submission",
        lambda *_args, **_kwargs: (
            contract,
            receipt,
            receipt_map,
            repair._expected_reassembly(),
        ),
    )
    monkeypatch.setattr(
        repair,
        "_validated_attempt1_chain",
        lambda _submission: {
            "schema_version": 1,
            "files": dict(predecessor_files),
            "source": {"schema_version": 1},
        },
    )
    monkeypatch.setattr(
        repair,
        "_validated_attempt1_worker_failure_terminal",
        lambda *_args, **_kwargs: (dict(terminal_value), terminal_sha),
    )
    monkeypatch.setattr(
        repair,
        "_validated_attempt2_predecessor",
        lambda *_args, **_kwargs: (dict(predecessor_value), predecessor_sha),
    )
    monkeypatch.setattr(repair, "_verified_live_repair_source", lambda _root: source)
    monkeypatch.setattr(
        repair, "_require_repair_filesystem_namespace", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        repair,
        "_directory_install_method_probe",
        lambda *_args, **_kwargs: repair.DIRECT_FINAL_INSTALL_METHOD,
    )
    monkeypatch.setattr(
        repair,
        "_job_control_observation",
        lambda *_args, **_kwargs: {
            "schema_version": 1,
            "captured_at_utc": repair._utc_now(),
            "fixture": "held_no_requeue",
        },
    )
    monkeypatch.setattr(
        repair,
        "_validated_job_control_observation",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        repair,
        "_build_failure_evidence",
        lambda *_args, **_kwargs: failure,
    )
    monkeypatch.setattr(
        repair,
        "_validate_failure_evidence",
        lambda value, **_kwargs: dict(value),
    )
    return repo, submission, contract, source


def _with_empty_pre_submit_census(repair, delegate):
    """Serve the mandatory three empty rounds before delegating to a fixture."""

    remaining = 3

    def wrapped(argv, cwd, environment):
        nonlocal remaining
        if remaining:
            assert argv[0] == "/usr/local/bin/squeue"
            remaining -= 1
            return _squeue_result(repair, [])
        return delegate(argv, cwd, environment)

    return wrapped


@pytest.mark.parametrize("prefix", ["empty_attempt", "complete_staging"])
def test_source_only_prefix_is_not_recovery_authority_without_explicit_submit(
    repair, tmp_path, monkeypatch, prefix
):
    repo, submission, _contract, source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )
    (submission / "journal/REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json").unlink()
    (submission / "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json").unlink()
    monkeypatch.setattr(
        repair, "_validated_attempt1_worker_failure_terminal", lambda *_args: None
    )
    monkeypatch.setattr(
        repair, "_validated_attempt2_predecessor", lambda *_args: None
    )
    if prefix == "empty_attempt":
        (submission / "report-repair/attempt-0002").mkdir(mode=0o700)
    else:
        staging = _pre_rename_source_staging(repair, submission, source)

    def forbidden_source_creation(*_args, **_kwargs):
        raise AssertionError(
            "uncommitted source prefix must not create recovery authority"
        )

    def forbidden_scheduler(*_args, **_kwargs):
        raise AssertionError("source-only recovery must not call scheduler")

    monkeypatch.setattr(
        repair, "_verified_live_repair_source", forbidden_source_creation
    )
    monkeypatch.setattr(
        repair, "_seal_repair_source_snapshot", forbidden_source_creation
    )

    with pytest.raises(
        repair.RepairError,
        match="cannot create the attempt1 terminal boundary|attempt namespace differs",
    ):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=False,
            runner=forbidden_scheduler,
            sleep=lambda _seconds: None,
        )

    assert not (
        submission / "CALLING_REPORT_REPAIR_0002_SUBMIT.json"
    ).exists()
    assert not repair._repair_source_root(submission).exists()
    if prefix == "complete_staging":
        assert os.path.lexists(staging)


def test_alternate_checkout_keeps_literal_production_submission_root(tmp_path):
    checkout = tmp_path / "clean-alternate-checkout"
    package = checkout / PACKAGE.relative_to(PACKAGE.parents[1])
    package.mkdir(parents=True)
    controller = package / "report_repair.py"
    controller.write_bytes((PACKAGE / "report_repair.py").read_bytes())
    spec = importlib.util.spec_from_file_location(
        "exp23_repair_alternate_checkout", controller
    )
    assert spec is not None and spec.loader is not None
    alternate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = alternate
    spec.loader.exec_module(alternate)
    expected = Path(
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch8/"
        "state/submission"
    )
    assert alternate.REPOSITORY_ROOT == checkout
    assert alternate.CANONICAL_PRODUCTION_SUBMISSION_ROOT == expected
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    assert (
        manifest["launch_contract"]["terminal_report_repair"][
            "submission_root"
        ]
        == str(alternate.CANONICAL_PRODUCTION_SUBMISSION_ROOT)
    )
    assert alternate.CANONICAL_PRODUCTION_SUBMISSION_ROOT != checkout / (
        "outputs/treewm-executable-prefix-repair-pilot-v1-launch8/"
        "state/submission"
    )


@pytest.mark.parametrize(
    ("repo_suffix", "submission_suffix"),
    [
        ("/", ""),
        ("", "/"),
        ("//", ""),
        ("/./", ""),
        ("/temporary/../", ""),
        ("", "//"),
        ("", "/./"),
        ("", "/temporary/../"),
    ],
)
def test_cli_rejects_nonliteral_root_spellings_before_path_normalization(
    repair, capsys, repo_suffix, submission_suffix
):
    repo = str(repair.REPOSITORY_ROOT) + repo_suffix
    submission = str(repair.CANONICAL_PRODUCTION_SUBMISSION_ROOT) + submission_suffix
    assert (
        repair.main(
            [
                "--describe",
                "--repo-root",
                repo,
                "--submission-root",
                submission,
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "not a canonical absolute path" in captured.err


def test_execute_accepts_only_canonical_production_root_independent_of_checkout(
    repair, tmp_path, monkeypatch
):
    checkout = tmp_path / "clean-checkout"
    production = tmp_path / "production" / "state" / "submission"
    clone_derived = checkout / (
        "outputs/treewm-executable-prefix-repair-pilot-v1-launch8/"
        "state/submission"
    )
    other = tmp_path / "other" / "state" / "submission"
    for path in (checkout, production, clone_derived, other):
        path.mkdir(parents=True, exist_ok=True)
    production_alias = tmp_path / "submission-alias"
    production_alias.symlink_to(production, target_is_directory=True)
    checkout_alias = tmp_path / "checkout-alias"
    checkout_alias.symlink_to(checkout, target_is_directory=True)

    monkeypatch.setattr(repair, "REPOSITORY_ROOT", checkout)
    monkeypatch.setattr(
        repair, "CANONICAL_PRODUCTION_SUBMISSION_ROOT", production
    )
    reached = []

    class BoundaryReached(RuntimeError):
        pass

    class StopAtLocks:
        def __init__(self, submission_root):
            reached.append(submission_root)

        def __enter__(self):
            raise BoundaryReached("canonical production lock boundary")

        def __exit__(self, _kind, _value, _traceback):
            return None

    monkeypatch.setattr(repair, "_RepairLocks", StopAtLocks)
    with pytest.raises(BoundaryReached, match="canonical production"):
        repair.execute_report_repair(
            checkout,
            production,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=False,
        )
    assert reached == [production]

    invalid_pairs = [
        (checkout, clone_derived),
        (checkout, other),
        (checkout, production_alias),
        (checkout_alias, production),
        (checkout, Path("//") / production.relative_to("/")),
    ]
    for source_root, submission_root in invalid_pairs:
        with pytest.raises(
            repair.RepairError,
            match="root differs|canonical absolute path|symlinked or noncanonical",
        ):
            repair.execute_report_repair(
                source_root,
                submission_root,
                repair.EXPECTED_SUBMISSION_SHA256,
                allow_initial_submission=False,
            )
    assert reached == [production]


@pytest.mark.parametrize("dirty", [" M report_repair.py\n", "?? untracked\n"])
def test_live_repair_source_requires_clean_tracked_origin_main(
    repair, tmp_path, monkeypatch, dirty
):
    checkout = tmp_path / "clean-checkout"
    package = checkout / repair.PACKAGE_RELATIVE
    package.mkdir(parents=True)
    protocol = "a" * 64
    for name in repair.SOURCE_NAMES:
        payload = f"fixture {name}\n".encode("ascii")
        if name == "protocol.sha256":
            payload = f"{protocol}\n".encode("ascii")
        path = package / name
        path.write_bytes(payload)
        path.chmod(0o644)

    class CampaignFixture:
        @staticmethod
        def load_contract(root):
            assert root == checkout

        @staticmethod
        def verify_protocol_lock(root):
            assert root == package
            return protocol

    state = {"status": ""}
    commit = "1" * 40

    def git_output(argv, root):
        assert root == checkout
        if argv[-1] == "HEAD":
            return f"{commit}\n"
        if argv[-1] == "origin/main":
            return f"{commit}\n"
        assert "status" in argv
        return state["status"]

    monkeypatch.setattr(repair, "PACKAGE_DIR", package)
    monkeypatch.setattr(repair, "REPOSITORY_ROOT", checkout)
    monkeypatch.setattr(repair, "_load_module", lambda *_args: CampaignFixture)
    monkeypatch.setattr(repair, "_git_output", git_output)
    source = repair._verified_live_repair_source(checkout)
    assert source["repair_source_commit"] == commit
    assert set(source["repair_source_files"]) == set(repair.SOURCE_NAMES)

    state["status"] = dirty
    with pytest.raises(
        repair.RepairError, match="not a clean origin/main commit"
    ):
        repair._verified_live_repair_source(checkout)


def test_full_held_submit_authorize_release_order_is_gap_free(
    repair, tmp_path, monkeypatch
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )

    calls = []
    squeue_count = 0

    def runner(argv, _cwd, _environment):
        nonlocal squeue_count
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/sbatch":
            journal = submission / "journal"
            assert (journal / "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json").is_file()
            assert (submission / "CALLING_REPORT_REPAIR_0002_SUBMIT.json").is_file()
            assert not (submission / "REPORT_REPAIR_0002_SUBMITTED.json").exists()
            return _command_result(repair, b"444444\n")
        if argv[0] == "/usr/local/bin/squeue":
            squeue_count += 1
            if squeue_count <= 3:
                return _squeue_result(repair, [])
            row = _repair_row(
                repair,
                state=("PENDING" if squeue_count <= 9 else "RUNNING"),
                reason=("JobHeldUser" if squeue_count <= 9 else "None"),
            )
            return _squeue_result(repair, [row])
        if argv[0] == "/usr/local/bin/scontrol":
            journal = submission / "journal"
            assert (submission / "REPORT_REPAIR_0002_AUTHORIZED.json").is_file()
            assert (submission / "CALLING_REPORT_REPAIR_0002_RELEASE_0000.json").is_file()
            assert not (submission / "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json").exists()
            return _command_result(repair)
        raise AssertionError(argv)

    result = repair.execute_report_repair(
        repo,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        allow_initial_submission=True,
        runner=runner,
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "report_repair_released_for_publication"
    assert result["repair_report_job_id"] == "444444"
    journal = submission / "journal"
    assert (submission / "REPORT_REPAIR_0002_SUBMITTED.json").is_file()
    assert (submission / "REPORT_REPAIR_0002_AUTHORIZED.json").is_file()
    assert (submission / "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json").is_file()
    assert (submission / "REPORT_REPAIR_0002_RELEASED.json").is_file()
    assert [call[0] for call in calls] == [
        *(["/usr/local/bin/squeue"] * 3),
        "/usr/local/bin/sbatch",
        *(["/usr/local/bin/squeue"] * 3),
        *(["/usr/local/bin/squeue"] * 3),
        "/usr/local/bin/scontrol",
        *(["/usr/local/bin/squeue"] * 3),
    ]

    # SUBMIT CALLING is the creator-live handoff for the source archive.  A
    # byte-identical replacement inode on restart is not the created source.
    source_archive = submission / repair.SOURCE_ARCHIVE_NAME
    source_original = tmp_path / "source-archive.original"
    source_payload = source_archive.read_bytes()
    source_inode = source_archive.lstat().st_ino
    source_archive.rename(source_original)
    source_archive.write_bytes(source_payload)
    source_archive.chmod(0o444)
    assert source_archive.lstat().st_ino != source_inode
    calling_value = json.loads(
        (submission / "CALLING_REPORT_REPAIR_0002_SUBMIT.json").read_text()
    )
    assert calling_value["repair_source_archive_file_identity"]["inode"] == (
        source_inode
    )
    assert calling_value["repair_source_archive_file_identity"]["inode"] != (
        source_archive.lstat().st_ino
    )
    calls_before_clone = len(calls)
    try:
        with pytest.raises(
            repair.RepairError, match="submit-calling evidence differs"
        ):
            repair.execute_report_repair(
                repo,
                submission,
                repair.EXPECTED_SUBMISSION_SHA256,
                allow_initial_submission=False,
                runner=lambda *_args: (_ for _ in ()).throw(
                    AssertionError("cloned source reached the scheduler")
                ),
                sleep=lambda _seconds: None,
            )
        assert len(calls) == calls_before_clone
    finally:
        source_archive.unlink()
        source_original.rename(source_archive)

    # A validated, already-published report wins without another scheduler read
    # or mutation.  The deep prefix validators still replay the sealed submit,
    # authorization, and release chain before returning.
    publication_payload = b"fixture publication archive\n"
    publication_digest = hashlib.sha256(publication_payload).hexdigest()
    publication_path = submission / (
        f"{repair.PUBLICATION_ARCHIVE_PREFIX}{publication_digest}"
        f"{repair.PUBLICATION_ARCHIVE_SUFFIX}"
    )
    publication_path.write_bytes(publication_payload)
    publication_path.chmod(0o444)
    repair.seal_json(
        submission / "REPORT_REPAIR_0002_COMPLETED.json",
        _phase_fixture_value(
            repair, "REPORT_REPAIR_0002_COMPLETED.json"
        ),
    )
    monkeypatch.setattr(
        repair,
        "_validated_repaired_report_tree",
        lambda *_args, **_kwargs: {"status": "rejected", "schema_version": 1},
    )
    monkeypatch.setattr(
        repair,
        "_seal_repair_completed",
        lambda *_args, **_kwargs: {
            "schema_version": 1,
            "status": "report_repair_publication_completed",
        },
    )

    def forbidden_scheduler(*_args):
        raise AssertionError("existing report precedence must not call scheduler")

    repeated = repair.execute_report_repair(
        repo,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        allow_initial_submission=False,
        runner=forbidden_scheduler,
        sleep=lambda _seconds: None,
    )
    assert repeated["status"] == "report_repair_already_published"
    assert repeated["scheduler_calls"] == 0


def test_predecessor_seal_before_submit_calling_resumes_with_exactly_one_sbatch(
    repair, tmp_path, monkeypatch
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )

    journal = submission / "journal"
    predecessor_path = submission / "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
    predecessor_path.unlink()
    predecessor_value = _phase_fixture_value(
        repair, "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
    )

    def validate_predecessor(*_args, **_kwargs):
        if not predecessor_path.exists():
            return None
        value, digest, _info = repair.read_json(
            predecessor_path, "fixture attempt2 predecessor"
        )
        return value, digest

    def seal_predecessor(*_args, **_kwargs):
        digest = repair.seal_json(predecessor_path, predecessor_value)
        return dict(predecessor_value), digest

    monkeypatch.setattr(
        repair, "_validated_attempt2_predecessor", validate_predecessor
    )
    monkeypatch.setattr(repair, "_seal_attempt2_predecessor", seal_predecessor)
    real_seal = repair.seal_json

    def crash_after_predecessor(path, value):
        digest = real_seal(path, value)
        if path.name == "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json":
            raise RuntimeError("predecessor-to-calling killpoint")
        return digest

    monkeypatch.setattr(repair, "seal_json", crash_after_predecessor)

    def no_scheduler_before_calling(*_args):
        raise AssertionError("scheduler call preceded durable submit calling")

    with pytest.raises(RuntimeError, match="predecessor-to-calling killpoint"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=no_scheduler_before_calling,
            sleep=lambda _seconds: None,
        )
    assert (journal / "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json").is_file()
    assert predecessor_path.is_file()
    assert not (submission / "CALLING_REPORT_REPAIR_0002_SUBMIT.json").exists()

    monkeypatch.setattr(repair, "seal_json", real_seal)
    calls = []
    squeue_round = 0

    def recover(argv, _cwd, _environment):
        nonlocal squeue_round
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/sbatch":
            return _command_result(repair, b"444444\n")
        if argv[0] == "/usr/local/bin/squeue":
            squeue_round += 1
            if squeue_round <= 3:
                return _squeue_result(repair, [])
            row = _repair_row(
                repair,
                state="PENDING" if squeue_round <= 9 else "RUNNING",
                reason="JobHeldUser" if squeue_round <= 9 else "None",
            )
            return _squeue_result(repair, [row])
        if argv[0] == "/usr/local/bin/scontrol":
            return _command_result(repair)
        raise AssertionError(argv)

    released = repair.execute_report_repair(
        repo,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        allow_initial_submission=False,
        runner=recover,
        sleep=lambda _seconds: None,
    )
    assert released["status"] == "report_repair_released_for_publication"
    assert sum(call[0] == "/usr/local/bin/sbatch" for call in calls) == 1
    assert (submission / "CALLING_REPORT_REPAIR_0002_SUBMIT.json").is_file()
    calling = json.loads(
        (submission / "CALLING_REPORT_REPAIR_0002_SUBMIT.json").read_text()
    )
    assert calling["scheduler_pre_submit_census"]["settled_rows"] == []
    assert calling["scheduler_pre_submit_census_sha256"] == repair.stable_hash(
        calling["scheduler_pre_submit_census"]
    )


def test_source_seal_before_calling_is_permanent_fail_stop_without_adoption(
    repair, tmp_path, monkeypatch
):
    real_seal_source = repair._seal_repair_source_snapshot
    real_load_source = repair._load_sealed_repair_source
    real_validate_source = repair._validate_sealed_repair_source
    repo, submission, contract, source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )
    monkeypatch.setattr(repair, "_seal_repair_source_snapshot", real_seal_source)
    monkeypatch.setattr(repair, "_load_sealed_repair_source", real_load_source)
    monkeypatch.setattr(
        repair, "_validate_sealed_repair_source", real_validate_source
    )
    report_programs = []

    def validate_original(*_args, report_program, **_kwargs):
        report_programs.append(report_program)
        return (
            contract,
            {"report_job_id": repair.EXPECTED_ORIGINAL_REPORT_JOB_ID},
            {"schema_version": 1, "files": {}},
            repair._expected_reassembly(),
        )

    monkeypatch.setattr(repair, "_validate_original_submission", validate_original)

    def seal_then_crash(*args, **kwargs):
        result = real_seal_source(*args, **kwargs)
        raise RuntimeError("source-to-calling killpoint")

    monkeypatch.setattr(repair, "_seal_repair_source_snapshot", seal_then_crash)
    with pytest.raises(RuntimeError, match="source-to-calling killpoint"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=lambda *_args: (_ for _ in ()).throw(
                AssertionError("scheduler call preceded sealed source")
            ),
            sleep=lambda _seconds: None,
        )
    source_root = repair._repair_source_root(submission)
    assert repair._source_base_matches(repair._load_sealed_repair_source(source_root), source)
    assert (submission / "journal/REPORT_REPAIR_0001_ORIGINAL_FAILURE.json").is_file()
    assert not (submission / "CALLING_REPORT_REPAIR_0002_SUBMIT.json").exists()

    monkeypatch.setattr(
        repair,
        "_verified_live_repair_source",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("recovery rebound sealed source to live checkout")
        ),
    )
    monkeypatch.setattr(repair, "_seal_repair_source_snapshot", real_seal_source)
    before = {
        path.name: (path.lstat().st_dev, path.lstat().st_ino, path.read_bytes())
        for path in submission.iterdir()
        if path.is_file()
    }
    with pytest.raises(
        repair.RepairError, match="unpaired source archive.*fail-stop"
    ):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=False,
            runner=lambda *_args: (_ for _ in ()).throw(
                AssertionError("orphan source recovery called the scheduler")
            ),
            sleep=lambda _seconds: None,
        )
    after = {
        path.name: (path.lstat().st_dev, path.lstat().st_ino, path.read_bytes())
        for path in submission.iterdir()
        if path.is_file()
    }
    assert after == before
    assert not (submission / "CALLING_REPORT_REPAIR_0002_SUBMIT.json").exists()


def test_controller_never_finalizes_unpaired_sealed_publication_archive(
    repair, tmp_path, monkeypatch
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )
    payload = b"sealed but unpaired publication archive\n"
    digest = hashlib.sha256(payload).hexdigest()
    archive = submission / (
        f"{repair.PUBLICATION_ARCHIVE_PREFIX}{digest}"
        f"{repair.PUBLICATION_ARCHIVE_SUFFIX}"
    )
    archive.write_bytes(payload)
    archive.chmod(0o444)
    before = (
        archive.lstat().st_dev,
        archive.lstat().st_ino,
        archive.lstat().st_mode,
        archive.read_bytes(),
    )
    with pytest.raises(
        repair.RepairError,
        match="(?:unpaired publication archive.*fail-stop|lacks authorization/release evidence)",
    ):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=False,
            runner=lambda *_args: (_ for _ in ()).throw(
                AssertionError("orphan publication reached scheduler recovery")
            ),
            sleep=lambda _seconds: None,
        )
    after = (
        archive.lstat().st_dev,
        archive.lstat().st_ino,
        archive.lstat().st_mode,
        archive.read_bytes(),
    )
    assert after == before
    assert not (submission / "REPORT_REPAIR_0002_COMPLETED.json").exists()


@pytest.mark.parametrize("case", ["broad_only", "exact_held", "round_drift"])
def test_predecessor_only_recovery_requires_fresh_empty_settled_owner_census(
    repair, tmp_path, monkeypatch, case
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )

    broad_row = {
        **_repair_row(repair, job_id="555555", state="RUNNING", reason="None"),
        "job_name": "exp23-launch8-foreign-report",
        "comment": "treewm-exp23:foreign-submission",
    }
    exact_row = _repair_row(repair)
    round_index = 0
    calls = []

    def hostile_census(argv, _cwd, _environment):
        nonlocal round_index
        calls.append(list(argv))
        assert argv[0] == "/usr/local/bin/squeue"
        if case == "broad_only":
            rows = [broad_row]
        elif case == "exact_held":
            rows = [exact_row]
        else:
            rows = [broad_row] if round_index == 1 else []
        round_index += 1
        return _squeue_result(repair, rows)

    expected = (
        "scheduler census did not settle"
        if case == "round_drift"
        else "fresh pre-submit scheduler census is not empty"
    )
    with pytest.raises(repair.RepairError, match=expected):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=False,
            runner=hostile_census,
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    assert [call[0] for call in calls] == ["/usr/local/bin/squeue"] * 3
    assert not (submission / "CALLING_REPORT_REPAIR_0002_SUBMIT.json").exists()
    assert not (submission / "REPORT_REPAIR_0002_SUBMITTED.json").exists()


def test_submit_calling_coherent_nonempty_census_is_rejected_on_recovery(
    repair, report, tmp_path, monkeypatch
):
    assert repair.SUBMIT_CALLING_KEYS == report.REPORT_REPAIR_SUBMIT_CALLING_KEYS
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )

    def lose_submit_response(argv, _cwd, _environment):
        assert argv[0] == "/usr/local/bin/sbatch"
        raise RuntimeError("lost report-repair submission response")

    with pytest.raises(RuntimeError, match="lost report-repair"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=_with_empty_pre_submit_census(repair, lose_submit_response),
            sleep=lambda _seconds: None,
        )
    calling_path = (
        submission / "CALLING_REPORT_REPAIR_0002_SUBMIT.json"
    )
    calling = json.loads(calling_path.read_text())
    broad_row = {
        **_repair_row(repair, job_id="555555", state="RUNNING", reason="None"),
        "job_name": "exp23-launch8-foreign-report",
        "comment": "treewm-exp23:foreign-submission",
    }
    calling["scheduler_pre_submit_census"] = _census(repair, [broad_row])
    calling["scheduler_pre_submit_census_sha256"] = repair.stable_hash(
        calling["scheduler_pre_submit_census"]
    )
    calling_path.chmod(0o600)
    calling_path.unlink()
    repair.seal_json(calling_path, calling)

    def forbidden(*_args):
        raise AssertionError("forged submit authority reached scheduler recovery")

    with pytest.raises(
        repair.RepairError,
        match="submit-calling fresh scheduler authority differs",
    ):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=False,
            runner=forbidden,
            sleep=lambda _seconds: None,
        )
    assert not (
        submission / "REPORT_REPAIR_0002_SUBMITTED.json"
    ).exists()


@pytest.mark.parametrize(
    "name",
    [
        "REPORT_REPAIR_0002_TERMINAL_SUBMIT_FAILURE.json",
        "REPORT_REPAIR_0002_TERMINAL_RELEASE_DENIED.json",
        "REPORT_REPAIR_0002_TERMINAL_WORKER_FAILURE.json",
        "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json",
        "CALLING_REPORT_REPAIR_0002_SCANCEL_0000_0000.json",
        "REPORT_REPAIR_0002_SCANCEL_RESULT_0000_0000.json",
        "REPORT_REPAIR_0002_CANCEL_TERMINAL_0000.json",
        "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json",
        "REPORT_REPAIR_0002_SUBMITTED.json",
        "REPORT_REPAIR_0002_AUTHORIZED.json",
        "CALLING_REPORT_REPAIR_0002_RELEASE_0000.json",
        "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json",
        "REPORT_REPAIR_0002_RELEASED.json",
        "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json",
    ],
)
def test_no_submit_calling_with_repair_successor_or_stop_never_calls_scheduler(
    repair, tmp_path, monkeypatch, name
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )
    hostile = submission / name
    hostile.write_bytes(_sealed_json_payload(_phase_fixture_value(repair, name)))
    hostile.chmod(0o444)

    def forbidden(*_args):
        raise AssertionError("split repair state must not call scheduler")

    with pytest.raises(
        repair.RepairError,
        match=(
            "successor/stop state|forbidden generation|mandatory predecessor chain|"
            "lacks submit calling|lacks submitted evidence|lacks authorization evidence"
        ),
    ):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=forbidden,
            sleep=lambda _seconds: None,
        )
    assert not (submission / "CALLING_REPORT_REPAIR_0002_SUBMIT.json").exists()


def test_forbidden_second_generation_blocks_calling_only_recovery_before_scheduler(
    repair, tmp_path, monkeypatch
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )

    def lose_submit_response(argv, _cwd, _environment):
        assert argv[0] == "/usr/local/bin/sbatch"
        raise RuntimeError("lost report-repair submission response")

    with pytest.raises(RuntimeError, match="lost report-repair"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=_with_empty_pre_submit_census(repair, lose_submit_response),
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    assert (submission / "CALLING_REPORT_REPAIR_0002_SUBMIT.json").is_file()
    (journal / "REPORT_REPAIR_0003_CANCEL_AUTHORIZED_0000.json").write_bytes(
        b"forbidden successor generation"
    )

    def forbidden(*_args):
        raise AssertionError("forbidden generation must precede scheduler recovery")

    with pytest.raises(
        repair.RepairError,
        match="forbidden generation|staging/generation namespace|journal generation namespace",
    ):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=False,
            runner=forbidden,
            sleep=lambda _seconds: None,
        )
    assert not (submission / "REPORT_REPAIR_0002_SUBMITTED.json").exists()


def test_orphan_worker_terminal_cannot_advance_lost_submit_recovery(
    repair, tmp_path, monkeypatch
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )

    def lose_submit_response(argv, _cwd, _environment):
        assert argv[0] == "/usr/local/bin/sbatch"
        raise RuntimeError("lost report-repair submission response")

    with pytest.raises(RuntimeError, match="lost report-repair"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=_with_empty_pre_submit_census(repair, lose_submit_response),
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    orphan = submission / "REPORT_REPAIR_0002_TERMINAL_WORKER_FAILURE.json"
    orphan.write_bytes(
        _sealed_json_payload(
            _phase_fixture_value(
                repair, "REPORT_REPAIR_0002_TERMINAL_WORKER_FAILURE.json"
            )
        )
    )
    orphan.chmod(0o444)

    def forbidden(*_args):
        raise AssertionError("orphan worker terminal must precede scheduler recovery")

    with pytest.raises(
        repair.RepairError,
        match=(
            "positive successor lacks submitted evidence|mandatory predecessor chain|"
            "release successor lacks authorization evidence"
        ),
    ):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=False,
            runner=forbidden,
            sleep=lambda _seconds: None,
        )
    assert not (submission / "REPORT_REPAIR_0002_SUBMITTED.json").exists()
    assert not (submission / "REPORT_REPAIR_0002_AUTHORIZED.json").exists()


@pytest.mark.parametrize(
    "name",
    [
        "REPORT_REPAIR_0002_AUTHORIZED.json",
        "CALLING_REPORT_REPAIR_0002_RELEASE_0000.json",
        "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json",
        "REPORT_REPAIR_0002_RELEASED.json",
        "REPORT_REPAIR_0002_TERMINAL_RELEASE_DENIED.json",
    ],
)
def test_positive_successor_cannot_advance_calling_only_recovery(
    repair, tmp_path, monkeypatch, name
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )

    def lose_submit_response(argv, _cwd, _environment):
        assert argv[0] == "/usr/local/bin/sbatch"
        raise RuntimeError("lost report-repair submission response")

    with pytest.raises(RuntimeError, match="lost report-repair"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=_with_empty_pre_submit_census(repair, lose_submit_response),
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    assert (submission / "CALLING_REPORT_REPAIR_0002_SUBMIT.json").is_file()
    assert not (submission / "REPORT_REPAIR_0002_SUBMITTED.json").exists()
    successor = submission / name
    successor.write_bytes(_sealed_json_payload(_phase_fixture_value(repair, name)))
    successor.chmod(0o444)

    def forbidden(*_args):
        raise AssertionError("positive successor must fail before scheduler recovery")

    with pytest.raises(
        repair.RepairError,
        match=(
            "positive successor lacks submitted evidence|authorization lacks submitted "
            "evidence|release successor lacks authorization evidence"
        ),
    ):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=False,
            runner=forbidden,
            sleep=lambda _seconds: None,
        )
    assert not (submission / "REPORT_REPAIR_0002_SUBMITTED.json").exists()


@pytest.mark.parametrize(
    "name",
    [
        "CALLING_REPORT_REPAIR_0002_RELEASE_0000.json",
        "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json",
        "REPORT_REPAIR_0002_RELEASED.json",
        "REPORT_REPAIR_0002_TERMINAL_RELEASE_DENIED.json",
    ],
)
def test_release_successor_cannot_advance_missing_authorization(
    repair, tmp_path, monkeypatch, name
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )

    def crash_before_authorization(argv, _cwd, _environment):
        if argv[0] == "/usr/local/bin/sbatch":
            return _command_result(repair, b"444444\n")
        assert argv[0] == "/usr/local/bin/squeue"
        raise RuntimeError("authorization census killpoint")

    with pytest.raises(RuntimeError, match="authorization census killpoint"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=_with_empty_pre_submit_census(
                repair, crash_before_authorization
            ),
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    assert (submission / "REPORT_REPAIR_0002_SUBMITTED.json").is_file()
    assert not (submission / "REPORT_REPAIR_0002_AUTHORIZED.json").exists()
    successor = submission / name
    successor.write_bytes(_sealed_json_payload(_phase_fixture_value(repair, name)))
    successor.chmod(0o444)

    def forbidden(*_args):
        raise AssertionError("release successor must fail before authorization census")

    with pytest.raises(
        repair.RepairError, match="release successor lacks authorization evidence"
    ):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=False,
            runner=forbidden,
            sleep=lambda _seconds: None,
        )
    assert not (submission / "REPORT_REPAIR_0002_AUTHORIZED.json").exists()


@pytest.mark.parametrize("release_effect", [False, True])
def test_release_calling_crash_is_reconciled_before_any_retry(
    repair, tmp_path, release_effect
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    authorization = {"repair_report_job_id": "444444"}

    def crash_after_calling(argv, _cwd, _environment):
        if argv[0] == "/usr/local/bin/squeue":
            return _squeue_result(repair, [_repair_row(repair)])
        assert list(argv) == ["/usr/local/bin/scontrol", "release", "444444"]
        raise RuntimeError("release killpoint")

    with pytest.raises(RuntimeError, match="killpoint"):
        _direct_release(repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            authorization,
            "a" * 64,
            _with_job_control(repair, submission, crash_after_calling),
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    calling = submission / "CALLING_REPORT_REPAIR_0002_RELEASE_0000.json"
    assert calling.is_file()
    assert not (
        submission / "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json"
    ).exists()

    calls = []
    census_round = 0

    def recover(argv, _cwd, _environment):
        nonlocal census_round
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            census_round += 1
            if census_round <= 6 and not release_effect:
                return _squeue_result(repair, [_repair_row(repair)])
            return _squeue_result(repair, [])
        if argv[0] == "/usr/local/bin/scontrol":
            assert not release_effect
            return _command_result(repair)
        if argv[0] == "/usr/local/bin/sacct":
            return _repair_sacct_result(repair)
        raise AssertionError(argv)

    released = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        authorization,
        "a" * 64,
        _with_job_control(repair, submission, recover),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert released["status"] == "report_repair_released"
    if release_effect:
        assert all(call[0] != "/usr/local/bin/scontrol" for call in calls)
        assert len(released["release_attempts"]) == 1
    else:
        assert sum(call[0] == "/usr/local/bin/scontrol" for call in calls) == 1
        assert len(released["release_attempts"]) == 2


@pytest.mark.parametrize("different_only", [False, True])
def test_ambiguous_release_reconciliation_seals_cleanup_before_result_and_replays(
    repair, tmp_path, monkeypatch, different_only
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    authorization = {"repair_report_job_id": "444444"}

    def crash_after_release_calling(argv, _cwd, _environment):
        if argv[0] == "/usr/local/bin/squeue":
            return _squeue_result(repair, [_repair_row(repair)])
        assert list(argv) == ["/usr/local/bin/scontrol", "release", "444444"]
        raise RuntimeError("release response lost")

    with pytest.raises(RuntimeError, match="response lost"):
        _direct_release(repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            authorization,
            "a" * 64,
            _with_job_control(repair, submission, crash_after_release_calling),
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )

    ambiguous_rows = [_repair_row(repair, job_id="555555")]
    if not different_only:
        ambiguous_rows.insert(0, _repair_row(repair, job_id="444444"))
    real_seal_successor = repair._RepairTransitionBinding.seal_successor

    def crash_after_ambiguous_result(binding, path, value):
        digest = real_seal_successor(binding, path, value)
        if path.name == "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json":
            raise RuntimeError("after ambiguous result")
        return digest

    monkeypatch.setattr(
        repair._RepairTransitionBinding,
        "seal_successor",
        crash_after_ambiguous_result,
    )

    def observe_ambiguity(argv, _cwd, _environment):
        assert argv[0] == "/usr/local/bin/squeue"
        return _squeue_result(repair, ambiguous_rows)

    with pytest.raises(RuntimeError, match="after ambiguous result"):
        _direct_release(repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            authorization,
            "a" * 64,
            _with_job_control(repair, submission, observe_ambiguity),
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    assert (submission / "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json").is_file()
    assert (submission / "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json").is_file()
    assert not (submission / "REPORT_REPAIR_0002_RELEASED.json").exists()

    monkeypatch.setattr(
        repair._RepairTransitionBinding,
        "seal_successor",
        real_seal_successor,
    )
    calls = []
    round_index = 0

    def resume_cleanup(argv, _cwd, _environment):
        nonlocal round_index
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            round_index += 1
            return _squeue_result(
                repair, ambiguous_rows if round_index <= 3 else []
            )
        if argv[0] == "/usr/local/bin/scancel":
            return _command_result(repair)
        raise AssertionError(argv)

    terminal = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        authorization,
        "a" * 64,
        _with_job_control(repair, submission, resume_cleanup),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"
    assert sum(call[0] == "/usr/local/bin/scancel" for call in calls) == 1
    assert all(call[0] != "/usr/local/bin/scontrol" for call in calls)
    assert not (submission / "REPORT_REPAIR_0002_RELEASED.json").exists()


@pytest.mark.parametrize("broad_extra", [False, True])
def test_direct_release_response_ambiguity_immediately_enters_cleanup(
    repair, tmp_path, broad_extra
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    correct = _repair_row(repair, state="RUNNING", reason="None")
    if broad_extra:
        other = {
            **correct,
            "job_id": "555555",
            "job_name": "exp23-launch8-other-report",
            "comment": "treewm-exp23:other",
        }
        ambiguous_rows = [correct, other]
    else:
        ambiguous_rows = [_repair_row(repair, job_id="555555")]
    calls = []
    squeue_round = 0

    def runner(argv, _cwd, _environment):
        nonlocal squeue_round
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            squeue_round += 1
            if squeue_round <= 3:
                return _squeue_result(repair, [_repair_row(repair)])
            if squeue_round <= 6:
                return _squeue_result(repair, ambiguous_rows)
            return _squeue_result(repair, [])
        if argv[0] == "/usr/local/bin/scontrol":
            return _command_result(repair)
        if argv[0] == "/usr/local/bin/scancel":
            return _command_result(repair)
        raise AssertionError(argv)

    terminal = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        _with_job_control(repair, submission, runner),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"
    assert sum(call[0] == "/usr/local/bin/scontrol" for call in calls) == 1
    assert sum(call[0] == "/usr/local/bin/scancel" for call in calls) == 1
    journal = submission / "journal"
    assert (submission / "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json").is_file()
    assert not (submission / "REPORT_REPAIR_0002_RELEASED.json").exists()


def test_direct_release_broad_only_row_is_never_cancelled_or_release_authority(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    unrelated = {
        **_repair_row(repair, job_id="555555", state="RUNNING", reason="None"),
        "job_name": "exp23-launch8-other-report",
        "comment": "treewm-exp23:other",
    }
    calls = []
    squeue_round = 0

    def runner(argv, _cwd, _environment):
        nonlocal squeue_round
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            squeue_round += 1
            return _squeue_result(
                repair,
                [_repair_row(repair)] if squeue_round <= 3 else [unrelated],
            )
        if argv[0] == "/usr/local/bin/scontrol":
            return _command_result(repair)
        if argv[0] == "/usr/local/bin/sacct":
            return _repair_sacct_result(repair)
        raise AssertionError(argv)

    outcome = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        _with_job_control(repair, submission, runner),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert outcome["status"] == (
        "report_repair_release_effect_awaiting_unambiguous_namespace"
    )
    assert outcome["accounting_classification"] == "active"
    assert sum(call[0] == "/usr/local/bin/scontrol" for call in calls) == 1
    assert all(call[0] != "/usr/local/bin/scancel" for call in calls)
    assert not (submission / "REPORT_REPAIR_0002_RELEASED.json").exists()
    assert not list(
        (submission / "journal").glob(
            "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_*.json"
        )
    )


@pytest.mark.parametrize(
    "state,reason,expected_classification",
    [
        ("PENDING", "JobHeldUser", "held"),
        ("PENDING", "JobHeldAdmin", "held"),
    ],
)
def test_absent_squeue_held_accounting_never_seals_released(
    repair, tmp_path, state, reason, expected_classification
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    squeue_round = 0
    calls = []

    def runner(argv, _cwd, _environment):
        nonlocal squeue_round
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            squeue_round += 1
            return _squeue_result(
                repair,
                [_repair_row(repair)] if squeue_round <= 3 else [],
            )
        if argv[0] == "/usr/local/bin/scontrol":
            return _command_result(repair)
        if argv[0] == "/usr/local/bin/sacct":
            return _repair_sacct_result(
                repair, state=state, reason=reason, start="", end=""
            )
        raise AssertionError(argv)

    outcome = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        _with_job_control(repair, submission, runner),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert outcome["status"] == "report_repair_release_effect_awaiting_accounting"
    assert outcome["accounting_classification"] == expected_classification
    assert sum(call[0] == "/usr/local/bin/scontrol" for call in calls) == 1
    assert not (submission / "REPORT_REPAIR_0002_RELEASED.json").exists()


def test_terminal_release_effect_never_rereleases_later_same_numeric_held_id(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    authorization = {"repair_report_job_id": "444444"}

    def lose_release_response(argv, _cwd, _environment):
        if argv[0] == "/usr/local/bin/squeue":
            return _squeue_result(repair, [_repair_row(repair)])
        assert argv[0] == "/usr/local/bin/scontrol"
        raise RuntimeError("lost release response")

    with pytest.raises(RuntimeError, match="lost release response"):
        _direct_release(repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            authorization,
            "a" * 64,
            _with_job_control(repair, submission, lose_release_response),
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )

    real_seal_successor = repair._RepairTransitionBinding.seal_successor

    def crash_after_effect_result(binding, path, value):
        digest = real_seal_successor(binding, path, value)
        if path.name == "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json":
            raise RuntimeError("after release effect result")
        return digest

    monkeypatch.setattr(
        repair._RepairTransitionBinding,
        "seal_successor",
        crash_after_effect_result,
    )

    def released_absent(argv, _cwd, _environment):
        if argv[0] == "/usr/local/bin/squeue":
            return _squeue_result(repair, [])
        assert argv[0] == "/usr/local/bin/sacct"
        return _repair_sacct_result(repair)

    with pytest.raises(RuntimeError, match="after release effect result"):
        _direct_release(repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            authorization,
            "a" * 64,
            _with_job_control(repair, submission, released_absent),
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    monkeypatch.setattr(
        repair._RepairTransitionBinding,
        "seal_successor",
        real_seal_successor,
    )
    calls = []
    round_index = 0

    def recycled_held(argv, _cwd, _environment):
        nonlocal round_index
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            round_index += 1
            return _squeue_result(
                repair,
                [_repair_row(repair)] if round_index <= 3 else [],
            )
        if argv[0] == "/usr/local/bin/scancel":
            return _command_result(repair)
        raise AssertionError(argv)

    terminal = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        authorization,
        "a" * 64,
        _with_job_control(repair, submission, recycled_held),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"
    assert sum(call[0] == "/usr/local/bin/scancel" for call in calls) == 1
    assert all(call[0] != "/usr/local/bin/scontrol" for call in calls)
    assert not (submission / "REPORT_REPAIR_0002_RELEASED.json").exists()


def test_cleanup_residual_job_uses_new_authority_generation(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    first = _repair_row(repair, job_id="444444")
    residual = _repair_row(repair, job_id="555555")
    first_calls = []
    first_round = 0

    def first_runner(argv, _cwd, _environment):
        nonlocal first_round
        first_calls.append(list(argv))
        if argv[0] == "/usr/local/bin/scancel":
            assert list(argv) == ["/usr/local/bin/scancel", "444444"]
            return _command_result(repair)
        first_round += 1
        return _squeue_result(repair, [residual])

    terminal0 = _direct_cleanup(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        _census(repair, [first]),
        "fixture_first",
        first_runner,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal0["status"] == "report_repair_cleanup_residual_jobs"
    second_calls = []

    def second_runner(argv, _cwd, _environment):
        second_calls.append(list(argv))
        if argv[0] == "/usr/local/bin/scancel":
            assert list(argv) == ["/usr/local/bin/scancel", "555555"]
            return _command_result(repair)
        return _squeue_result(repair, [])

    terminal1 = _direct_cleanup(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        _census(repair, [residual]),
        "fixture_residual",
        second_runner,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal1["status"] == "report_repair_terminal_cleanup_complete"
    assert terminal1["cancel_generation"] == 1
    assert (
        submission / "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0001.json"
    ).is_file()


def test_cleanup_authority_or_calling_forgery_rejects_before_scheduler(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    row = _repair_row(repair)

    def crash(argv, _cwd, _environment):
        assert argv[0] == "/usr/local/bin/scancel"
        raise RuntimeError("cleanup killpoint")

    with pytest.raises(RuntimeError):
        _direct_cleanup(repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            _census(repair, [row]),
            "fixture",
            crash,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    authority_path = (
        submission / "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json"
    )
    authority = json.loads(authority_path.read_text())
    authority["job_ids"] = ["999999"]
    authority_path.chmod(0o600)
    authority_path.write_text(json.dumps(authority, sort_keys=True, indent=2) + "\n")
    authority_path.chmod(0o444)

    def forbidden(*_args):
        raise AssertionError("scheduler must not be called for forged cleanup evidence")

    with pytest.raises(repair.RepairError, match="cleanup"):
        _direct_cleanup(repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            _census(repair, [row]),
            "ignored",
            forbidden,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )


def test_cleanup_authority_only_crash_resumes_before_scancel(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    row = _repair_row(repair)
    real_prefix = repair._cancel_attempt_prefix

    def crash_after_authority(*_args, **_kwargs):
        raise RuntimeError("authority-only killpoint")

    monkeypatch.setattr(repair, "_cancel_attempt_prefix", crash_after_authority)
    with pytest.raises(RuntimeError, match="authority-only"):
        _direct_cleanup(repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            _census(repair, [row]),
            "fixture",
            lambda *_args: (_ for _ in ()).throw(AssertionError("no call")),
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    assert (
        submission / "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json"
    ).is_file()
    assert not list(
        submission.glob("CALLING_REPORT_REPAIR_0002_SCANCEL_*.json")
    )
    monkeypatch.setattr(repair, "_cancel_attempt_prefix", real_prefix)
    squeue_round = 0

    def resume(argv, _cwd, _environment):
        nonlocal squeue_round
        if argv[0] == "/usr/local/bin/scancel":
            return _command_result(repair)
        squeue_round += 1
        return _squeue_result(repair, [])

    terminal = _direct_cleanup(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        _census(repair, [row]),
        "ignored",
        resume,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"


def test_crash_after_authorization_before_release_resumes_without_resubmit(
    repair, tmp_path, monkeypatch
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )
    real_release = repair._release_authorized_job

    def crash_before_release(*_args, **_kwargs):
        raise RuntimeError("post-authorization killpoint")

    monkeypatch.setattr(repair, "_release_authorized_job", crash_before_release)
    first_calls = []

    def first(argv, _cwd, _environment):
        first_calls.append(list(argv))
        if argv[0] == "/usr/local/bin/sbatch":
            return _command_result(repair, b"444444\n")
        if argv[0] == "/usr/local/bin/squeue":
            if sum(call[0] == "/usr/local/bin/squeue" for call in first_calls) <= 3:
                return _squeue_result(repair, [])
            return _squeue_result(repair, [_repair_row(repair)])
        raise AssertionError(argv)

    with pytest.raises(RuntimeError, match="post-authorization"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=first,
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    assert (submission / "REPORT_REPAIR_0002_AUTHORIZED.json").is_file()
    assert not (submission / "CALLING_REPORT_REPAIR_0002_RELEASE_0000.json").exists()
    monkeypatch.setattr(repair, "_release_authorized_job", real_release)
    second_calls = []
    second_round = 0

    def second(argv, _cwd, _environment):
        nonlocal second_round
        second_calls.append(list(argv))
        if argv[0] == "/usr/local/bin/scontrol":
            return _command_result(repair)
        if argv[0] == "/usr/local/bin/squeue":
            second_round += 1
            return _squeue_result(
                repair,
                [_repair_row(repair)] if second_round <= 3 else [],
            )
        if argv[0] == "/usr/local/bin/sacct":
            return _repair_sacct_result(repair)
        raise AssertionError(argv)

    result = repair.execute_report_repair(
        repo,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        allow_initial_submission=False,
        runner=second,
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "report_repair_released_for_publication"
    assert all(call[0] != "/usr/local/bin/sbatch" for call in second_calls)


def test_fresh_historical_numeric_id_is_cleanup_only_never_authorized(
    repair, tmp_path, monkeypatch
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )
    historical = repair.EXPECTED_ORIGINAL_REPORT_JOB_ID
    squeue_round = 0
    calls = []

    def runner(argv, _cwd, _environment):
        nonlocal squeue_round
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/sbatch":
            return _command_result(repair, f"{historical}\n".encode("ascii"))
        if argv[0] == "/usr/local/bin/squeue":
            squeue_round += 1
            rows = (
                [_repair_row(repair, job_id=historical)]
                if 4 <= squeue_round <= 6
                else []
            )
            return _squeue_result(repair, rows)
        if argv[0] == "/usr/local/bin/scancel":
            assert list(argv) == ["/usr/local/bin/scancel", historical]
            return _command_result(repair)
        raise AssertionError(argv)

    terminal = repair.execute_report_repair(
        repo,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        allow_initial_submission=True,
        runner=runner,
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"
    assert not (submission / "REPORT_REPAIR_0002_AUTHORIZED.json").exists()
    assert not (submission / "REPORT_REPAIR_0002_RELEASED.json").exists()
    authority = json.loads(
        (
            submission
            / "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json"
        ).read_text()
    )
    assert authority["reason"] == "historical_numeric_id_recycled"
    assert sum(call[0] == "/usr/local/bin/scancel" for call in calls) == 1


def test_lost_submit_response_with_multiple_exact_jobs_cancels_all_once(
    repair, tmp_path, monkeypatch
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )

    def crash_submit(argv, _cwd, _environment):
        assert argv[0] == "/usr/local/bin/sbatch"
        raise RuntimeError("lost sbatch response")

    with pytest.raises(RuntimeError, match="lost sbatch"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=_with_empty_pre_submit_census(repair, crash_submit),
            sleep=lambda _seconds: None,
        )
    assert (submission / "CALLING_REPORT_REPAIR_0002_SUBMIT.json").is_file()
    assert not (submission / "REPORT_REPAIR_0002_SUBMITTED.json").exists()

    rows = [
        _repair_row(repair, job_id="444444"),
        _repair_row(repair, job_id="555555"),
    ]
    squeue_round = 0
    calls = []

    def recover(argv, _cwd, _environment):
        nonlocal squeue_round
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            squeue_round += 1
            return _squeue_result(repair, rows if squeue_round <= 3 else [])
        if argv[0] == "/usr/local/bin/scancel":
            assert list(argv) == [
                "/usr/local/bin/scancel",
                "444444",
                "555555",
            ]
            return _command_result(repair)
        raise AssertionError(argv)

    terminal = repair.execute_report_repair(
        repo,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        allow_initial_submission=False,
        runner=recover,
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"
    assert terminal["remaining_job_ids"] == []
    assert sum(call[0] == "/usr/local/bin/scancel" for call in calls) == 1
    assert not (submission / "REPORT_REPAIR_0002_AUTHORIZED.json").exists()


def _fixture_repair_authority(report, submission, job_id="444444"):
    return {
        "original_report_job_id": "33311218",
        "repair_report_job_id": job_id,
        "original_failure_evidence": "journal/REPORT_REPAIR_0001_ORIGINAL_FAILURE.json",
        "original_failure_evidence_sha256": "1" * 64,
        "predecessor_failure_evidence": (
            "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
        ),
        "predecessor_failure_evidence_sha256": "b" * 64,
        "worker_receipt_map_sha256": "2" * 64,
        "snapshot_root": str(submission / "source-snapshot/repo"),
        "snapshot_inventory_sha256": "3" * 64,
        "original_package_protocol_sha256": "4" * 64,
        "repair_source_root": str(submission / "report-repair/attempt-0002/source"),
        "repair_source_commit": "5" * 40,
        "repair_package_protocol_sha256": "6" * 64,
        "repair_source_files_sha256": "7" * 64,
        "repair_source_installation_method": "renameat2_noreplace",
        "report_publication_installation_method": "renameat2_noreplace",
        "scheduler_job_control_observation_sha256": "c" * 64,
        "worker_handoff": dict(report.REPAIR_WORKER_HANDOFF),
        "expected_reassembly": dict(report.EXPECTED_REPAIR_REASSEMBLY),
        "_validated_release_sha256": "8" * 64,
    }


def legacy_repair_publication_v2_authority_preserves_exact_14_key_commit(
    report, tmp_path, monkeypatch
):
    submission = tmp_path / "run/state/submission"
    submission.mkdir(parents=True)
    submission.chmod(0o700)
    (submission / "journal").mkdir(mode=0o700)
    transaction_lock = report._repair_transaction_lock_path(submission)
    transaction_lock.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    transaction_lock.touch(mode=0o600)
    (submission / ".REPORT_CANCEL.lock").touch(mode=0o600)
    repair_parent = submission / "report-repair"
    repair_parent.mkdir(mode=0o700)
    for attempt in (1, 2):
        attempt_root = repair_parent / f"attempt-{attempt:04d}"
        attempt_root.mkdir(mode=0o700)
        source_root = attempt_root / "source"
        source_root.mkdir(mode=0o700)
        source_root.chmod(0o555)
    authorization_sha = "9" * 64
    repair_authority = _fixture_repair_authority(report, submission)
    expected = report._repair_publication_authority(
        repair_authority, authorization_sha, 2
    )
    monkeypatch.setattr(
        report,
        "_validated_report_publication_prerequisite",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        report,
        "_validated_report_publication_receipt",
        lambda *_args, **_kwargs: {"report_job_id": "33311218"},
    )
    monkeypatch.setattr(
        report,
        "_validated_report_repair_authorization",
        lambda *_args, **_kwargs: dict(repair_authority),
    )
    # This legacy test covers the public v2 authority projection and the
    # unchanged 14-key commit.  Exact deterministic quartet authentication is
    # exercised separately with the pinned reassembly fixture.
    monkeypatch.setattr(
        report._CompletedReportTreeBinding,
        "authenticate",
        lambda _self, **kwargs: dict(kwargs["commit"]),
    )
    monkeypatch.setenv("SLURM_JOB_ID", "444444")
    provenance = {
        "schema_version": 2,
        "campaign_id": report.CAMPAIGN_ID,
        "submission_sha256": "a" * 64,
        "publication_authority": expected,
    }
    body = {"status": "rejected", "reason": "unchanged scientific rejection"}
    decision = {**body, "gate_sha256": report.stable_hash(body)}
    commit = report.publish_report(
        submission,
        "a" * 64,
        {"schema_version": 1},
        decision,
        provenance,
        repair_attempt=2,
        repair_authorization_sha256=authorization_sha,
    )
    assert len(commit) == 14
    assert commit["status"] == "rejected"
    assert commit["scientific_rejection"] is True
    stored = json.loads(
        (submission / "report" / commit["provenance"]).read_text()
    )
    assert stored["schema_version"] == 2
    assert stored["publication_authority"] == expected
    serialized_authority = json.dumps(
        stored["publication_authority"], sort_keys=True, separators=(",", ":")
    )
    assert "retained_environment_evidence" not in serialized_authority
    assert "retained_batch_script_evidence" not in serialized_authority
    assert "failure_log" not in serialized_authority
    assert stored["publication_authority"]["attempt1_environment_evidence"] == {
        "schema_version": 1,
        "raw_stdout_sha256": report.EXPECTED_ATTEMPT1_ENV_STDOUT_SHA256,
        "raw_stdout_size": report.EXPECTED_ATTEMPT1_ENV_STDOUT_SIZE,
        "allowlisted_projection": {
            "slurm_export_env": "NONE",
            "slurm_restart_count_present": False,
        },
    }
    assert '"data"' not in serialized_authority
    assert '"environment"' not in serialized_authority
    completed_path = submission / "REPORT_REPAIR_0002_COMPLETED.json"
    completed_bytes = completed_path.read_bytes()
    assert (
        report.publish_report(
            submission,
            "a" * 64,
            {"schema_version": 1},
            decision,
            provenance,
            repair_attempt=2,
            repair_authorization_sha256=authorization_sha,
        )
        == commit
    )
    assert completed_path.read_bytes() == completed_bytes

    other = tmp_path / "forged-run/state/submission"
    other.mkdir(parents=True)
    other.chmod(0o700)
    (other / "journal").mkdir(mode=0o700)
    other_transaction_lock = report._repair_transaction_lock_path(other)
    other_transaction_lock.touch(mode=0o600)
    (other / ".REPORT_CANCEL.lock").touch(mode=0o600)
    forged = json.loads(json.dumps(provenance))
    forged["publication_authority"]["repair_report_job_id"] = "555555"
    with pytest.raises(report.ReportError, match="provenance authority"):
        report.publish_report(
            other,
            "a" * 64,
            {"schema_version": 1},
            decision,
            forged,
            repair_attempt=2,
            repair_authorization_sha256=authorization_sha,
        )
    assert not (other / "report").exists()


def test_completed_repair_prefix_without_report_is_a_publication_stop(report, tmp_path):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    report.seal_json(
        submission / "REPORT_REPAIR_0002_COMPLETED.json",
        {"schema_version": 1, "status": "forged_without_report"},
    )
    assert report._durable_repair_stop_prefix_exists(
        submission, repair_attempt=2
    )

def test_report_and_cancel_contenders_linearize_on_shared_lock(
    report, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    submission.mkdir()
    monkeypatch.setattr(
        report,
        "_validated_report_publication_prerequisite",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        report,
        "_validated_report_publication_receipt",
        lambda *_args, **_kwargs: {"report_job_id": "12345"},
    )
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    body = {"status": "rejected", "reason": "race fixture"}
    decision = {**body, "gate_sha256": report.stable_hash(body)}
    barrier = threading.Barrier(2)
    outcomes = []

    def publish():
        barrier.wait()
        try:
            report.publish_report(
                submission,
                "a" * 64,
                {"schema_version": 1},
                decision,
                {"schema_version": 1},
            )
            outcomes.append("report")
        except report.ReportError:
            outcomes.append("report_blocked")

    def cancel():
        barrier.wait()
        with report._ReportCancelLock(submission):
            if os.path.lexists(submission / "report"):
                outcomes.append("cancel_saw_report")
            else:
                report.seal_json(
                    submission / "CANCEL_REQUESTED.json",
                    {"schema_version": 1, "status": "cancel_requested"},
                )
                outcomes.append("cancel")

    threads = [threading.Thread(target=publish), threading.Thread(target=cancel)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert not (
        os.path.lexists(submission / "report")
        and os.path.lexists(submission / "CANCEL_REQUESTED.json")
    )
    assert not list(submission.glob(".report.tmp.*"))
    assert set(outcomes) in (
        {"report", "cancel_saw_report"},
        {"cancel", "report_blocked"},
    )


def test_original_report_failure_terminal_mismatch_rejected_coherently(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    (submission / "logs").mkdir()
    contract = {
        "snapshot_root": str(submission / "source-snapshot/repo"),
        "snapshot_inventory_sha256": repair.EXPECTED_SNAPSHOT_INVENTORY_SHA256,
        "package_protocol_sha256": repair.EXPECTED_ORIGINAL_PROTOCOL,
        "git_provenance": _actual_git_provenance(),
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"},
    }
    calling_sha = repair.seal_json(
        submission / "journal/CALLING_REPORT.json", {"fixture": "calling"}
    )
    submitted_sha = repair.seal_json(
        submission / "journal/0005_REPORT_SUBMITTED.json", {"fixture": "submitted"}
    )
    authorization_sha = repair.seal_json(
        submission / "SUBMISSION_AUTHORIZATION.json", {"fixture": "authorization"}
    )
    receipt_sha = repair.seal_json(
        submission / "SUBMISSION_RECEIPT.json", {"fixture": "receipt"}
    )
    log_payload = (
        "Exp23 report engineering error: staged report artifact identity differs: "
        f"REPORT_BUNDLE.{repair.EXPECTED_BUNDLE_SHA256}.json\n"
    ).encode("utf-8")
    log_path = submission / f"logs/report_{repair.EXPECTED_ORIGINAL_REPORT_JOB_ID}.out"
    log_path.write_bytes(log_payload)
    log_path.chmod(0o600)
    monkeypatch.setattr(
        repair, "EXPECTED_ORIGINAL_REPORT_CALLING_SHA256", calling_sha
    )
    monkeypatch.setattr(
        repair, "EXPECTED_ORIGINAL_REPORT_SUBMITTED_SHA256", submitted_sha
    )
    monkeypatch.setattr(repair, "EXPECTED_AUTHORIZATION_RAW_SHA256", authorization_sha)
    monkeypatch.setattr(repair, "EXPECTED_RECEIPT_RAW_SHA256", receipt_sha)
    monkeypatch.setattr(
        repair,
        "EXPECTED_ORIGINAL_REPORT_LOG_SHA256",
        hashlib.sha256(log_payload).hexdigest(),
    )
    monkeypatch.setattr(repair, "EXPECTED_ORIGINAL_REPORT_LOG_SIZE", len(log_payload))
    calls = 0

    def runner(argv, _cwd, _environment):
        nonlocal calls
        calls += 1
        if argv[0] == "/usr/local/bin/squeue":
            return _squeue_result(repair, [])
        if argv[0] == "/usr/local/bin/sacct":
            parsed = {
                "JobIDRaw": repair.EXPECTED_ORIGINAL_REPORT_JOB_ID,
                "JobName": repair.EXPECTED_ORIGINAL_REPORT_JOB_NAME,
                "State": "FAILED",
                "ExitCode": "2:0",
                "ElapsedRaw": "355",
                "AllocNodes": "1",
                "NodeList": "cpu-00090",
                "Submit": "2026-08-29T08:00:00",
                "Eligible": "2026-08-29T08:00:01",
                "Start": "2026-08-29T08:28:49",
                "End": "2026-08-29T08:34:44",
                "Comment": repair.EXPECTED_ORIGINAL_SCHEDULER_COMMENT,
            }
            payload = (
                "|".join(parsed[field] for field in repair.SACCT_FIELDS) + "\n"
            ).encode("utf-8")
            return _command_result(repair, payload)
        raise AssertionError(argv)

    failure = repair._build_failure_evidence(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"schema_version": 1, "files": {}},
        repair._expected_reassembly(),
        runner,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    repair._validate_failure_evidence(
        failure,
        submission_root=submission,
        submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
        contract=contract,
        receipt_map={"schema_version": 1, "files": {}},
        expected_reassembly=repair._expected_reassembly(),
    )
    assert calls == 4

    forged = json.loads(json.dumps(failure))
    parsed = forged["terminal_scheduler_observation"]["parsed_row"]
    parsed["State"] = "COMPLETED"
    canonical = forged["terminal_scheduler_observation"]["canonical"]
    canonical["rows"] = [[parsed[field] for field in repair.SACCT_FIELDS]]
    forged["terminal_scheduler_observation"]["canonical_sha256"] = repair.stable_hash(
        canonical
    )
    environment = forged["terminal_scheduler_observation"]["raw"]["environment"]
    argv = forged["terminal_scheduler_observation"]["raw"]["argv"]
    raw_payload = (
        "|".join(parsed[field] for field in repair.SACCT_FIELDS) + "\n"
    ).encode("utf-8")
    forged["terminal_scheduler_observation"]["raw"] = repair._command_evidence(
        argv, environment, repair.CommandResult(0, raw_payload, b"")
    )
    with pytest.raises(repair.RepairError, match="terminal scheduler semantics"):
        repair._validate_failure_evidence(
            forged,
            submission_root=submission,
            submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
            contract=contract,
            receipt_map={"schema_version": 1, "files": {}},
            expected_reassembly=repair._expected_reassembly(),
        )

    forged_timing = json.loads(json.dumps(failure))
    timing_row = forged_timing["terminal_scheduler_observation"]["parsed_row"]
    timing_row["Submit"] = "2026-08-29T08:30:00"
    timing_canonical = forged_timing["terminal_scheduler_observation"]["canonical"]
    timing_canonical["rows"] = [
        [timing_row[field] for field in repair.SACCT_FIELDS]
    ]
    forged_timing["terminal_scheduler_observation"][
        "canonical_sha256"
    ] = repair.stable_hash(timing_canonical)
    timing_environment = forged_timing["terminal_scheduler_observation"]["raw"][
        "environment"
    ]
    timing_argv = forged_timing["terminal_scheduler_observation"]["raw"]["argv"]
    timing_payload = (
        "|".join(timing_row[field] for field in repair.SACCT_FIELDS) + "\n"
    ).encode("utf-8")
    forged_timing["terminal_scheduler_observation"]["raw"] = (
        repair._command_evidence(
            timing_argv,
            timing_environment,
            repair.CommandResult(0, timing_payload, b""),
        )
    )
    with pytest.raises(repair.RepairError, match="terminal scheduler semantics"):
        repair._validate_failure_evidence(
            forged_timing,
            submission_root=submission,
            submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
            contract=contract,
            receipt_map={"schema_version": 1, "files": {}},
            expected_reassembly=repair._expected_reassembly(),
        )


@pytest.mark.parametrize("name", ["transaction.lock", ".REPORT_CANCEL.lock"])
def test_existing_lock_revalidates_named_inode_after_blocking_flock(
    repair, tmp_path, monkeypatch, name
):
    path = tmp_path / name
    path.write_bytes(b"")
    path.chmod(0o600)
    displaced = tmp_path / f"{name}.displaced"
    real_flock = repair.fcntl.flock

    def replace_after_lock(descriptor, operation):
        real_flock(descriptor, operation)
        path.rename(displaced)
        path.write_bytes(b"replacement")
        path.chmod(0o600)

    monkeypatch.setattr(repair.fcntl, "flock", replace_after_lock)
    with pytest.raises(repair.RepairError, match="binding changed while waiting"):
        with repair._ExistingLock(path, name):
            raise AssertionError("stale lock binding was accepted")
    assert path.read_bytes() == b"replacement"
    assert displaced.read_bytes() == b""


def test_report_publication_lock_revalidates_named_inode_after_blocking_flock(
    report, tmp_path, monkeypatch
):
    path = tmp_path / ".REPORT_CANCEL.lock"
    path.write_bytes(b"")
    path.chmod(0o600)
    displaced = tmp_path / ".REPORT_CANCEL.lock.displaced"
    real_flock = report.fcntl.flock

    def replace_after_lock(descriptor, operation):
        real_flock(descriptor, operation)
        if operation == report.fcntl.LOCK_EX:
            path.rename(displaced)
            path.write_bytes(b"replacement")
            path.chmod(0o600)

    monkeypatch.setattr(report.fcntl, "flock", replace_after_lock)
    with pytest.raises(report.ReportError, match="binding changed while waiting"):
        with report._ReportCancelLock(tmp_path):
            raise AssertionError("stale report lock binding was accepted")
    assert path.read_bytes() == b"replacement"
    assert displaced.read_bytes() == b""


def test_lost_submit_running_identity_is_cleanup_only_before_submitted_seal(
    repair, tmp_path, monkeypatch
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )

    def lost_submit(argv, _cwd, _environment):
        assert argv[0] == "/usr/local/bin/sbatch"
        raise RuntimeError("lost sbatch response")

    with pytest.raises(RuntimeError, match="lost sbatch"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=_with_empty_pre_submit_census(repair, lost_submit),
            sleep=lambda _seconds: None,
        )

    squeue_round = 0
    calls = []

    def recover(argv, _cwd, _environment):
        nonlocal squeue_round
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            squeue_round += 1
            rows = (
                [_repair_row(repair, state="RUNNING", reason="None")]
                if squeue_round <= 3
                else []
            )
            return _squeue_result(repair, rows)
        if argv[0] == "/usr/local/bin/scancel":
            return _command_result(repair)
        raise AssertionError(argv)

    terminal = repair.execute_report_repair(
        repo,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        allow_initial_submission=False,
        runner=recover,
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"
    assert not (submission / "REPORT_REPAIR_0002_SUBMITTED.json").exists()
    assert not (submission / "REPORT_REPAIR_0002_AUTHORIZED.json").exists()
    assert sum(call[0] == "/usr/local/bin/scancel" for call in calls) == 1


def test_fresh_release_requires_present_sole_held_authorized_identity(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    calls = []

    def absent(argv, _cwd, _environment):
        calls.append(list(argv))
        assert argv[0] == "/usr/local/bin/squeue"
        return _squeue_result(repair, [])

    terminal = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        _with_job_control(repair, submission, absent),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_release_denied"
    assert terminal["publication_allowed"] is False
    assert all(call[0] != "/usr/local/bin/scontrol" for call in calls)
    assert not list(
        (submission / "journal").glob(
            "CALLING_REPORT_REPAIR_0002_RELEASE_*.json"
        )
    )

    repeat_calls = []

    def still_absent(argv, _cwd, _environment):
        repeat_calls.append(list(argv))
        assert argv[0] == "/usr/local/bin/squeue"
        return _squeue_result(repair, [])

    repeated = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        _with_job_control(repair, submission, still_absent),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert repeated == terminal
    assert len(repeat_calls) == 3

    delayed_calls = []
    delayed_round = 0

    def delayed_visible(argv, _cwd, _environment):
        nonlocal delayed_round
        delayed_calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            delayed_round += 1
            return _squeue_result(
                repair,
                [_repair_row(repair)] if delayed_round <= 3 else [],
            )
        if argv[0] == "/usr/local/bin/scancel":
            return _command_result(repair)
        raise AssertionError(argv)

    cleaned = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        _with_job_control(repair, submission, delayed_visible),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert cleaned["status"] == "report_repair_terminal_cleanup_complete"
    assert sum(call[0] == "/usr/local/bin/scancel" for call in delayed_calls) == 1
    assert all(call[0] != "/usr/local/bin/scontrol" for call in delayed_calls)


def test_fresh_release_running_identity_is_cancelled_before_any_release(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    calls = []
    round_index = 0

    def runner(argv, _cwd, _environment):
        nonlocal round_index
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            round_index += 1
            rows = (
                [_repair_row(repair, state="RUNNING", reason="None")]
                if round_index <= 3
                else []
            )
            return _squeue_result(repair, rows)
        if argv[0] == "/usr/local/bin/scancel":
            return _command_result(repair)
        raise AssertionError(argv)

    terminal = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        _with_job_control(repair, submission, runner),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"
    assert sum(call[0] == "/usr/local/bin/scancel" for call in calls) == 1
    assert all(call[0] != "/usr/local/bin/scontrol" for call in calls)
    assert not list(
        (submission / "journal").glob(
            "CALLING_REPORT_REPAIR_0002_RELEASE_*.json"
        )
    )


def test_release_rc0_with_stderr_remains_reconcilable_calling_only(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }

    def warning(argv, _cwd, _environment):
        if argv[0] == "/usr/local/bin/squeue":
            return _squeue_result(repair, [_repair_row(repair)])
        assert argv[0] == "/usr/local/bin/scontrol"
        return _command_result(repair, stderr=b"warning\n")

    with pytest.raises(repair.RepairError, match="scontrol release failed"):
        _direct_release(repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            {"repair_report_job_id": "444444"},
            "a" * 64,
            _with_job_control(repair, submission, warning),
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    assert (submission / "CALLING_REPORT_REPAIR_0002_RELEASE_0000.json").is_file()
    assert not (submission / "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json").exists()
    assert not (submission / "REPORT_REPAIR_0002_RELEASED.json").exists()

    calls = []

    def reconcile(argv, _cwd, _environment):
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            return _squeue_result(repair, [])
        if argv[0] == "/usr/local/bin/sacct":
            return _repair_sacct_result(repair)
        raise AssertionError(argv)

    released = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        _with_job_control(repair, submission, reconcile),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert released["status"] == "report_repair_released"
    assert released["release_attempts"][0]["release_attempt"] == 0
    assert all(call[0] != "/usr/local/bin/scontrol" for call in calls)


@pytest.mark.parametrize("visible_running", [False, True])
def test_direct_release_result_crash_recovers_effect_without_second_mutation(
    repair, tmp_path, visible_running
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    first_round = 0

    def crash_before_post_release_census(argv, _cwd, _environment):
        nonlocal first_round
        if argv[0] == "/usr/local/bin/squeue":
            first_round += 1
            if first_round <= 3:
                return _squeue_result(repair, [_repair_row(repair)])
            raise RuntimeError("post-release census killpoint")
        assert argv[0] == "/usr/local/bin/scontrol"
        return _command_result(repair)

    with pytest.raises(RuntimeError, match="post-release census killpoint"):
        _direct_release(repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            {"repair_report_job_id": "444444"},
            "a" * 64,
            _with_job_control(
                repair, submission, crash_before_post_release_census
            ),
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    assert (submission / "CALLING_REPORT_REPAIR_0002_RELEASE_0000.json").is_file()
    assert (submission / "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json").is_file()
    assert not (submission / "REPORT_REPAIR_0002_RELEASED.json").exists()

    calls = []

    def recover(argv, _cwd, _environment):
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            rows = (
                [_repair_row(repair, state="RUNNING", reason="None")]
                if visible_running
                else []
            )
            return _squeue_result(repair, rows)
        if argv[0] == "/usr/local/bin/sacct":
            return _repair_sacct_result(repair)
        raise AssertionError(argv)

    released = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        _with_job_control(repair, submission, recover),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert released["status"] == "report_repair_released"
    assert len(released["release_attempts"]) == 1
    assert all(call[0] in {"/usr/local/bin/squeue", "/usr/local/bin/sacct"} for call in calls)
    assert (not visible_running) == any(
        call[0] == "/usr/local/bin/sacct" for call in calls
    )
    assert not (submission / "REPORT_REPAIR_0002_TERMINAL_RELEASE_DENIED.json").exists()


def test_terminal_worker_accounting_prevents_release_evidence_and_publication(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    calls = []
    squeue_round = 0

    def runner(argv, _cwd, _environment):
        nonlocal squeue_round
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            squeue_round += 1
            return _squeue_result(
                repair,
                [_repair_row(repair)] if squeue_round <= 3 else [],
            )
        if argv[0] == "/usr/local/bin/scontrol":
            return _command_result(repair)
        if argv[0] == "/usr/local/bin/sacct":
            return _repair_sacct_result(
                repair,
                state="FAILED",
                exit_code="2:0",
                end="2026-08-29T10:01:03",
            )
        raise AssertionError(argv)

    terminal = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        _with_job_control(repair, submission, runner),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_worker_failure"
    assert terminal["reason"] == "repair_worker_terminal_before_release_evidence"
    assert terminal["publication_allowed"] is False
    journal = submission / "journal"
    assert (submission / "REPORT_REPAIR_0002_TERMINAL_WORKER_FAILURE.json").is_file()
    assert not (submission / "REPORT_REPAIR_0002_RELEASED.json").exists()
    assert sum(call[0] == "/usr/local/bin/scontrol" for call in calls) == 1
    assert all(call[0] != "/usr/local/bin/scancel" for call in calls)


def test_released_worker_terminal_wins_and_broad_only_replay_is_idempotent(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    authorization_sha = "a" * 64
    squeue_round = 0

    def release_runner(argv, _cwd, _environment):
        nonlocal squeue_round
        if argv[0] == "/usr/local/bin/squeue":
            squeue_round += 1
            row = _repair_row(
                repair,
                state="PENDING" if squeue_round <= 3 else "RUNNING",
                reason="JobHeldUser" if squeue_round <= 3 else "None",
            )
            return _squeue_result(repair, [row])
        if argv[0] == "/usr/local/bin/scontrol":
            return _command_result(repair)
        raise AssertionError(argv)

    released = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        authorization_sha,
        _with_job_control(repair, submission, release_runner),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert released["status"] == "report_repair_released"

    def terminal_runner(argv, _cwd, _environment):
        if argv[0] == "/usr/local/bin/squeue":
            return _squeue_result(repair, [])
        if argv[0] == "/usr/local/bin/sacct":
            return _repair_sacct_result(
                repair,
                state="FAILED",
                exit_code="2:0",
                end="2026-08-29T10:01:03",
            )
        raise AssertionError(argv)

    terminal = repair._reconcile_released_worker(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        authorization_sha,
        released,
        terminal_runner,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_worker_failure"
    assert terminal["reason"] == "repair_worker_terminal_after_release_evidence"

    unrelated = {
        **_repair_row(repair, job_id="555555", state="RUNNING", reason="None"),
        "job_name": "exp23-launch8-other-report",
        "comment": "treewm-exp23:other",
    }
    replay_calls = []

    def broad_only(argv, _cwd, _environment):
        replay_calls.append(list(argv))
        assert argv[0] == "/usr/local/bin/squeue"
        return _squeue_result(repair, [unrelated])

    repeated = repair._reconcile_released_worker(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        authorization_sha,
        released,
        broad_only,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert repeated == terminal
    assert len(replay_calls) == 3
    assert all(call[0] == "/usr/local/bin/squeue" for call in replay_calls)


def test_released_cleanup_authority_dominates_crash_before_cleanup_terminal(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    authorization_sha = "a" * 64
    squeue_round = 0

    def release_runner(argv, _cwd, _environment):
        nonlocal squeue_round
        if argv[0] == "/usr/local/bin/squeue":
            squeue_round += 1
            row = _repair_row(
                repair,
                state="PENDING" if squeue_round <= 3 else "RUNNING",
                reason="JobHeldUser" if squeue_round <= 3 else "None",
            )
            return _squeue_result(repair, [row])
        if argv[0] == "/usr/local/bin/scontrol":
            return _command_result(repair)
        raise AssertionError(argv)

    released = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        authorization_sha,
        _with_job_control(repair, submission, release_runner),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert released["status"] == "report_repair_released"

    def crash_during_cleanup(argv, _cwd, _environment):
        if argv[0] == "/usr/local/bin/squeue":
            return _squeue_result(repair, [_repair_row(repair)])
        assert list(argv) == ["/usr/local/bin/scancel", "444444"]
        raise RuntimeError("released-cleanup killpoint")

    with pytest.raises(RuntimeError, match="released-cleanup killpoint"):
        repair._reconcile_released_worker(
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            authorization_sha,
            released,
            crash_during_cleanup,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    assert (submission / "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json").is_file()
    assert (submission / "CALLING_REPORT_REPAIR_0002_SCANCEL_0000_0000.json").is_file()
    assert not (submission / "REPORT_REPAIR_0002_CANCEL_TERMINAL_0000.json").exists()

    replay_calls = []

    def replay(argv, _cwd, _environment):
        replay_calls.append(list(argv))
        assert argv[0] == "/usr/local/bin/squeue"
        return _squeue_result(repair, [])

    terminal = repair._reconcile_released_worker(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        authorization_sha,
        released,
        replay,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"
    assert len(replay_calls) == 3
    assert all(call[0] == "/usr/local/bin/squeue" for call in replay_calls)
    result = json.loads(
        (submission / "REPORT_REPAIR_0002_SCANCEL_RESULT_0000_0000.json").read_text()
    )
    assert result["mode"] == "lost_response_reconciled_cancel_effect"


def test_released_and_release_denied_are_mutually_exclusive(repair, tmp_path):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    authorization_sha = "a" * 64
    squeue_round = 0

    def release_runner(argv, _cwd, _environment):
        nonlocal squeue_round
        if argv[0] == "/usr/local/bin/squeue":
            squeue_round += 1
            row = _repair_row(
                repair,
                state="PENDING" if squeue_round <= 3 else "RUNNING",
                reason="JobHeldUser" if squeue_round <= 3 else "None",
            )
            return _squeue_result(repair, [row])
        if argv[0] == "/usr/local/bin/scontrol":
            return _command_result(repair)
        raise AssertionError(argv)

    released = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        authorization_sha,
        _with_job_control(repair, submission, release_runner),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert released["status"] == "report_repair_released"
    (submission / "REPORT_REPAIR_0002_TERMINAL_RELEASE_DENIED.json").write_bytes(
        b"conflicting terminal"
    )
    with pytest.raises(repair.RepairError, match="released and release-denied"):
        repair._validate_existing_release(
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            authorization_sha,
            "444444",
            contract,
            _FakeLocks(),
        )


@pytest.mark.parametrize(
    ("artifact", "field", "replacement"),
    [
        ("calling", "schema_version", True),
        ("calling", "attempt", True),
        ("calling", "release_attempt", False),
        ("calling", "called_at_utc", ""),
        ("result", "schema_version", True),
        ("result", "attempt", True),
        ("result", "release_attempt", False),
        ("result", "observed_at_utc", ""),
        ("released", "schema_version", True),
        ("released", "attempt", True),
        ("released", "released_at_utc", ""),
    ],
)
def test_release_chain_rejects_type_erasure_and_empty_timestamps(
    repair, tmp_path, artifact, field, replacement
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    authorization_sha = "a" * 64
    squeue_round = 0

    def release_runner(argv, _cwd, _environment):
        nonlocal squeue_round
        if argv[0] == "/usr/local/bin/squeue":
            squeue_round += 1
            row = _repair_row(
                repair,
                state="PENDING" if squeue_round <= 3 else "RUNNING",
                reason="JobHeldUser" if squeue_round <= 3 else "None",
            )
            return _squeue_result(repair, [row])
        if argv[0] == "/usr/local/bin/scontrol":
            return _command_result(repair)
        raise AssertionError(argv)

    released = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        authorization_sha,
        _with_job_control(repair, submission, release_runner),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    calling_path = submission / "CALLING_REPORT_REPAIR_0002_RELEASE_0000.json"
    result_path = submission / "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json"
    released_path = submission / "REPORT_REPAIR_0002_RELEASED.json"
    calling, calling_sha, _calling_info = repair.read_json(
        calling_path, "fixture release calling"
    )
    result, _result_sha, _result_info = repair.read_json(
        result_path, "fixture release result"
    )

    if artifact == "calling":
        calling[field] = replacement
        with pytest.raises(repair.RepairError, match="release-calling"):
            repair._validate_release_calling(
                calling,
                submission_root=submission,
                submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
                index=0,
                job_id="444444",
                authorization_sha256=authorization_sha,
                contract=contract,
                locks=_FakeLocks(),
            )
    elif artifact == "result":
        result[field] = replacement
        with pytest.raises(repair.RepairError, match="release-result"):
            repair._validate_release_result(
                result,
                submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
                index=0,
                job_id="444444",
                authorization_sha256=authorization_sha,
                calling_sha256=calling_sha,
                scheduler_environment=calling["scheduler_environment"],
                contract=contract,
            )
    else:
        released[field] = replacement
        released_path.unlink()
        repair.seal_json(released_path, released)
        with pytest.raises(repair.RepairError, match="released result"):
            repair._validate_existing_release(
                submission,
                repair.EXPECTED_SUBMISSION_SHA256,
                authorization_sha,
                "444444",
                contract,
                _FakeLocks(),
            )


@pytest.mark.parametrize("include_exact", [False, True])
def test_reconciled_release_result_rejects_broad_ambiguous_bound_census(
    repair, include_exact
):
    unrelated = {
        **_repair_row(repair, job_id="555555", state="RUNNING", reason="None"),
        "job_name": "exp23-launch8-other-report",
        "comment": "treewm-exp23:other",
    }
    rows = [_repair_row(repair), unrelated] if include_exact else [unrelated]
    census = _census(repair, rows)
    result = {
        "schema_version": 1,
        "status": "report_repair_release_attempt_observed",
        "campaign_id": repair.CAMPAIGN_ID,
        "submission_sha256": repair.EXPECTED_SUBMISSION_SHA256,
        "attempt": 2,
        "release_attempt": 0,
        "repair_report_job_id": "444444",
        "authorization_sha256": "a" * 64,
        "release_calling_sha256": "b" * 64,
        "mode": "lost_response_reconciled_release_effect",
        "scheduler_evidence": {
            "census": census,
            "census_sha256": repair.stable_hash(census),
        },
        "observed_at_utc": "2026-08-29T00:00:00Z",
    }
    with pytest.raises(repair.RepairError, match="reconciled release is ambiguous"):
        repair._validate_release_result(
            result,
            submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
            index=0,
            job_id="444444",
            authorization_sha256="a" * 64,
            calling_sha256="b" * 64,
            scheduler_environment=repair._scheduler_environment("/tmp/slurm.conf"),
            contract=_scheduler_contract(),
        )


def test_worker_terminal_timeline_is_state_conditioned(repair):
    cancelled = {
        "parsed_row": {
            "State": "CANCELLED",
            "Reason": "None",
            "Submit": "2026-08-29T10:00:00",
            "Eligible": "2026-08-29T10:00:01",
            "Start": "Unknown",
            "End": "2026-08-29T10:00:02",
            "ExitCode": "0:15",
        }
    }
    assert repair._repair_accounting_classification(cancelled) == "terminal"
    malformed_failed = {
        "parsed_row": {
            **cancelled["parsed_row"],
            "State": "FAILED",
            "ExitCode": "2:0",
        }
    }
    with pytest.raises(repair.RepairError, match="terminal accounting timeline"):
        repair._repair_accounting_classification(malformed_failed)


@pytest.mark.parametrize("final_state", ["absent", "running", "held"])
def test_three_paired_release_results_are_censused_before_limit_disposition(
    repair, tmp_path, final_state
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    authorization = {"repair_report_job_id": "444444"}

    for _attempt in range(3):
        def still_held(argv, _cwd, _environment):
            if argv[0] == "/usr/local/bin/squeue":
                return _squeue_result(repair, [_repair_row(repair)])
            assert argv[0] == "/usr/local/bin/scontrol"
            return _command_result(repair)

        with pytest.raises(repair.RepairError, match="remains held"):
            _direct_release(repair,
                submission,
                repair.EXPECTED_SUBMISSION_SHA256,
                contract,
                authorization,
                "a" * 64,
                _with_job_control(repair, submission, still_held),
                _FakeLocks(),
                sleep=lambda _seconds: None,
            )

    journal = submission / "journal"
    assert len(list(submission.glob("REPORT_REPAIR_0002_RELEASE_RESULT_*.json"))) == 3
    calls = []
    round_index = 0

    def final_observation(argv, _cwd, _environment):
        nonlocal round_index
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            round_index += 1
            if final_state == "absent" or (final_state == "held" and round_index > 3):
                return _squeue_result(repair, [])
            if final_state == "running":
                return _squeue_result(
                    repair, [_repair_row(repair, state="RUNNING", reason="None")]
                )
            return _squeue_result(repair, [_repair_row(repair)])
        if argv[0] == "/usr/local/bin/scancel" and final_state == "held":
            return _command_result(repair)
        if argv[0] == "/usr/local/bin/sacct" and final_state == "absent":
            return _repair_sacct_result(repair)
        raise AssertionError(argv)

    outcome = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        authorization,
        "a" * 64,
        _with_job_control(repair, submission, final_observation),
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    if final_state == "held":
        assert outcome["status"] == "report_repair_terminal_cleanup_complete"
        assert sum(call[0] == "/usr/local/bin/scancel" for call in calls) == 1
    else:
        assert outcome["status"] == "report_repair_released"
    assert all(call[0] != "/usr/local/bin/scontrol" for call in calls)


def test_scancel_rc0_with_stderr_remains_reconcilable_calling_only(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    contract = {
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"}
    }
    row = _repair_row(repair)

    def warning(argv, _cwd, _environment):
        assert argv[0] == "/usr/local/bin/scancel"
        return _command_result(repair, stderr=b"warning\n")

    with pytest.raises(repair.RepairError, match="cleanup scancel failed"):
        _direct_cleanup(repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            _census(repair, [row]),
            "fixture_warning",
            warning,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    assert (submission / "CALLING_REPORT_REPAIR_0002_SCANCEL_0000_0000.json").is_file()
    assert not (submission / "REPORT_REPAIR_0002_SCANCEL_RESULT_0000_0000.json").exists()

    def reconcile(argv, _cwd, _environment):
        assert argv[0] == "/usr/local/bin/squeue"
        return _squeue_result(repair, [])

    terminal = _direct_cleanup(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        _census(repair, []),
        "ignored",
        reconcile,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"


def test_terminal_submit_failure_cleans_delayed_visible_exact_job(
    repair, tmp_path, monkeypatch
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )
    first_calls = []

    def first(argv, _cwd, _environment):
        first_calls.append(list(argv))
        if argv[0] == "/usr/local/bin/sbatch":
            return _command_result(repair, stderr=b"client failure\n", returncode=1)
        assert argv[0] == "/usr/local/bin/squeue"
        return _squeue_result(repair, [])

    terminal = repair.execute_report_repair(
        repo,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        allow_initial_submission=True,
        runner=first,
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_submit_failure"

    second_calls = []
    squeue_round = 0

    def second(argv, _cwd, _environment):
        nonlocal squeue_round
        second_calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            squeue_round += 1
            rows = [_repair_row(repair)] if squeue_round <= 3 else []
            return _squeue_result(repair, rows)
        if argv[0] == "/usr/local/bin/scancel":
            return _command_result(repair)
        raise AssertionError(argv)

    cleaned = repair.execute_report_repair(
        repo,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        allow_initial_submission=False,
        runner=second,
        sleep=lambda _seconds: None,
    )
    assert cleaned["status"] == "report_repair_terminal_cleanup_complete"
    assert all(call[0] != "/usr/local/bin/sbatch" for call in second_calls)
    assert sum(call[0] == "/usr/local/bin/scancel" for call in second_calls) == 1
    assert not (submission / "REPORT_REPAIR_0002_AUTHORIZED.json").exists()


def test_controller_and_worker_share_strict_original_failure_timeline(
    repair, report
):
    valid = {
        "Submit": "2026-08-29T08:00:00",
        "Eligible": "2026-08-29T08:00:01",
        "Start": "2026-08-29T08:28:49",
        "End": "2026-08-29T08:34:44",
    }
    invalid = {**valid, "Submit": "2026-08-29T08:30:00"}
    assert repair._original_report_timeline_is_ordered(valid)
    assert report._original_report_timeline_is_ordered(valid)
    assert not repair._original_report_timeline_is_ordered(invalid)
    assert not report._original_report_timeline_is_ordered(invalid)


def test_report_audit_reassembly_can_cross_repair_stop_but_never_publish(
    report, tmp_path, monkeypatch
):
    snapshot = tmp_path / "snapshot"
    submission = tmp_path / "submission"
    snapshot.mkdir()
    (submission / "journal").mkdir(parents=True)
    (submission / "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json").write_bytes(
        b"fixture"
    )
    monkeypatch.setattr(report, "reject_environment", lambda: None)
    reached = []

    def stop_after_cleanup_boundary(*_args, **_kwargs):
        reached.append(True)
        raise RuntimeError("audit crossed repair stop boundary")

    monkeypatch.setattr(report, "verify_snapshot_inventory", stop_after_cleanup_boundary)
    with pytest.raises(report.ReportError, match="terminal report repair state"):
        report.assemble_report(snapshot, submission, "a" * 64)
    assert reached == []
    with pytest.raises(report.ReportError, match="cannot authorize publication"):
        report.assemble_report(
            snapshot,
            submission,
            "a" * 64,
            require_publish_job=True,
            allow_repair_cleanup_for_audit=True,
        )
    assert reached == []
    with pytest.raises(RuntimeError, match="crossed repair stop boundary"):
        report.assemble_report(
            snapshot,
            submission,
            "a" * 64,
            allow_repair_cleanup_for_audit=True,
        )
    assert reached == [True]


@pytest.mark.parametrize(
    "rows,job_id",
    [
        (["444444", "555555"], "444444"),
        (["33311218"], "33311218"),
    ],
)
def test_lost_response_submitted_validator_requires_sole_nonhistorical_job(
    repair, rows, job_id
):
    census = _census(repair, [_repair_row(repair, job_id=value) for value in rows])
    submitted = {
        "schema_version": 1,
        "status": "held_report_repair_submitted",
        "campaign_id": repair.CAMPAIGN_ID,
        "submission_sha256": repair.EXPECTED_SUBMISSION_SHA256,
        "attempt": repair.ATTEMPT,
        "submit_calling_sha256": "a" * 64,
        "repair_report_job_id": job_id,
        "submission_evidence": {
            "mode": "lost_response_census_adoption",
            "census": census,
            "census_sha256": repair.stable_hash(census),
        },
        "accepted_at_utc": "2026-08-29T00:00:00Z",
    }
    with pytest.raises(repair.RepairError, match="not exact and held"):
        repair._validate_submitted(
            submitted,
            submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
            calling_sha256="a" * 64,
            calling={},
            contract=_scheduler_contract(),
        )


def test_authorization_validator_requires_sole_settled_scheduler_authority(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    submission.mkdir()
    contract = {
        "snapshot_root": str(submission / "source-snapshot/repo"),
        "snapshot_inventory_sha256": "3" * 64,
        "package_protocol_sha256": repair.EXPECTED_ORIGINAL_PROTOCOL,
        "git_provenance": _actual_git_provenance(),
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"},
    }
    receipt = {"report_job_id": repair.EXPECTED_ORIGINAL_REPORT_JOB_ID}
    receipt_map = {"schema_version": 1, "files": {}}
    source = _repair_archive_source(repair, submission)
    census = _census(
        repair,
        [_repair_row(repair), _repair_row(repair, job_id="555555")],
    )
    authorization = repair._authorization_value(
        submission_root=submission,
        submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
        contract=contract,
        receipt=receipt,
        receipt_map=receipt_map,
        expected_reassembly=repair._expected_reassembly(),
        source=source,
        failure_sha256="a" * 64,
        predecessor_sha256="d" * 64,
        calling_sha256="b" * 64,
        submitted_sha256="c" * 64,
        job_id="444444",
        census=census,
        job_control={"schema_version": 1, "fixture": "held_no_requeue"},
        report_installation_method=repair.PUBLICATION_ARCHIVE_INSTALL_METHOD,
    )
    with pytest.raises(repair.RepairError, match="one exact held job"):
        repair._validate_authorization(
            authorization,
            submission_root=submission,
            submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
            contract=contract,
            receipt=receipt,
            receipt_map=receipt_map,
            expected_reassembly=repair._expected_reassembly(),
            source=source,
            failure_sha256="a" * 64,
            predecessor_sha256="d" * 64,
            calling_sha256="b" * 64,
            submitted_sha256="c" * 64,
        )


@pytest.mark.parametrize(
    "name",
    [
        "REPORT_REPAIR_0002_TERMINAL_SUBMIT_FAILURE.json",
        "REPORT_REPAIR_0002_TERMINAL_RELEASE_DENIED.json",
        "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json",
        "CALLING_REPORT_REPAIR_0002_SCANCEL_0000_0000.json",
        "REPORT_REPAIR_0002_SCANCEL_RESULT_0000_0000.json",
        "REPORT_REPAIR_0002_CANCEL_TERMINAL_0000.json",
        "CALLING_REPORT_REPAIR_0002_SUBMIT.json",
        "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json",
    ],
)
def test_repair_terminal_or_cleanup_prefix_blocks_publication(report, tmp_path, name):
    journal = tmp_path / "journal"
    journal.mkdir()
    (journal / name).write_bytes(b"fixture")
    assert report._durable_cleanup_prefix_exists(tmp_path)


def test_ambiguous_release_result_is_a_worker_stop_prefix(report, tmp_path):
    journal = tmp_path / "journal"
    journal.mkdir()
    report.seal_json(
        tmp_path / "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json",
        {
            "schema_version": 1,
            "mode": "lost_response_reconciled_ambiguous_identity",
        },
    )
    assert report._durable_repair_stop_prefix_exists(tmp_path)


def _sealed_json_payload(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


@pytest.mark.parametrize("module_fixture", ["repair", "report"])
@pytest.mark.parametrize(
    "crash_state",
    [
        "partial_0600_stage",
        "sealed_0444_stage",
        "linked_stage_and_target",
        "target_only_after_stage_unlink",
    ],
)
def legacy_atomic_json_sealer_recovers_every_durable_crash_prefix(
    request, tmp_path, module_fixture, crash_state
):
    module = request.getfixturevalue(module_fixture)
    parent = tmp_path / module_fixture
    parent.mkdir(mode=0o700)
    target = parent / "REPORT_REPAIR_0002_AUTHORIZED.json"
    stage = parent / f".{target.name}.seal.tmp"
    value = {
        "schema_version": 1,
        "status": "authorized_terminal_report_repair",
        "attempt": 2,
    }
    payload = _sealed_json_payload(value)
    if crash_state == "partial_0600_stage":
        stage.write_bytes(payload[: max(1, len(payload) // 3)])
        stage.chmod(0o600)
    elif crash_state == "sealed_0444_stage":
        stage.write_bytes(payload)
        stage.chmod(0o444)
    elif crash_state == "linked_stage_and_target":
        stage.write_bytes(payload)
        stage.chmod(0o444)
        os.link(stage, target)
        assert stage.lstat().st_nlink == target.lstat().st_nlink == 2
    else:
        target.write_bytes(payload)
        target.chmod(0o444)

    expected_sha = hashlib.sha256(payload).hexdigest()
    assert module.seal_json(target, value) == expected_sha
    assert target.read_bytes() == payload
    assert stat.S_IMODE(target.lstat().st_mode) == 0o444
    assert target.lstat().st_uid == os.getuid()
    assert target.lstat().st_nlink == 1
    assert not os.path.lexists(stage)
    # A third invocation is the post-unlink/pre-parent-fsync recovery shape.
    assert module.seal_json(target, value) == expected_sha
    assert target.read_bytes() == payload


@pytest.mark.parametrize("module_fixture", ["repair", "report"])
def legacy_atomic_json_sealer_handles_forced_short_writes(
    request, tmp_path, monkeypatch, module_fixture
):
    module = request.getfixturevalue(module_fixture)
    parent = tmp_path / module_fixture
    parent.mkdir(mode=0o700)
    target = parent / "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json"
    value = {
        "schema_version": 1,
        "status": "report_repair_release_attempt_observed",
        "payload": "x" * 4096,
    }
    real_write = os.write
    writes = []

    def short_write(descriptor, payload):
        size = min(len(payload), 17)
        writes.append(size)
        return real_write(descriptor, payload[:size])

    monkeypatch.setattr(module.os, "write", short_write)
    assert module.seal_json(target, value) == hashlib.sha256(
        _sealed_json_payload(value)
    ).hexdigest()
    assert len(writes) > 2
    assert target.read_bytes() == _sealed_json_payload(value)


@pytest.mark.parametrize("module_fixture", ["repair", "report"])
@pytest.mark.parametrize("forgery", ["hardlink", "symlink", "fifo", "mode"])
def legacy_atomic_json_sealer_rejects_hostile_staging_identity(
    request, tmp_path, module_fixture, forgery
):
    module = request.getfixturevalue(module_fixture)
    parent = tmp_path / module_fixture
    parent.mkdir(mode=0o700)
    target = parent / "REPORT_REPAIR_0002_COMPLETED.json"
    stage = parent / f".{target.name}.seal.tmp"
    if forgery == "hardlink":
        external = tmp_path / f"external-{module_fixture}"
        external.write_bytes(b"hostile")
        external.chmod(0o600)
        os.link(external, stage)
    elif forgery == "symlink":
        stage.symlink_to(tmp_path / "missing")
    elif forgery == "fifo":
        os.mkfifo(stage, 0o600)
    else:
        stage.write_bytes(b"hostile")
        stage.chmod(0o640)
    error = module.RepairError if module_fixture == "repair" else module.ReportError
    with pytest.raises(error, match="staging"):
        module.seal_json(target, {"schema_version": 1})
    assert os.path.lexists(stage)
    assert not os.path.lexists(target)


@pytest.mark.parametrize("module_fixture", ["repair", "report"])
@pytest.mark.parametrize(
    "forgery",
    ["distinct_0600", "distinct_0444", "wrong_linked_0444", "foreign_stage"],
)
def legacy_atomic_json_sealer_rejects_ambiguous_namespace_untouched(
    request, tmp_path, module_fixture, forgery
):
    module = request.getfixturevalue(module_fixture)
    parent = tmp_path / module_fixture
    parent.mkdir(mode=0o700)
    target = parent / "REPORT_REPAIR_0002_COMPLETED.json"
    stage = parent / f".{target.name}.seal.tmp"
    value = {"schema_version": 1, "status": "fixture"}
    payload = _sealed_json_payload(value)
    if forgery == "foreign_stage":
        foreign = parent / ".FOREIGN.json.seal.tmp"
        foreign.write_bytes(b"preserve")
        foreign.chmod(0o600)
    else:
        stage.write_bytes(
            b"wrong\n" if forgery == "wrong_linked_0444" else payload
        )
        stage.chmod(0o600 if forgery == "distinct_0600" else 0o444)
        target.write_bytes(payload if "distinct" in forgery else b"wrong\n")
        target.chmod(0o444)
        if forgery == "wrong_linked_0444":
            target.unlink()
            os.link(stage, target)
    before = {
        path.name: (
            path.read_bytes(),
            path.lstat().st_dev,
            path.lstat().st_ino,
            path.lstat().st_nlink,
            stat.S_IMODE(path.lstat().st_mode),
        )
        for path in parent.iterdir()
    }
    error = module.RepairError if module_fixture == "repair" else module.ReportError
    with pytest.raises(error):
        module.seal_json(target, value)
    after = {
        path.name: (
            path.read_bytes(),
            path.lstat().st_dev,
            path.lstat().st_ino,
            path.lstat().st_nlink,
            stat.S_IMODE(path.lstat().st_mode),
        )
        for path in parent.iterdir()
    }
    assert after == before


@pytest.mark.parametrize("module_fixture", ["repair", "report"])
def legacy_atomic_json_sealer_partial_stage_swap_is_rejected_untouched(
    request, tmp_path, monkeypatch, module_fixture
):
    module = request.getfixturevalue(module_fixture)
    parent = tmp_path / module_fixture
    parent.mkdir(mode=0o700)
    target = parent / "REPORT_REPAIR_0002_COMPLETED.json"
    stage = parent / f".{target.name}.seal.tmp"
    displaced = parent / ".displaced-partial"
    stage.write_bytes(b"partial-original")
    stage.chmod(0o600)
    real_fsync = module.os.fsync
    swapped = False

    def swap_after_partial_fsync(descriptor):
        nonlocal swapped
        result = real_fsync(descriptor)
        opened = os.fstat(descriptor)
        if not swapped and stat.S_ISREG(opened.st_mode):
            swapped = True
            stage.rename(displaced)
            stage.write_bytes(b"partial-replacement")
            stage.chmod(0o600)
        return result

    monkeypatch.setattr(module.os, "fsync", swap_after_partial_fsync)
    error = module.RepairError if module_fixture == "repair" else module.ReportError
    with pytest.raises(error, match="partial staging changed"):
        module.seal_json(target, {"schema_version": 1})
    assert stage.read_bytes() == b"partial-replacement"
    assert displaced.read_bytes() == b"partial-original"
    assert not target.exists()


def test_controller_rejects_foreign_journal_seal_stage(repair, tmp_path):
    submission = tmp_path / "submission"
    journal = submission / "journal"
    journal.mkdir(parents=True, mode=0o700)
    foreign = journal / ".REPORT_REPAIR_0003_AUTHORIZED.json.seal.tmp"
    foreign.write_bytes(b"partial")
    foreign.chmod(0o600)
    with pytest.raises(
        repair.RepairError,
        match="journal staging namespace|journal staging is permanent fail-stop",
    ):
        repair._repair_journal_seal_staging_names(submission)
    assert foreign.read_bytes() == b"partial"


@pytest.mark.parametrize(
    "name",
    [
        "CALLING_REPORT_REPAIR_0002_SUBMIT.json",
        "REPORT_REPAIR_0002_AUTHORIZED.json",
        "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json",
        "REPORT_REPAIR_0002_COMPLETED.json",
    ],
)
def test_controller_json_sealer_is_idempotent_across_journal_classes(
    repair, tmp_path, name
):
    journal = tmp_path / "journal"
    journal.mkdir(mode=0o700)
    target = journal / name
    value = {"schema_version": 1, "status": name.removesuffix(".json")}
    first = repair.seal_json(target, value)
    before = target.read_bytes()
    assert repair.seal_json(target, value) == first
    assert target.read_bytes() == before
    assert target.lstat().st_nlink == 1


def _different_json_value(value):
    if isinstance(value, bool):
        return not value
    if type(value) is int:
        return value + 1
    if isinstance(value, str):
        return value + "-forged"
    if isinstance(value, list):
        return [*value, "forged"]
    if isinstance(value, dict):
        return {**value, "forged": True}
    return "forged"


def _synthetic_recovered_report_tree(
    repair,
    tmp_path,
    monkeypatch,
    *,
    provenance_v1_mutation=None,
    authority_mutation=None,
    provenance_extra=None,
    authority_extra=None,
    archive_forgery=None,
):
    submission = tmp_path / "submission"
    journal = submission / "journal"
    journal.mkdir(parents=True, mode=0o700)
    submission_sha256 = "a" * 64

    original_failure_sha = repair.seal_json(
        journal / "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json",
        {"schema_version": 1, "status": "synthetic_original_failure"},
    )
    predecessor_sha = repair.seal_json(
        submission / "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json",
        {"schema_version": 1, "status": "synthetic_attempt2_predecessor"},
    )
    authorization = {
        "original_report_job_id": repair.EXPECTED_ORIGINAL_REPORT_JOB_ID,
        "repair_report_job_id": "444444",
        "original_failure_evidence": (
            "journal/REPORT_REPAIR_0001_ORIGINAL_FAILURE.json"
        ),
        "original_failure_evidence_sha256": original_failure_sha,
        "predecessor_failure_evidence": (
            "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
        ),
        "predecessor_failure_evidence_sha256": predecessor_sha,
        "worker_receipt_map_sha256": "2" * 64,
        "snapshot_root": str(submission / "source-snapshot/repo"),
        "snapshot_inventory_sha256": repair.EXPECTED_SNAPSHOT_INVENTORY_SHA256,
        "original_package_protocol_sha256": repair.EXPECTED_ORIGINAL_PROTOCOL,
        "repair_source_root": str(submission / repair.SOURCE_ARCHIVE_NAME),
        "repair_source_commit": "5" * 40,
        "repair_package_protocol_sha256": "6" * 64,
        "repair_source_files_sha256": "7" * 64,
        "repair_source_installation_method": repair.SOURCE_ARCHIVE_INSTALL_METHOD,
        "report_publication_installation_method": (
            repair.PUBLICATION_ARCHIVE_INSTALL_METHOD
        ),
        "scheduler_job_control_observation_sha256": "c" * 64,
        "worker_handoff": copy.deepcopy(repair.REPAIR_WORKER_HANDOFF),
    }
    authorization_sha = repair.seal_json(
        submission / "REPORT_REPAIR_0002_AUTHORIZED.json", authorization
    )
    release = {
        "schema_version": 1,
        "status": "report_repair_released",
        "authorization_sha256": authorization_sha,
    }
    release_sha = repair.seal_json(
        submission / "REPORT_REPAIR_0002_RELEASED.json", release
    )
    bundle = {"schema_version": 1, "status": "synthetic_rejected_bundle"}
    gate_body = {"status": "rejected", "reason": "synthetic exact fixture"}
    gate = {**gate_body, "gate_sha256": repair.stable_hash(gate_body)}
    bundle_file = _sealed_json_payload(bundle)
    gate_file = _sealed_json_payload(gate)
    bundle_sha = repair.stable_hash(bundle)
    gate_sha = gate["gate_sha256"]
    monkeypatch.setattr(repair, "EXPECTED_BUNDLE_SHA256", bundle_sha)
    monkeypatch.setattr(
        repair, "EXPECTED_BUNDLE_FILE_SHA256", hashlib.sha256(bundle_file).hexdigest()
    )
    monkeypatch.setattr(repair, "EXPECTED_BUNDLE_FILE_SIZE", len(bundle_file))
    monkeypatch.setattr(repair, "EXPECTED_GATE_SHA256", gate_sha)
    monkeypatch.setattr(
        repair,
        "EXPECTED_DECISION_FILE_SHA256",
        hashlib.sha256(gate_file).hexdigest(),
    )
    monkeypatch.setattr(repair, "EXPECTED_DECISION_FILE_SIZE", len(gate_file))

    prerequisite = {"schema_version": 1, "accepted": True}
    provenance_v1 = {
        "schema_version": 1,
        "campaign_id": repair.CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "production_authorization_prerequisite": prerequisite,
        "production_authorization_prerequisite_sha256": repair.stable_hash(
            prerequisite
        ),
        "outcome_blind_phase": {"status": "complete"},
        "event_artifacts": [{"path": "events", "sha256": "d" * 64}],
        "terminal_artifacts": [{"path": "terminal", "sha256": "e" * 64}],
        "report_bundle_sha256": bundle_sha,
        "gate_sha256": gate_sha,
    }
    monkeypatch.setattr(
        repair,
        "EXPECTED_PROVENANCE_V1_SHA256",
        repair.stable_hash(provenance_v1),
    )
    provenance_v1_file = _sealed_json_payload(provenance_v1)
    monkeypatch.setattr(
        repair,
        "EXPECTED_PROVENANCE_V1_FILE_SHA256",
        hashlib.sha256(provenance_v1_file).hexdigest(),
    )
    monkeypatch.setattr(
        repair, "EXPECTED_PROVENANCE_V1_FILE_SIZE", len(provenance_v1_file)
    )

    authority = repair._expected_publication_authority(
        authorization, authorization_sha, release_sha
    )
    provenance = copy.deepcopy(provenance_v1)
    provenance["schema_version"] = 2
    provenance["publication_authority"] = copy.deepcopy(authority)
    if provenance_v1_mutation is not None:
        key = provenance_v1_mutation
        if key == "schema_version":
            provenance[key] = 3
        else:
            provenance[key] = _different_json_value(provenance[key])
    if authority_mutation is not None:
        provenance["publication_authority"][authority_mutation] = (
            _different_json_value(
                provenance["publication_authority"][authority_mutation]
            )
        )
    if provenance_extra is not None:
        provenance[provenance_extra] = {"forged": True}
    if authority_extra is not None:
        provenance["publication_authority"][authority_extra] = {
            "encoding": "base64",
            "data": "Zm9yZ2Vk",
        }

    bundle_name = f"REPORT_BUNDLE.{bundle_sha}.json"
    gate_name = f"GATE_DECISION.{gate_sha}.json"
    provenance_name = (
        f"REPORT_PROVENANCE.{repair.stable_hash(provenance)}.json"
    )
    provenance_file = _sealed_json_payload(provenance)
    if archive_forgery == "compact_provenance":
        provenance_file = (
            json.dumps(provenance, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
    if archive_forgery == "alternate_bundle_name":
        bundle_name = "ALTERNATE_report_bundle.json"
    elif archive_forgery == "alternate_gate_name":
        gate_name = "ALTERNATE_gate_decision.json"
    elif archive_forgery == "alternate_provenance_name":
        provenance_name = "ALTERNATE_provenance.json"
    bundle_file_sha = hashlib.sha256(bundle_file).hexdigest()
    gate_file_sha = hashlib.sha256(gate_file).hexdigest()
    provenance_file_sha = hashlib.sha256(provenance_file).hexdigest()
    commit = {
        "schema_version": 1,
        "status": "rejected",
        "scientific_rejection": True,
        "campaign_id": repair.CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "report_bundle": bundle_name,
        "report_bundle_sha256": bundle_sha,
        "report_bundle_file_sha256": bundle_file_sha,
        "gate_decision": gate_name,
        "gate_sha256": gate_sha,
        "gate_decision_file_sha256": gate_file_sha,
        "provenance": provenance_name,
        "provenance_sha256": repair.stable_hash(provenance),
        "provenance_file_sha256": provenance_file_sha,
    }
    commit_file = _sealed_json_payload(commit)
    if archive_forgery == "compact_commit":
        commit_file = (
            json.dumps(commit, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
    entry_values = {
        "report_bundle": (
            bundle_name,
            bundle_file,
            bundle_sha,
        ),
        "gate_decision": (
            gate_name,
            gate_file,
            gate_sha,
        ),
        "provenance": (
            provenance_name,
            provenance_file,
            repair.stable_hash(provenance),
        ),
        "report_commit": (
            "REPORT_COMMIT.json",
            commit_file,
            repair.stable_hash(commit),
        ),
    }
    entries = [
        {
            "kind": kind,
            "name": entry_values[kind][0],
            "size": len(entry_values[kind][1]),
            "sha256": hashlib.sha256(entry_values[kind][1]).hexdigest(),
            "logical_sha256": entry_values[kind][2],
        }
        for kind in repair.PUBLICATION_ARCHIVE_ENTRY_ORDER
    ]
    header = {
        "archive_kind": repair.PUBLICATION_ARCHIVE_KIND,
        "schema_version": 2,
        "campaign_id": repair.CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "entry_order": list(repair.PUBLICATION_ARCHIVE_ENTRY_ORDER),
        "entries": entries,
        "report_commit_sha256": hashlib.sha256(commit_file).hexdigest(),
        "report_commit_value_sha256": repair.stable_hash(commit),
    }
    header_file = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    blocks = [
        repair.PUBLICATION_ARCHIVE_MAGIC,
        len(header_file).to_bytes(8, "big"),
        header_file,
    ]
    for kind in repair.PUBLICATION_ARCHIVE_ENTRY_ORDER:
        name, payload, _logical = entry_values[kind]
        name_bytes = name.encode("ascii")
        blocks.extend(
            [
                len(name_bytes).to_bytes(8, "big"),
                name_bytes,
                len(payload).to_bytes(8, "big"),
                payload,
            ]
        )
    archive_payload = b"".join(blocks)
    archive_digest = hashlib.sha256(archive_payload).hexdigest()
    archive_path = submission / (
        f"{repair.PUBLICATION_ARCHIVE_PREFIX}{archive_digest}"
        f"{repair.PUBLICATION_ARCHIVE_SUFFIX}"
    )
    archive_path.write_bytes(archive_payload)
    archive_path.chmod(0o444)
    return submission, submission_sha256, commit


def _replace_synthetic_publication_archive(repair, submission, payload):
    archives = list(
        submission.glob(
            f"{repair.PUBLICATION_ARCHIVE_PREFIX}*"
            f"{repair.PUBLICATION_ARCHIVE_SUFFIX}"
        )
    )
    assert len(archives) == 1
    archives[0].unlink()
    digest = hashlib.sha256(payload).hexdigest()
    path = submission / (
        f"{repair.PUBLICATION_ARCHIVE_PREFIX}{digest}"
        f"{repair.PUBLICATION_ARCHIVE_SUFFIX}"
    )
    path.write_bytes(payload)
    path.chmod(0o444)
    return path


@pytest.mark.parametrize(
    ("header_size", "message"),
    [
        (1 << 20, "publication archive header differs"),
        ((1 << 20) + 1, "publication archive header size differs"),
    ],
)
def test_publication_archive_header_size_cap_is_exact(
    repair, tmp_path, monkeypatch, header_size, message
):
    submission, submission_sha256, _commit = _synthetic_recovered_report_tree(
        repair, tmp_path, monkeypatch
    )
    archive = next(submission.glob("REPORT_REPAIR_0002_PUBLICATION.*.archive"))
    raw = archive.read_bytes()
    prefix_size = len(repair.PUBLICATION_ARCHIVE_MAGIC)
    old_size = int.from_bytes(raw[prefix_size : prefix_size + 8], "big")
    suffix = raw[prefix_size + 8 + old_size :]
    forged = (
        repair.PUBLICATION_ARCHIVE_MAGIC
        + header_size.to_bytes(8, "big")
        + b" " * header_size
        + suffix
    )
    _replace_synthetic_publication_archive(repair, submission, forged)
    with pytest.raises(repair.RepairError, match=message):
        repair._validated_repaired_report_tree(submission, submission_sha256)


@pytest.mark.parametrize(
    ("name_size", "message"),
    [
        (256, "publication archive frame differs"),
        (257, "publication archive entry name size differs"),
    ],
)
def test_publication_archive_member_name_size_cap_is_exact(
    repair, tmp_path, monkeypatch, name_size, message
):
    submission, submission_sha256, _commit = _synthetic_recovered_report_tree(
        repair, tmp_path, monkeypatch
    )
    archive = next(submission.glob("REPORT_REPAIR_0002_PUBLICATION.*.archive"))
    raw = archive.read_bytes()
    cursor = len(repair.PUBLICATION_ARCHIVE_MAGIC)
    header_size = int.from_bytes(raw[cursor : cursor + 8], "big")
    cursor += 8 + header_size
    old_name_size = int.from_bytes(raw[cursor : cursor + 8], "big")
    old_name_end = cursor + 8 + old_name_size
    forged = (
        raw[:cursor]
        + name_size.to_bytes(8, "big")
        + b"x" * name_size
        + raw[old_name_end:]
    )
    _replace_synthetic_publication_archive(repair, submission, forged)
    with pytest.raises(repair.RepairError, match=message):
        repair._validated_repaired_report_tree(submission, submission_sha256)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("trailing", "trailing bytes"),
        ("truncated", "truncated"),
        ("frame_size", "publication (?:entry hash|archive frame) differs"),
    ],
)
def test_publication_archive_member_framing_is_exact(
    repair, tmp_path, monkeypatch, mutation, message
):
    submission, submission_sha256, _commit = _synthetic_recovered_report_tree(
        repair, tmp_path, monkeypatch
    )
    archive = next(submission.glob("REPORT_REPAIR_0002_PUBLICATION.*.archive"))
    raw = archive.read_bytes()
    if mutation == "trailing":
        forged = raw + b"x"
    elif mutation == "truncated":
        forged = raw[:-1]
    else:
        cursor = len(repair.PUBLICATION_ARCHIVE_MAGIC)
        header_size = int.from_bytes(raw[cursor : cursor + 8], "big")
        cursor += 8 + header_size
        name_size = int.from_bytes(raw[cursor : cursor + 8], "big")
        cursor += 8 + name_size
        frame_size = int.from_bytes(raw[cursor : cursor + 8], "big")
        forged = (
            raw[:cursor]
            + (frame_size + 1).to_bytes(8, "big")
            + raw[cursor + 8 :]
        )
    _replace_synthetic_publication_archive(repair, submission, forged)
    with pytest.raises(repair.RepairError, match=message):
        repair._validated_repaired_report_tree(submission, submission_sha256)


@pytest.mark.parametrize(
    "key",
    [
        "schema_version",
        "campaign_id",
        "submission_sha256",
        "production_authorization_prerequisite",
        "production_authorization_prerequisite_sha256",
        "outcome_blind_phase",
        "event_artifacts",
        "terminal_artifacts",
        "report_bundle_sha256",
        "gate_sha256",
    ],
)
def test_recovered_report_rejects_every_provenance_v1_field_mutation(
    repair, tmp_path, monkeypatch, key
):
    submission, submission_sha256, _commit = _synthetic_recovered_report_tree(
        repair, tmp_path, monkeypatch, provenance_v1_mutation=key
    )
    with pytest.raises(repair.RepairError, match="provenance"):
        repair._validated_repaired_report_tree(submission, submission_sha256)


@pytest.mark.parametrize(
    "key",
    [
        "schema_version",
        "status",
        "attempt",
        "authorization",
        "authorization_sha256",
        "release",
        "release_sha256",
        "original_report_job_id",
        "repair_report_job_id",
        "original_failure_evidence",
        "original_failure_evidence_sha256",
        "predecessor_failure_evidence",
        "predecessor_failure_evidence_sha256",
        "attempt1_environment_evidence",
        "worker_receipt_map_sha256",
        "original_snapshot_root",
        "original_snapshot_inventory_sha256",
        "original_package_protocol_sha256",
        "repair_source_root",
        "repair_source_commit",
        "repair_package_protocol_sha256",
        "repair_source_files_sha256",
        "repair_source_installation_method",
        "report_publication_installation_method",
        "scheduler_job_control_observation_sha256",
        "worker_handoff_sha256",
        "expected_report_bundle_sha256",
        "expected_report_bundle_file_sha256",
        "expected_gate_sha256",
        "expected_gate_decision_file_sha256",
        "deterministic_reassembly_allowed",
        "scientific_input_change_allowed",
        "gate_change_allowed",
    ],
)
def test_recovered_report_rejects_every_public_authority_field_mutation(
    repair, tmp_path, monkeypatch, key
):
    submission, submission_sha256, _commit = _synthetic_recovered_report_tree(
        repair, tmp_path, monkeypatch, authority_mutation=key
    )
    with pytest.raises(repair.RepairError, match="public(?:ation)? authority"):
        repair._validated_repaired_report_tree(submission, submission_sha256)


@pytest.mark.parametrize(
    ("provenance_extra", "authority_extra"),
    [("unexpected_top_level", None), (None, "retained_environment_evidence")],
)
def test_recovered_report_rejects_extra_or_raw_environment_provenance(
    repair, tmp_path, monkeypatch, provenance_extra, authority_extra
):
    submission, submission_sha256, _commit = _synthetic_recovered_report_tree(
        repair,
        tmp_path,
        monkeypatch,
        provenance_extra=provenance_extra,
        authority_extra=authority_extra,
    )
    with pytest.raises(
        repair.RepairError, match="provenance|public(?:ation)? authority"
    ):
        repair._validated_repaired_report_tree(submission, submission_sha256)


def test_recovered_report_accepts_exact_v1_and_public_authority_reconstruction(
    repair, tmp_path, monkeypatch
):
    submission, submission_sha256, commit = _synthetic_recovered_report_tree(
        repair, tmp_path, monkeypatch
    )
    assert repair._validated_repaired_report_tree(
        submission, submission_sha256
    ) == commit


@pytest.mark.parametrize(
    "forgery",
    [
        "compact_commit",
        "compact_provenance",
        "alternate_bundle_name",
        "alternate_gate_name",
        "alternate_provenance_name",
    ],
)
def test_controller_recovered_report_requires_worker_canonical_names_and_bytes(
    repair, tmp_path, monkeypatch, forgery
):
    submission, submission_sha256, _commit = _synthetic_recovered_report_tree(
        repair, tmp_path, monkeypatch, archive_forgery=forgery
    )
    with pytest.raises(repair.RepairError, match="commit|provenance|scientific entries"):
        repair._validated_repaired_report_tree(submission, submission_sha256)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"444444\n", {"schema_version": 1, "job_id": "444444", "cluster": None}),
        (
            b"444444;cluster-1.example\n",
            {
                "schema_version": 1,
                "job_id": "444444",
                "cluster": "cluster-1.example",
            },
        ),
    ],
)
def test_controller_and_worker_share_exact_sbatch_parsable_grammar(
    repair, report, payload, expected
):
    assert repair._parsed_sbatch_stdout(payload) == expected
    assert report._validated_repair_sbatch_stdout(payload) == expected
    assert repair._parse_sbatch_job_id(
        _command_result(repair, payload)
    ) == expected["job_id"]


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"444444",
        b" 444444\n",
        b"444444 \n",
        b"444444\r\n",
        b"444444\n555555\n",
        b"444444;\n",
        b"444444;bad cluster\n",
        b"444444;one;two\n",
        b"044444\n",
        "１２３４４４\n".encode("utf-8"),
    ],
)
def test_controller_and_worker_reject_malformed_sbatch_parsable_stdout(
    repair, report, payload
):
    with pytest.raises(repair.RepairError):
        repair._parsed_sbatch_stdout(payload)
    with pytest.raises(report.ReportError):
        report._validated_repair_sbatch_stdout(payload)


@pytest.mark.parametrize(
    "foreign_name",
    [
        ".report.install-probe-foreign",
        ".report.install-probe-source.extra",
        ".report.install-probe-target.old",
    ],
)
def test_foreign_report_install_probe_blocks_controller_before_scheduler_or_write(
    repair, tmp_path, monkeypatch, foreign_name
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )
    foreign = submission / foreign_name
    foreign.write_bytes(b"preserve")
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (submission / "journal").iterdir()
        if path.is_file()
    }

    def forbidden_scheduler(*_args, **_kwargs):
        raise AssertionError("foreign install probe reached scheduler")

    with pytest.raises(
        repair.RepairError,
        match="install-probe|staging/probe residue",
    ):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=forbidden_scheduler,
            sleep=lambda _seconds: None,
        )
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (submission / "journal").iterdir()
        if path.is_file()
    }
    assert after == before
    assert foreign.read_bytes() == b"preserve"


def _make_exact_repair_attempt_namespace(repair, tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir(mode=0o700)
    (submission / "journal").mkdir(mode=0o700)
    predecessor = (
        submission / "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
    )
    predecessor.write_bytes(b"fixture\n")
    predecessor.chmod(0o444)
    repair_parent = submission / "report-repair"
    repair_parent.mkdir(mode=0o700)
    attempt_root = repair_parent / "attempt-0001"
    attempt_root.mkdir(mode=0o700)
    source = attempt_root / "source"
    source.mkdir(mode=0o700)
    source.chmod(0o555)
    _write_source_archive_fixture(
        submission, b"print('namespace fixture')\n", write_authorization=False
    )
    return submission


def test_controller_and_worker_accept_exact_two_attempt_source_namespace(
    repair, report, tmp_path
):
    submission = _make_exact_repair_attempt_namespace(repair, tmp_path)
    repair._require_repair_filesystem_namespace(
        submission, source_must_be_installed=True
    )
    report._validated_report_repair_filesystem_namespace(submission)


@pytest.mark.parametrize(
    "forgery",
    [
        "attempt3",
        "attempt1_peer",
        "attempt2_peer",
        "repair_parent_mode",
        "attempt_root_mode",
        "source_root_mode",
        "source_root_symlink",
    ],
)
def test_controller_and_worker_reject_repair_attempt_namespace_forgery(
    repair, report, tmp_path, forgery
):
    submission = _make_exact_repair_attempt_namespace(repair, tmp_path)
    repair_parent = submission / "report-repair"
    attempt1 = repair_parent / "attempt-0001"
    if forgery == "attempt3":
        (repair_parent / "attempt-0003").mkdir(mode=0o700)
    elif forgery == "attempt1_peer":
        (attempt1 / "unexpected").write_bytes(b"forged")
    elif forgery == "attempt2_peer":
        (repair_parent / "attempt-0002").mkdir(mode=0o700)
    elif forgery == "repair_parent_mode":
        repair_parent.chmod(0o755)
    elif forgery == "attempt_root_mode":
        attempt1.chmod(0o755)
    elif forgery == "source_root_mode":
        (attempt1 / "source").chmod(0o755)
    else:
        source = attempt1 / "source"
        source.chmod(0o700)
        source.rmdir()
        source.symlink_to(tmp_path / "external-source")
    with pytest.raises(repair.RepairError):
        repair._require_repair_filesystem_namespace(
            submission, source_must_be_installed=True
        )
    with pytest.raises(report.ReportError):
        report._validated_report_repair_filesystem_namespace(submission)


def test_attempt2_source_authority_requires_exact_raw_serialization(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    source_archive, _digest, _size = _write_source_archive_fixture(
        submission,
        b"print('canonical source archive')\n",
        write_authorization=False,
    )
    payload = source_archive.read_bytes()
    prefix, tail = payload.split(repair.SOURCE_ARCHIVE_MARKER, 1)
    body = tail[: -len(repair.SOURCE_ARCHIVE_END)]
    envelope = json.loads(body.decode("ascii"))
    forged = (
        prefix
        + repair.SOURCE_ARCHIVE_MARKER
        + json.dumps(
            envelope, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
        ).encode("ascii")
        + repair.SOURCE_ARCHIVE_END
    )
    source_archive.chmod(0o600)
    source_archive.write_bytes(forged)
    source_archive.chmod(0o444)
    with pytest.raises(repair.RepairError, match="source archive envelope"):
        repair._load_sealed_repair_source(source_archive)


@pytest.mark.parametrize(
    ("suffix", "report_present", "message"),
    [
        (
            {"REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"},
            False,
            "attempt1 terminal",
        ),
        (
            {"CALLING_REPORT_REPAIR_0002_SUBMIT.json"},
            False,
            "predecessor",
        ),
        ({"REPORT_REPAIR_0002_SUBMITTED.json"}, False, "submit calling"),
        ({"REPORT_REPAIR_0002_AUTHORIZED.json"}, False, "submitted"),
        (
            {"CALLING_REPORT_REPAIR_0002_RELEASE_0000.json"},
            False,
            "authorization",
        ),
        (
            {
                "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
                "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json",
                "CALLING_REPORT_REPAIR_0002_SUBMIT.json",
                "REPORT_REPAIR_0002_SUBMITTED.json",
                "REPORT_REPAIR_0002_AUTHORIZED.json",
                "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json",
            },
            False,
            "gap-free",
        ),
        (
            {
                "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
                "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json",
                "CALLING_REPORT_REPAIR_0002_SUBMIT.json",
                "REPORT_REPAIR_0002_SUBMITTED.json",
                "REPORT_REPAIR_0002_AUTHORIZED.json",
                "REPORT_REPAIR_0002_RELEASED.json",
            },
            False,
            "paired release",
        ),
        (
            {
                "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
                "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json",
                "CALLING_REPORT_REPAIR_0002_SUBMIT.json",
                "REPORT_REPAIR_0002_SUBMITTED.json",
                "REPORT_REPAIR_0002_AUTHORIZED.json",
                "REPORT_REPAIR_0002_TERMINAL_WORKER_FAILURE.json",
            },
            False,
            "complete release evidence",
        ),
        (set(), True, "authorization/release"),
    ],
)
def test_prefix_graph_rejects_impossible_positive_successors_before_mutation(
    repair, tmp_path, suffix, report_present, message
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True, mode=0o700)
    names = set(repair.EXPECTED_ATTEMPT1_CHAIN_SHA256) | set(suffix)
    with pytest.raises(repair.RepairError, match=message):
        repair._require_repair_prefix_graph(
            submission, sorted(names), report_present=report_present
        )
    assert list((submission / "journal").iterdir()) == []


def test_completed_without_report_blocks_execute_before_scheduler_or_stage_cleanup(
    repair, tmp_path, monkeypatch
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )
    for name in (
        "CALLING_REPORT_REPAIR_0002_SUBMIT.json",
        "REPORT_REPAIR_0002_SUBMITTED.json",
        "REPORT_REPAIR_0002_AUTHORIZED.json",
        "CALLING_REPORT_REPAIR_0002_RELEASE_0000.json",
        "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json",
        "REPORT_REPAIR_0002_RELEASED.json",
        "REPORT_REPAIR_0002_COMPLETED.json",
    ):
        value = _phase_fixture_value(repair, name)
        if name == "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json":
            value["mode"] = "direct_release_response"
        path = submission / name
        path.write_bytes(_sealed_json_payload(value))
        path.chmod(0o444)
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in submission.iterdir()
        if path.is_file()
    }

    def forbidden_scheduler(*_args, **_kwargs):
        raise AssertionError("completed-without-report reached scheduler")

    with pytest.raises(
        repair.RepairError,
        match="without published report|publication|staging/generation namespace",
    ):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=False,
            runner=forbidden_scheduler,
            sleep=lambda _seconds: None,
        )
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in submission.iterdir()
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    "field",
    [
        "JobId",
        "JobName",
        "UserId",
        "JobState",
        "Reason",
        "Requeue",
        "Restarts",
        "BatchFlag",
        "TimeLimit",
        "Comment",
        "Partition",
        "Account",
        "QOS",
        "NumNodes",
        "NumCPUs",
        "NumTasks",
        "CPUs/Task",
        "MinMemoryNode",
        "Command",
        "WorkDir",
        "StdOut",
        "StdErr",
        "StdIn",
        "ArrayJobId",
        "HetJobId",
    ],
)
def test_fresh_held_job_control_rejects_every_identity_or_resource_drift(
    repair, tmp_path, field
):
    submission = tmp_path / "submission"
    submission.mkdir(mode=0o700)
    source = submission / repair.SOURCE_ARCHIVE_NAME
    result = _held_job_control_result(repair, submission)
    text = result.stdout.decode("utf-8").rstrip("\n")
    if field in {"ArrayJobId", "HetJobId"}:
        text += f" {field}=999"
    else:
        marker = f"{field}="
        tokens = text.split()
        index = next(index for index, token in enumerate(tokens) if token.startswith(marker))
        tokens[index] = f"{field}=forged"
        text = " ".join(tokens)
    with pytest.raises(repair.RepairError, match="control-plane identity"):
        repair._job_control_projection(
            (text + "\n").encode("utf-8"),
            submission_root=submission,
            submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
            job_id="444444",
            source_root=source,
        )


def test_new_release_binds_fresh_census_then_scontrol_before_mutation(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True, mode=0o700)
    (submission / "logs").mkdir(mode=0o700)
    contract = _scheduler_contract()
    calls = []
    squeue_round = 0

    locks = _FakeLocks()

    def runner(argv, _cwd, _environment):
        nonlocal squeue_round
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            squeue_round += 1
            row = _repair_row(
                repair,
                state="PENDING" if squeue_round <= 3 else "RUNNING",
                reason="JobHeldUser" if squeue_round <= 3 else "None",
            )
            return _squeue_result(repair, [row])
        if list(argv[:4]) == [
            "/usr/local/bin/scontrol",
            "show",
            "job",
            "-dd",
        ]:
            return _held_job_control_result(repair, submission, argv[4])
        if list(argv[:2]) == ["/usr/local/bin/scontrol", "release"]:
            calling_path = (
                submission
                / "CALLING_REPORT_REPAIR_0002_RELEASE_0000.json"
            )
            calling, calling_sha, info = repair.read_json(
                calling_path, "test release calling"
            )
            assert stat.S_IMODE(info.st_mode) == 0o444
            assert calling_sha == hashlib.sha256(calling_path.read_bytes()).hexdigest()
            assert repair._validate_release_calling(
                calling,
                submission_root=submission,
                submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
                index=0,
                job_id="444444",
                authorization_sha256="a" * 64,
                contract=contract,
                locks=locks,
            ) == calling
            return _command_result(repair)
        raise AssertionError(argv)

    result = _direct_release(repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        runner,
        locks,
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "report_repair_released"
    assert [call[:2] for call in calls] == [
        *([["/usr/local/bin/squeue", "--noheader"]] * 3),
        ["/usr/local/bin/scontrol", "show"],
        ["/usr/local/bin/scontrol", "release"],
        *([["/usr/local/bin/squeue", "--noheader"]] * 3),
    ]


def _assert_controller_and_worker_census_reject(
    repair, report, census, contract
):
    with pytest.raises(repair.RepairError):
        repair._validated_scheduler_census(census, contract)
    expected_environment = repair._scheduler_environment("/tmp/slurm.conf")
    with pytest.raises(report.ReportError):
        report._validated_repair_census(
            census,
            submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
            expected_environment=expected_environment,
            label="hostile shared census",
        )


def test_controller_and_worker_accept_exact_external_census_contract(
    repair, report
):
    contract = _scheduler_contract()
    census = _census(repair, [_repair_row(repair)])
    assert repair._validated_scheduler_census(census, contract) == census
    assert report._validated_repair_census(
        census,
        submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
        expected_environment=repair._scheduler_environment("/tmp/slurm.conf"),
        label="exact shared census",
    ) == census


@pytest.mark.parametrize(
    "forgery",
    ["evidence_owned_environment", "argv_order", "round_order", "unicode_job_id"],
)
def test_controller_and_worker_reject_census_contract_forgery(
    repair, report, forgery
):
    contract = _scheduler_contract()
    if forgery == "unicode_job_id":
        census = _census(repair, [_repair_row(repair, job_id="１２３４４４")])
    else:
        census = _census(repair, [])
    if forgery == "evidence_owned_environment":
        for round_value in census["rounds"]:
            environment = round_value["raw"]["environment"]
            environment["USER"] = environment["LOGNAME"] = "forged-owner"
            round_value["raw"]["argv"][2] = "--user=forged-owner"
    elif forgery == "argv_order":
        for round_value in census["rounds"]:
            argv = round_value["raw"]["argv"]
            argv[2], argv[3] = argv[3], argv[2]
    elif forgery == "round_order":
        census["rounds"][0]["round"] = 1
    _assert_controller_and_worker_census_reject(
        repair, report, census, contract
    )


def test_all_durable_census_consumers_require_external_contract_arguments():
    controller_tree = ast.parse((PACKAGE / "report_repair.py").read_text())
    worker_tree = ast.parse((PACKAGE / "report.py").read_text())
    controller_calls = [
        node
        for node in ast.walk(controller_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validated_scheduler_census"
    ]
    repair_row_calls = [
        node
        for node in ast.walk(controller_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_repair_rows"
    ]
    worker_calls = [
        node
        for node in ast.walk(worker_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validated_repair_census"
    ]
    assert controller_calls and all(len(node.args) >= 2 for node in controller_calls)
    assert repair_row_calls and all(len(node.args) >= 3 for node in repair_row_calls)
    assert worker_calls and all(
        any(keyword.arg == "expected_environment" for keyword in node.keywords)
        for node in worker_calls
    )


def _journal_stage(path, value, *, mode=0o444, link_target=False):
    stage = path.parent / f".{path.name}.seal.tmp"
    payload = _sealed_json_payload(value)
    stage.write_bytes(payload)
    stage.chmod(mode)
    if link_target:
        os.link(stage, path)
    return stage, payload


def legacy_virtual_phase_rejects_authorization_stage_without_submitted_untouched(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    _seed_direct_authorized_phase(repair, submission)
    journal = submission / "journal"
    (submission / "REPORT_REPAIR_0002_AUTHORIZED.json").unlink()
    (submission / "REPORT_REPAIR_0002_SUBMITTED.json").unlink()
    stage = journal / ".REPORT_REPAIR_0002_AUTHORIZED.json.seal.tmp"
    stage.write_bytes(b"partial authorization")
    stage.chmod(0o600)
    before = stage.read_bytes()
    with pytest.raises(repair.RepairError, match="phase-next|submitted"):
        repair._classify_repair_phase(
            submission, source_must_be_installed=True
        )
    assert stage.read_bytes() == before
    assert not (submission / "REPORT_REPAIR_0002_AUTHORIZED.json").exists()


def legacy_virtual_phase_rejects_distinct_target_and_partial_stage_untouched(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    _seed_direct_authorized_phase(repair, submission)
    journal = submission / "journal"
    target = submission / "REPORT_REPAIR_0002_SUBMITTED.json"
    target_before = target.read_bytes()
    stage = journal / f".{target.name}.seal.tmp"
    stage.write_bytes(b"partial successor")
    stage.chmod(0o600)
    stage_before = stage.read_bytes()
    with pytest.raises(
        repair.RepairError,
        match="target/staging identity differs|target and staging are distinct",
    ):
        repair._classify_repair_phase(
            submission, source_must_be_installed=True
        )
    assert target.read_bytes() == target_before
    assert stage.read_bytes() == stage_before


def legacy_phase_next_partial_discard_rejects_stage_inode_swap_untouched(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    _seed_direct_authorized_phase(repair, submission)
    journal = submission / "journal"
    stage = journal / ".REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json.seal.tmp"
    stage.write_bytes(b"original partial")
    stage.chmod(0o600)
    phase = repair._classify_repair_phase(
        submission, source_must_be_installed=True
    )
    displaced = journal / "displaced.partial"
    stage.rename(displaced)
    stage.write_bytes(b"replacement partial")
    stage.chmod(0o600)
    before_replacement = stage.read_bytes()
    before_displaced = displaced.read_bytes()
    with pytest.raises(repair.RepairError, match="binding changed"):
        repair._discard_phase_next_partial_stage(submission, phase)
    assert stage.read_bytes() == before_replacement
    assert displaced.read_bytes() == before_displaced


def legacy_phase_next_partial_discard_rejects_journal_inode_swap_untouched(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    _seed_direct_authorized_phase(repair, submission)
    journal = submission / "journal"
    stage_name = ".REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json.seal.tmp"
    stage = journal / stage_name
    stage.write_bytes(b"original partial")
    stage.chmod(0o600)
    phase = repair._classify_repair_phase(
        submission, source_must_be_installed=True
    )
    displaced = submission / "journal.displaced"
    real_open = repair.os.open
    swapped = False

    def swap_parent_on_stage_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == stage_name and not swapped:
            swapped = True
            journal.rename(displaced)
            journal.mkdir(mode=0o700)
            (journal / "replacement").write_bytes(b"preserve")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(repair.os, "open", swap_parent_on_stage_open)
    with pytest.raises(repair.RepairError, match="binding changed"):
        repair._discard_phase_next_partial_stage(submission, phase)
    assert (displaced / stage_name).read_bytes() == b"original partial"
    assert (journal / "replacement").read_bytes() == b"preserve"


def legacy_virtual_cleanup_authority_partial_dominates_positive_release(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    _seed_direct_authorized_phase(repair, submission)
    stage = (
        submission
        / "journal/.REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json.seal.tmp"
    )
    stage.write_bytes(b"partial cleanup authority")
    stage.chmod(0o600)
    calls = []

    def census_only(argv, _cwd, _environment):
        calls.append(list(argv))
        assert argv[0] == "/usr/local/bin/squeue"
        return _squeue_result(repair, [])

    result = _direct_release(
        repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        _scheduler_contract(),
        {"repair_report_job_id": "444444"},
        "a" * 64,
        census_only,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "report_repair_cleanup_partial_authority_discarded"
    assert not os.path.lexists(stage)
    assert len(calls) == 3
    assert not list(
        (submission / "journal").glob(
            "CALLING_REPORT_REPAIR_0002_RELEASE_*.json"
        )
    )


def legacy_virtual_cleanup_partial_dominates_lost_submit_positive_recovery(
    repair, tmp_path, monkeypatch
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )

    def lose_submit(argv, _cwd, _environment):
        assert argv[0] == "/usr/local/bin/sbatch"
        raise RuntimeError("lost submit response")

    with pytest.raises(RuntimeError, match="lost submit response"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=_with_empty_pre_submit_census(repair, lose_submit),
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    stage = journal / ".REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json.seal.tmp"
    stage.write_bytes(b"partial cleanup authority")
    stage.chmod(0o600)
    calls = []

    def census_only(argv, _cwd, _environment):
        calls.append(list(argv))
        assert argv[0] == "/usr/local/bin/squeue"
        return _squeue_result(repair, [])

    result = repair.execute_report_repair(
        repo,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        allow_initial_submission=False,
        runner=census_only,
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "report_repair_cleanup_partial_authority_discarded"
    assert not stage.exists()
    assert not (submission / "REPORT_REPAIR_0002_SUBMITTED.json").exists()
    assert not (submission / "REPORT_REPAIR_0002_AUTHORIZED.json").exists()
    assert all(call[0] == "/usr/local/bin/squeue" for call in calls)


def legacy_virtual_cleanup_terminal_stage_dominates_positive_release(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    row = _repair_row(repair)
    original_seal = repair.seal_json

    def stop_at_cleanup_terminal(path, value):
        if path.name == "REPORT_REPAIR_0002_CANCEL_TERMINAL_0000.json":
            _journal_stage(path, value)
            raise RuntimeError("cleanup terminal stage")
        return original_seal(path, value)

    def cleanup_runner(argv, _cwd, _environment):
        if argv[0] == "/usr/local/bin/scancel":
            return _command_result(repair)
        if argv[0] == "/usr/local/bin/squeue":
            return _squeue_result(repair, [])
        raise AssertionError(argv)

    monkeypatch.setattr(repair, "seal_json", stop_at_cleanup_terminal)
    with pytest.raises(RuntimeError, match="cleanup terminal stage"):
        _direct_cleanup(
            repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            _scheduler_contract(),
            _census(repair, [row]),
            "fixture_cleanup",
            cleanup_runner,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    monkeypatch.setattr(repair, "seal_json", original_seal)
    calls = []

    def census_only(argv, _cwd, _environment):
        calls.append(list(argv))
        assert argv[0] == "/usr/local/bin/squeue"
        return _squeue_result(repair, [])

    result = _direct_release(
        repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        _scheduler_contract(),
        {"repair_report_job_id": "444444"},
        "a" * 64,
        census_only,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "report_repair_terminal_cleanup_complete"
    assert all(call[0] == "/usr/local/bin/squeue" for call in calls)
    assert not list(
        (submission / "journal").glob(
            "CALLING_REPORT_REPAIR_0002_RELEASE_*.json"
        )
    )


@pytest.mark.parametrize(
    "staged_name",
    [
        "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json",
        "CALLING_REPORT_REPAIR_0002_SCANCEL_0000_0000.json",
        "REPORT_REPAIR_0002_SCANCEL_RESULT_0000_0000.json",
    ],
)
def legacy_every_virtual_cleanup_stage_class_dominates_positive_release(
    repair, tmp_path, monkeypatch, staged_name
):
    submission = tmp_path / "submission"
    row = _repair_row(repair)
    original_seal = repair.seal_json

    def stop_at_selected_cleanup_stage(path, value):
        if path.name == staged_name:
            _journal_stage(path, value)
            raise RuntimeError("selected cleanup stage")
        return original_seal(path, value)

    first_calls = []

    def first_runner(argv, _cwd, _environment):
        first_calls.append(list(argv))
        if argv[0] == "/usr/local/bin/scancel":
            return _command_result(repair)
        raise AssertionError(argv)

    monkeypatch.setattr(repair, "seal_json", stop_at_selected_cleanup_stage)
    with pytest.raises(RuntimeError, match="selected cleanup stage"):
        _direct_cleanup(
            repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            _scheduler_contract(),
            _census(repair, [row]),
            "fixture_cleanup",
            first_runner,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    stage = submission / "journal" / f".{staged_name}.seal.tmp"
    target = submission / "journal" / staged_name
    assert stage.is_file() and not target.exists()
    assert stat.S_IMODE(stage.lstat().st_mode) == 0o444

    monkeypatch.setattr(repair, "seal_json", original_seal)
    replay_calls = []

    def empty_census(argv, _cwd, _environment):
        replay_calls.append(list(argv))
        assert argv[0] == "/usr/local/bin/squeue"
        return _squeue_result(repair, [])

    terminal = _direct_release(
        repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        _scheduler_contract(),
        {"repair_report_job_id": "444444"},
        "a" * 64,
        empty_census,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"
    assert not stage.exists() and target.is_file()
    assert all(call[0] == "/usr/local/bin/squeue" for call in replay_calls)
    assert not list(
        (submission / "journal").glob(
            "CALLING_REPORT_REPAIR_0002_RELEASE_*.json"
        )
    )


@pytest.mark.parametrize("transition", ["release", "scancel"])
def test_scheduler_mutation_rebinds_source_after_calling_seal(
    repair, tmp_path, monkeypatch, transition
):
    submission = tmp_path / "submission"
    original_create = repair._RepairTransitionBinding.create_direct_final_file
    mutated = False

    def mutate_source_after_calling(binding, path, payload, *, label):
        nonlocal mutated
        result = original_create(binding, path, payload, label=label)
        is_boundary = (
            transition == "release"
            and path.name == "CALLING_REPORT_REPAIR_0002_RELEASE_0000.json"
        ) or (
            transition == "scancel"
            and path.name
            == "CALLING_REPORT_REPAIR_0002_SCANCEL_0000_0000.json"
        )
        if is_boundary:
            source_archive = submission / repair.SOURCE_ARCHIVE_NAME
            source_archive.chmod(0o600)
            source_archive.write_bytes(source_archive.read_bytes() + b"drift")
            source_archive.chmod(0o444)
            mutated = True
        return result

    monkeypatch.setattr(
        repair._RepairTransitionBinding,
        "create_direct_final_file",
        mutate_source_after_calling,
    )
    calls = []

    def runner(argv, _cwd, _environment):
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            return _squeue_result(repair, [_repair_row(repair)])
        raise AssertionError(f"scheduler mutation escaped post-CALLING rebind: {argv}")

    with pytest.raises(
        repair.RepairError,
        match=(
            "transition (?:predecessor changed.*SOURCE_ARCHIVE|phase changed)"
            "|retained.*source.*changed"
        ),
    ):
        if transition == "release":
            _direct_release(
                repair,
                submission,
                repair.EXPECTED_SUBMISSION_SHA256,
                _scheduler_contract(),
                {"repair_report_job_id": "444444"},
                "a" * 64,
                _with_job_control(repair, submission, runner),
                _FakeLocks(),
                sleep=lambda _seconds: None,
            )
        else:
            _direct_cleanup(
                repair,
                submission,
                repair.EXPECTED_SUBMISSION_SHA256,
                _scheduler_contract(),
                _census(repair, [_repair_row(repair)]),
                "fixture_cleanup",
                runner,
                _FakeLocks(),
                sleep=lambda _seconds: None,
            )
    assert mutated
    assert all(
        not (
            call[:2] == ["/usr/local/bin/scontrol", "release"]
            or call[0] == "/usr/local/bin/scancel"
        )
        for call in calls
    )


def _rewrite_sealed_fixture_json(path, mutate):
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.chmod(0o600)
    path.write_bytes(_sealed_json_payload(value))
    path.chmod(0o444)


@pytest.mark.parametrize("timing", ["after_calling_seal", "during_sbatch"])
def test_submit_transition_retains_calling_through_scheduler_mutation(
    repair, tmp_path, monkeypatch, timing
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )
    calling_path = (
        submission / "CALLING_REPORT_REPAIR_0002_SUBMIT.json"
    )
    original_create = repair._RepairTransitionBinding.create_direct_final_file
    calls = []
    tampered = False

    def tamper_calling():
        nonlocal tampered
        _rewrite_sealed_fixture_json(
            calling_path,
            lambda value: value.__setitem__(
                "called_at_utc", value["called_at_utc"] + "-tampered"
            ),
        )
        tampered = True

    def create_then_maybe_tamper(binding, path, payload, *, label):
        result = original_create(binding, path, payload, label=label)
        if timing == "after_calling_seal" and path == calling_path:
            tamper_calling()
        return result

    monkeypatch.setattr(
        repair._RepairTransitionBinding,
        "create_direct_final_file",
        create_then_maybe_tamper,
    )
    squeue_count = 0

    def runner(argv, _cwd, _environment):
        nonlocal squeue_count
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            squeue_count += 1
            return _squeue_result(repair, [])
        if argv[0] == "/usr/local/bin/sbatch":
            assert timing == "during_sbatch"
            tamper_calling()
            return _command_result(repair, b"444444\n")
        raise AssertionError(argv)

    with pytest.raises(
        repair.RepairError,
        match=(
            "(?:transition.*(?:changed|differs)|retained file changed"
            "|retained repair artifact identity differs)"
        ),
    ):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=runner,
            sleep=lambda _seconds: None,
        )
    assert tampered
    assert sum(call[0] == "/usr/local/bin/sbatch" for call in calls) == (
        1 if timing == "during_sbatch" else 0
    )
    assert not (submission / "REPORT_REPAIR_0002_SUBMITTED.json").exists()


@pytest.mark.parametrize(
    ("artifact_name", "mutation_round"),
    [
        ("REPORT_REPAIR_0002_SUBMITTED.json", 4),
        ("REPORT_REPAIR_0002_AUTHORIZED.json", 7),
    ],
)
def test_positive_transition_retains_exact_predecessor_across_scheduler_census(
    repair, tmp_path, monkeypatch, artifact_name, mutation_round
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )
    artifact = submission / artifact_name
    calls = []
    squeue_count = 0
    tampered = False

    def runner(argv, _cwd, _environment):
        nonlocal squeue_count, tampered
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/sbatch":
            return _command_result(repair, b"444444\n")
        if argv[0] == "/usr/local/bin/squeue":
            squeue_count += 1
            if squeue_count == mutation_round:
                assert artifact.is_file()
                _rewrite_sealed_fixture_json(
                    artifact,
                    lambda value: value.__setitem__(
                        next(
                            key
                            for key in (
                                "accepted_at_utc",
                                "authorized_at_utc",
                            )
                            if key in value
                        ),
                        "2026-08-29T23:59:59Z",
                    ),
                )
                tampered = True
            if squeue_count <= 3:
                return _squeue_result(repair, [])
            return _squeue_result(repair, [_repair_row(repair)])
        raise AssertionError(
            f"scheduler mutation escaped predecessor transaction: {argv}"
        )

    with pytest.raises(
        repair.RepairError,
        match=(
            "(?:transition.*changed|authorization.*changed|retained file changed"
            "|retained repair artifact identity differs)"
        ),
    ):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=runner,
            sleep=lambda _seconds: None,
        )
    assert tampered
    assert squeue_count == mutation_round
    assert all(call[0] != "/usr/local/bin/scontrol" for call in calls)


def test_cleanup_transition_rebinds_authority_immediately_before_scancel(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    original_rebind = repair._rebind_scheduler_calling_before_mutation
    tampered = False

    def rebind_then_tamper(*args, **kwargs):
        nonlocal tampered
        value = original_rebind(*args, **kwargs)
        calling_path = kwargs["calling_path"]
        if not tampered and "SCANCEL" in calling_path.name:
            authority = repair._cancel_authority_path(submission, 0)
            _rewrite_sealed_fixture_json(
                authority,
                lambda item: item.__setitem__(
                    "authorized_at_utc", "2026-08-29T23:59:59Z"
                ),
            )
            tampered = True
        return value

    monkeypatch.setattr(
        repair,
        "_rebind_scheduler_calling_before_mutation",
        rebind_then_tamper,
    )
    calls = []

    def forbidden_scheduler(argv, _cwd, _environment):
        calls.append(list(argv))
        raise AssertionError(f"scancel escaped authority rebind: {argv}")

    with pytest.raises(
        repair.RepairError,
        match=(
            "cleanup authorization|transition predecessor changed"
            "|retained file changed|retained repair artifact identity differs"
        ),
    ):
        _direct_cleanup(
            repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            _scheduler_contract(),
            _census(repair, [_repair_row(repair)]),
            "fixture_cleanup",
            forbidden_scheduler,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    assert tampered
    assert not calls


def test_controller_completed_recovery_authenticates_report_before_phase(
    repair, tmp_path, monkeypatch
):
    reached = []

    def reject_unauthenticated_archive(*_args, **_kwargs):
        reached.append("publication-archive-authority")
        raise repair.RepairError("fixture publication archive authority differs")

    def forbidden_phase(*_args, **_kwargs):
        raise AssertionError("COMPLETED recovery classified phase before report auth")

    monkeypatch.setattr(
        repair,
        "_read_publication_archive_fd",
        reject_unauthenticated_archive,
    )
    monkeypatch.setattr(repair, "_classify_repair_phase", forbidden_phase)
    transition = SimpleNamespace(
        retained_publication_archive=lambda: (
            tmp_path
            / f"REPORT_REPAIR_0002_PUBLICATION.{'a' * 64}.archive",
            -1,
            "a" * 64,
            1,
        )
    )
    with pytest.raises(repair.RepairError, match="archive authority"):
        repair._seal_repair_completed(
            tmp_path / "submission",
            "a" * 64,
            {},
            "b" * 64,
            {},
            object(),
            transition=transition,
        )
    assert reached == ["publication-archive-authority"]


def legacy_virtual_phase_rejects_malformed_linked_completed_without_unlink(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    _seed_direct_authorized_phase(repair, submission)
    journal = submission / "journal"
    for name in (
        "CALLING_REPORT_REPAIR_0002_RELEASE_0000.json",
        "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json",
        "REPORT_REPAIR_0002_RELEASED.json",
    ):
        path = journal / name
        path.write_bytes(b"fixture\n")
        path.chmod(0o444)
    report_root = submission / "report"
    report_root.mkdir(mode=0o700)
    report_root.chmod(0o555)
    target = submission / "REPORT_REPAIR_0002_COMPLETED.json"
    stage, payload = _journal_stage(
        target,
        {"schema_version": 1, "status": "malformed_completed"},
        link_target=True,
    )
    with pytest.raises(repair.RepairError, match="staged artifact payload"):
        repair._classify_repair_phase(
            submission, source_must_be_installed=True
        )
    assert target.read_bytes() == stage.read_bytes() == payload
    assert target.lstat().st_nlink == stage.lstat().st_nlink == 2


def legacy_wrong_sealed_release_denial_stage_is_rejected_before_scheduler(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    _seed_direct_authorized_phase(repair, submission)
    contract = _scheduler_contract()
    census = _census(repair, [])
    value = {
        "schema_version": 1,
        "status": "report_repair_terminal_release_denied",
        "campaign_id": repair.CAMPAIGN_ID,
        "submission_sha256": repair.EXPECTED_SUBMISSION_SHA256,
        "attempt": repair.ATTEMPT,
        "repair_report_job_id": "555555",
        "authorization_sha256": "a" * 64,
        "reason": "authorized_repair_job_absent_before_release",
        "pre_release_census": census,
        "pre_release_census_sha256": repair.stable_hash(census),
        "publication_allowed": False,
        "retry_allowed": False,
        "sealed_at_utc": "2026-08-29T00:00:00Z",
    }
    target = submission / "REPORT_REPAIR_0002_TERMINAL_RELEASE_DENIED.json"
    stage, payload = _journal_stage(target, value)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("wrong sealed phase-next stage reached scheduler")

    with pytest.raises(repair.RepairError, match="release-denied terminal differs"):
        _direct_release(
            repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            {"repair_report_job_id": "444444"},
            "a" * 64,
            forbidden,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    assert stage.read_bytes() == payload
    assert not target.exists()


def legacy_cleanup_terminal_sealed_stage_replays_without_duplicate_scancel(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True, mode=0o700)
    contract = _scheduler_contract()
    row = _repair_row(repair)
    original_seal = repair.seal_json
    terminal_stage = None

    def crash_after_terminal_stage(path, value):
        nonlocal terminal_stage
        if path.name == "REPORT_REPAIR_0002_CANCEL_TERMINAL_0000.json":
            terminal_stage, _payload = _journal_stage(path, value)
            raise RuntimeError("cleanup-terminal staged killpoint")
        return original_seal(path, value)

    calls = []

    def first(argv, _cwd, _environment):
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/scancel":
            return _command_result(repair)
        if argv[0] == "/usr/local/bin/squeue":
            return _squeue_result(repair, [])
        raise AssertionError(argv)

    monkeypatch.setattr(repair, "seal_json", crash_after_terminal_stage)
    with pytest.raises(RuntimeError, match="cleanup-terminal staged"):
        _direct_cleanup(
            repair,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            _census(repair, [row]),
            "ambiguous_test_identity",
            first,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    assert terminal_stage is not None and terminal_stage.exists()
    assert sum(call[0] == "/usr/local/bin/scancel" for call in calls) == 1

    monkeypatch.setattr(repair, "seal_json", original_seal)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("terminal stage replay repeated scheduler mutation")

    terminal = _direct_cleanup(
        repair,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        _census(repair, []),
        "ignored_replay_reason",
        forbidden,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"
    assert not terminal_stage.exists()
    assert (
        submission / "REPORT_REPAIR_0002_CANCEL_TERMINAL_0000.json"
    ).is_file()


@pytest.mark.parametrize("sibling", [".report.tmp.1.1", ".report.install-probe-source"])
def test_published_report_rejects_every_reserved_sibling_in_controller_and_worker(
    repair, report, tmp_path, sibling
):
    submission = _make_exact_repair_attempt_namespace(repair, tmp_path)
    residue = submission / sibling
    residue.mkdir(mode=0o700)
    with pytest.raises(
        repair.RepairError,
        match=(
            "publication sibling|staging/probe residue"
            "|install-probe/local staging residue"
        ),
    ):
        repair._require_repair_filesystem_namespace(
            submission,
            source_must_be_installed=True,
            durable_journal_names=(
                "REPORT_REPAIR_0002_SUBMITTED.json",
                "REPORT_REPAIR_0002_RELEASED.json",
            ),
        )
    with pytest.raises(
        report.ReportError,
        match=(
            "publication sibling|staging/probe residue|publication residue"
            "|publication namespace differs"
        ),
    ):
        report._validated_repair_publication_namespace(submission)
    assert residue.is_dir()


@pytest.mark.parametrize("kind", ["file", "symlink", "fifo", "nonempty", "mode"])
def legacy_report_install_probe_hostile_type_is_rejected_untouched(
    repair, tmp_path, kind
):
    submission = tmp_path / "submission"
    submission.mkdir(mode=0o700)
    probe = submission / ".report.install-probe-source"
    if kind == "file":
        probe.write_bytes(b"preserve")
    elif kind == "symlink":
        probe.symlink_to(tmp_path / "missing")
    elif kind == "fifo":
        os.mkfifo(probe, 0o600)
    else:
        probe.mkdir(mode=0o700)
        if kind == "nonempty":
            (probe / "preserve").write_bytes(b"preserve")
        else:
            probe.chmod(0o755)
    with pytest.raises(repair.RepairError, match="probe residue identity"):
        repair._require_report_install_probe_namespace(
            submission, allow_exact_crash_residue=True
        )
    assert os.path.lexists(probe)


def legacy_source_staging_named_inode_swap_rejects_without_touching_replacement(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    submission.mkdir(mode=0o700)
    source = _repair_source(repair)
    staging = _empty_source_staging(repair, submission)
    partial = staging / repair.SOURCE_NAMES[0]
    partial.write_bytes(b"partial")
    partial.chmod(0o600)
    displaced = staging.with_name(staging.name + ".displaced")
    real_fchmod = os.fchmod
    swapped = False

    def swap_before_first_directory_mutation(descriptor, mode):
        nonlocal swapped
        opened = os.fstat(descriptor)
        if not swapped and stat.S_ISDIR(opened.st_mode):
            swapped = True
            staging.rename(displaced)
            staging.mkdir(mode=0o700)
            (staging / "replacement").write_bytes(b"preserve")
        return real_fchmod(descriptor, mode)

    monkeypatch.setattr(repair.os, "fchmod", swap_before_first_directory_mutation)
    with pytest.raises(repair.RepairError, match="binding changed"):
        repair._remove_repair_source_staging(
            staging, source, staging.parent, _FakeLocks()
        )
    assert (staging / "replacement").read_bytes() == b"preserve"
    assert (displaced / repair.SOURCE_NAMES[0]).read_bytes() == b"partial"


@pytest.mark.parametrize("binding_name", ["controller", "worker"])
@pytest.mark.parametrize(
    "mutation", ["root_swap", "root_mode", "bundle_bytes", "bundle_swap"]
)
def legacy_completed_report_tree_binding_rejects_root_or_bundle_drift(
    repair, report, tmp_path, binding_name, mutation
):
    submission = tmp_path / "submission"
    submission.mkdir(mode=0o700)
    bundle, decision, provenance = _report_payload_fixture(report)
    report._publish_report_locked(
        submission,
        "a" * 64,
        bundle,
        decision,
        provenance,
    )
    root = submission / "report"
    binding_type = (
        repair._ReportTreeBinding
        if binding_name == "controller"
        else report._CompletedReportTreeBinding
    )
    binding = binding_type(root) if binding_name == "controller" else binding_type(submission)
    try:
        if mutation == "root_swap":
            displaced = submission / "report.displaced"
            root.rename(displaced)
            shutil.copytree(displaced, root)
        elif mutation == "root_mode":
            root.chmod(0o700)
        else:
            commit = json.loads((root / "REPORT_COMMIT.json").read_text())
            bundle_path = root / commit["report_bundle"]
            root.chmod(0o700)
            if mutation == "bundle_swap":
                payload = bundle_path.read_bytes()
                bundle_path.rename(
                    submission / f"{binding_name}.bundle.displaced"
                )
                bundle_path.write_bytes(payload)
                bundle_path.chmod(0o444)
            else:
                bundle_path.chmod(0o600)
                bundle_path.write_bytes(bundle_path.read_bytes() + b" ")
                bundle_path.chmod(0o444)
            root.chmod(0o555)
        with pytest.raises((repair.RepairError, report.ReportError)):
            binding.revalidate()
    finally:
        binding.close()


@pytest.mark.parametrize("corrupt_bundle_before_binding", [False, True])
def legacy_completed_worker_binding_authenticates_exact_quartet_before_completion(
    report, tmp_path, monkeypatch, corrupt_bundle_before_binding
):
    submission = tmp_path / "submission"
    submission.mkdir(mode=0o700)
    bundle = {"schema_version": 1, "cells": []}
    gate_body = {"schema_version": 1, "status": "rejected"}
    decision = {**gate_body, "gate_sha256": report.stable_hash(gate_body)}
    prerequisite = {"schema_version": 1, "status": "fixture"}
    provenance_v1 = {
        "schema_version": 1,
        "campaign_id": report.CAMPAIGN_ID,
        "submission_sha256": "a" * 64,
        "production_authorization_prerequisite": prerequisite,
        "production_authorization_prerequisite_sha256": report.stable_hash(
            prerequisite
        ),
        "outcome_blind_phase": {"status": "fixture"},
        "event_artifacts": [],
        "terminal_artifacts": [],
        "report_bundle_sha256": report.stable_hash(bundle),
        "gate_sha256": decision["gate_sha256"],
    }
    bundle_payload = _sealed_json_payload(bundle)
    decision_payload = _sealed_json_payload(decision)
    provenance_v1_payload = _sealed_json_payload(provenance_v1)
    expected = {
        **report.EXPECTED_REPAIR_REASSEMBLY,
        "status": "rejected",
        "report_bundle_sha256": report.stable_hash(bundle),
        "gate_sha256": decision["gate_sha256"],
        "report_bundle_file_sha256": hashlib.sha256(
            bundle_payload
        ).hexdigest(),
        "gate_decision_file_sha256": hashlib.sha256(
            decision_payload
        ).hexdigest(),
        "original_provenance_v1_file_sha256": hashlib.sha256(
            provenance_v1_payload
        ).hexdigest(),
        "original_provenance_v1_sha256": report.stable_hash(provenance_v1),
        "report_bundle_file_size": len(bundle_payload),
        "gate_decision_file_size": len(decision_payload),
        "original_provenance_v1_file_size": len(provenance_v1_payload),
    }
    monkeypatch.setattr(report, "EXPECTED_REPAIR_REASSEMBLY", expected)
    publication_authority = {
        "schema_version": 2,
        "status": "authorized_terminal_report_repair",
        "attempt": 2,
    }
    provenance = {
        **provenance_v1,
        "schema_version": 2,
        "publication_authority": publication_authority,
    }
    commit = report._publish_report_locked(
        submission, "a" * 64, bundle, decision, provenance
    )
    if corrupt_bundle_before_binding:
        bundle_path = submission / "report" / commit["report_bundle"]
        (submission / "report").chmod(0o700)
        bundle_path.chmod(0o600)
        bundle_path.write_bytes(bundle_path.read_bytes() + b" ")
        bundle_path.chmod(0o444)
        (submission / "report").chmod(0o555)
    binding = report._CompletedReportTreeBinding(submission)
    try:
        if corrupt_bundle_before_binding:
            with pytest.raises(report.ReportError):
                binding.authenticate(
                    submission_sha256="a" * 64,
                    commit=commit,
                    publication_authority=publication_authority,
                )
        else:
            assert binding.authenticate(
                submission_sha256="a" * 64,
                commit=commit,
                publication_authority=publication_authority,
            ) == commit
    finally:
        binding.close()


@pytest.mark.parametrize(
    ("recovery_state", "drift_lock_during_completed"),
    [("durable", False), ("linked", False), ("linked", True)],
)
def legacy_real_worker_completed_recovery_consumes_authenticated_report(
    report,
    tmp_path,
    monkeypatch,
    recovery_state,
    drift_lock_during_completed,
):
    submission = tmp_path / "submission"
    submission.mkdir(mode=0o700)
    (submission / "journal").mkdir(mode=0o700)
    bundle = {"schema_version": 1, "cells": []}
    gate_body = {"schema_version": 1, "status": "rejected"}
    decision = {**gate_body, "gate_sha256": report.stable_hash(gate_body)}
    prerequisite = {"schema_version": 1, "status": "fixture"}
    provenance_v1 = {
        "schema_version": 1,
        "campaign_id": report.CAMPAIGN_ID,
        "submission_sha256": "a" * 64,
        "production_authorization_prerequisite": prerequisite,
        "production_authorization_prerequisite_sha256": report.stable_hash(
            prerequisite
        ),
        "outcome_blind_phase": {"status": "fixture"},
        "event_artifacts": [],
        "terminal_artifacts": [],
        "report_bundle_sha256": report.stable_hash(bundle),
        "gate_sha256": decision["gate_sha256"],
    }
    bundle_payload = _sealed_json_payload(bundle)
    decision_payload = _sealed_json_payload(decision)
    provenance_v1_payload = _sealed_json_payload(provenance_v1)
    expected = {
        **report.EXPECTED_REPAIR_REASSEMBLY,
        "status": "rejected",
        "report_bundle_sha256": report.stable_hash(bundle),
        "gate_sha256": decision["gate_sha256"],
        "report_bundle_file_sha256": hashlib.sha256(
            bundle_payload
        ).hexdigest(),
        "gate_decision_file_sha256": hashlib.sha256(
            decision_payload
        ).hexdigest(),
        "original_provenance_v1_file_sha256": hashlib.sha256(
            provenance_v1_payload
        ).hexdigest(),
        "original_provenance_v1_sha256": report.stable_hash(provenance_v1),
        "report_bundle_file_size": len(bundle_payload),
        "gate_decision_file_size": len(decision_payload),
        "original_provenance_v1_file_size": len(provenance_v1_payload),
    }
    monkeypatch.setattr(report, "EXPECTED_REPAIR_REASSEMBLY", expected)
    authorization = _fixture_repair_authority(report, submission)
    authorization_sha256 = "9" * 64
    publication_authority = report._repair_publication_authority(
        authorization, authorization_sha256, 2
    )
    provenance = {
        **provenance_v1,
        "schema_version": 2,
        "publication_authority": publication_authority,
    }
    commit = report._publish_report_locked(
        submission, "a" * 64, bundle, decision, provenance
    )
    commit_payload = (submission / "report/REPORT_COMMIT.json").read_bytes()
    completed = {
        "schema_version": 1,
        "status": "report_repair_terminal_publication_complete",
        "campaign_id": report.CAMPAIGN_ID,
        "submission_sha256": "a" * 64,
        "attempt": 2,
        "repair_report_job_id": authorization["repair_report_job_id"],
        "predecessor_failure_evidence": authorization[
            "predecessor_failure_evidence"
        ],
        "predecessor_failure_evidence_sha256": authorization[
            "predecessor_failure_evidence_sha256"
        ],
        "authorization": "REPORT_REPAIR_0002_AUTHORIZED.json",
        "authorization_sha256": authorization_sha256,
        "release": "REPORT_REPAIR_0002_RELEASED.json",
        "release_sha256": authorization["_validated_release_sha256"],
        "report_commit": "report/REPORT_COMMIT.json",
        "report_commit_sha256": hashlib.sha256(commit_payload).hexdigest(),
        "report_commit_value": commit,
        "report_commit_value_sha256": report.stable_hash(commit),
        "publication_authority_sha256": report.stable_hash(
            publication_authority
        ),
        "repair_source_installation_method": authorization[
            "repair_source_installation_method"
        ],
        "report_publication_installation_method": authorization[
            "report_publication_installation_method"
        ],
        "expected_reassembly": authorization["expected_reassembly"],
        "publication_complete": True,
        "retry_allowed": False,
        "successor_attempt_allowed": False,
        "completed_at_utc": "2026-08-29T15:00:00Z",
    }
    target = submission / "REPORT_REPAIR_0002_COMPLETED.json"
    if recovery_state == "durable":
        report.seal_json(target, completed)
    else:
        stage = target.parent / f".{target.name}.seal.tmp"
        stage.write_bytes(_sealed_json_payload(completed))
        stage.chmod(0o444)
        os.link(stage, target)
    phase_binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True
    )
    try:
        report._validate_completed_recovery_phase(
            submission,
            "a" * 64,
            phase_binding,
            authorization,
            authorization_sha256,
            2,
        )
    finally:
        phase_binding.close()
    class CompletionLocks(_FakeLocks):
        drifted = False

        def bindings(self):
            transaction, report_cancel = super().bindings()
            if self.drifted:
                transaction = {**transaction, "inode": transaction["inode"] + 1}
            return transaction, report_cancel

    locks = CompletionLocks()
    if drift_lock_during_completed:
        original_seal = report.seal_json

        def seal_then_drift(path, value):
            digest = original_seal(path, value)
            if path.name == "REPORT_REPAIR_0002_COMPLETED.json":
                locks.drifted = True
            return digest

        monkeypatch.setattr(report, "seal_json", seal_then_drift)
        with pytest.raises(report.ReportError, match="root/lock binding differs"):
            report._seal_report_repair_completed(
                submission,
                "a" * 64,
                authorization,
                authorization_sha256,
                commit,
                publication_authority,
                locks,
            )
        assert locks.drifted
        return
    result = report._seal_report_repair_completed(
        submission,
        "a" * 64,
        authorization,
        authorization_sha256,
        commit,
        publication_authority,
        locks,
    )
    assert result == completed
    assert target.lstat().st_nlink == 1
    assert not os.path.lexists(target.parent / f".{target.name}.seal.tmp")


@pytest.mark.parametrize(
    "mutation",
    [
        "observation_extra",
        "captured_empty",
        "control_plane",
        "canonical_extra",
        "parsed_missing",
        "argv",
        "environment",
        "returncode",
        "stderr",
        "stdout_row",
    ],
)
def test_attempt1_terminal_sacct_envelope_is_exact_in_controller_and_worker(
    repair, report, tmp_path, monkeypatch, mutation
):
    fixture = _synthetic_attempt1_terminal_and_predecessor(
        repair, tmp_path, monkeypatch
    )
    observation = copy.deepcopy(
        fixture["terminal"]["terminal_scheduler_observation"]
    )
    if mutation == "observation_extra":
        observation["unexpected"] = True
    elif mutation == "captured_empty":
        observation["captured_at_utc"] = ""
    elif mutation == "control_plane":
        observation["scheduler_control_plane"]["unexpected"] = True
    elif mutation == "canonical_extra":
        observation["canonical"]["unexpected"] = True
        observation["canonical_sha256"] = repair.stable_hash(
            observation["canonical"]
        )
    elif mutation == "parsed_missing":
        observation["parsed_row"].pop("Reason")
    elif mutation == "argv":
        observation["raw"]["argv"] = list(reversed(observation["raw"]["argv"]))
    elif mutation == "environment":
        observation["raw"]["environment"]["USER"] = "forged"
    elif mutation == "returncode":
        observation["raw"]["returncode"] = 1
    elif mutation == "stderr":
        observation["raw"]["stderr"] = _stream(b"warning")
    else:
        raw = observation["raw"]
        stdout = _stream(b"forged\n")
        if mutation == "stdout_row":
            raw["stdout"] = stdout
        else:
            raise AssertionError(mutation)
    with pytest.raises(repair.RepairError):
        repair._validated_attempt1_terminal_scheduler_observation(
            observation, fixture["contract"]
        )
    with pytest.raises(report.ReportError):
        report._validated_attempt1_terminal_scheduler_observation(
            observation, fixture["contract"]
        )


@pytest.mark.parametrize("ordering", ["census_after_control", "control_after_auth"])
def test_authorization_rejects_scheduler_observation_timestamp_inversion(
    repair, tmp_path, monkeypatch, ordering
):
    submission = tmp_path / "submission"
    submission.mkdir(mode=0o700)
    contract = {
        "snapshot_root": str(submission / "source-snapshot/repo"),
        "snapshot_inventory_sha256": "3" * 64,
        "package_protocol_sha256": repair.EXPECTED_ORIGINAL_PROTOCOL,
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"},
    }
    receipt = {"report_job_id": repair.EXPECTED_ORIGINAL_REPORT_JOB_ID}
    receipt_map = {"schema_version": 1, "files": {}}
    source = _repair_archive_source(repair, submission)
    census = _census(repair, [_repair_row(repair)])
    census["captured_at_utc"] = "2026-08-29T10:00:02Z"
    job_control = {
        "schema_version": 1,
        "captured_at_utc": "2026-08-29T10:00:03Z",
    }
    authorization = repair._authorization_value(
        submission_root=submission,
        submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
        contract=contract,
        receipt=receipt,
        receipt_map=receipt_map,
        expected_reassembly=repair._expected_reassembly(),
        source=source,
        failure_sha256="a" * 64,
        predecessor_sha256="b" * 64,
        calling_sha256="c" * 64,
        submitted_sha256="d" * 64,
        job_id="444444",
        census=census,
        job_control=job_control,
        report_installation_method=repair.PUBLICATION_ARCHIVE_INSTALL_METHOD,
    )
    authorization["authorized_at_utc"] = "2026-08-29T10:00:04Z"
    if ordering == "census_after_control":
        authorization["scheduler_authority_census"][
            "captured_at_utc"
        ] = "2026-08-29T10:00:04Z"
        authorization["scheduler_authority_census_sha256"] = repair.stable_hash(
            authorization["scheduler_authority_census"]
        )
    else:
        authorization["authorized_at_utc"] = "2026-08-29T10:00:02Z"
    monkeypatch.setattr(
        repair,
        "_validated_job_control_observation",
        lambda value, **_kwargs: dict(value),
    )
    with pytest.raises(repair.RepairError, match="observation order"):
        repair._validate_authorization(
            authorization,
            submission_root=submission,
            submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
            contract=contract,
            receipt=receipt,
            receipt_map=receipt_map,
            expected_reassembly=repair._expected_reassembly(),
            source=source,
            failure_sha256="a" * 64,
            predecessor_sha256="b" * 64,
            calling_sha256="c" * 64,
            submitted_sha256="d" * 64,
        )


def test_cleanup_authorization_rejects_census_after_authorized_timestamp(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    _seed_direct_authorized_phase(repair, submission)
    contract = _scheduler_contract()
    census = _census(repair, [_repair_row(repair)])
    census["captured_at_utc"] = "2026-08-29T10:00:02Z"
    transaction, report_cancel = _FakeLocks().bindings()
    value = {
        "schema_version": 1,
        "status": "report_repair_cleanup_authorized",
        "campaign_id": repair.CAMPAIGN_ID,
        "submission_sha256": repair.EXPECTED_SUBMISSION_SHA256,
        "attempt": repair.ATTEMPT,
        "cancel_generation": 0,
        "reason": "fixture_cleanup",
        "job_ids": ["444444"],
        "pre_cancel_census": census,
        "pre_cancel_census_sha256": repair.stable_hash(census),
        "transaction_lock": transaction,
        "report_cancel_lock": report_cancel,
        "authorized_at_utc": "2026-08-29T10:00:01Z",
    }
    with pytest.raises(repair.RepairError, match="observation order"):
        repair._validated_cancel_authority(
            value,
            submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
            generation=0,
            contract=contract,
            locks=_FakeLocks(),
        )


def test_cleanup_authority_requires_nonempty_authorization_time(repair):
    contract = _scheduler_contract()
    census = _census(repair, [_repair_row(repair)])
    transaction, report_cancel = _FakeLocks().bindings()
    value = {
        "schema_version": 1,
        "status": "report_repair_cleanup_authorized",
        "campaign_id": repair.CAMPAIGN_ID,
        "submission_sha256": repair.EXPECTED_SUBMISSION_SHA256,
        "attempt": repair.ATTEMPT,
        "cancel_generation": 0,
        "reason": "test_cleanup",
        "job_ids": ["444444"],
        "pre_cancel_census": census,
        "pre_cancel_census_sha256": repair.stable_hash(census),
        "transaction_lock": transaction,
        "report_cancel_lock": report_cancel,
        "authorized_at_utc": "",
    }
    with pytest.raises(repair.RepairError, match="cleanup authorization differs"):
        repair._validated_cancel_authority(
            value,
            submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
            generation=0,
            contract=contract,
            locks=_FakeLocks(),
        )


@pytest.mark.parametrize("terminal_mode", ["ambiguous", "release_effect"])
def test_virtual_phase_treats_terminal_release_result_as_no_successor_boundary(
    repair, tmp_path, terminal_mode
):
    submission = tmp_path / "submission"
    _seed_direct_authorized_phase(repair, submission)
    journal = submission / "journal"

    def release_calling(index):
        value = {key: None for key in repair.RELEASE_CALLING_KEYS}
        value.update(
            {
                "schema_version": 1,
                "status": "calling_report_repair_release",
                "campaign_id": repair.CAMPAIGN_ID,
                "submission_sha256": repair.EXPECTED_SUBMISSION_SHA256,
                "attempt": repair.ATTEMPT,
                "release_attempt": index,
            }
        )
        return value

    def release_result(index, mode):
        value = {key: None for key in repair.RELEASE_RESULT_KEYS}
        value.update(
            {
                "schema_version": 1,
                "status": "report_repair_release_attempt_observed",
                "campaign_id": repair.CAMPAIGN_ID,
                "submission_sha256": repair.EXPECTED_SUBMISSION_SHA256,
                "attempt": repair.ATTEMPT,
                "release_attempt": index,
                "mode": mode,
            }
        )
        return value

    repair.seal_json(
        submission / "CALLING_REPORT_REPAIR_0002_RELEASE_0000.json",
        release_calling(0),
    )
    repair.seal_json(
        submission / "REPORT_REPAIR_0002_RELEASE_RESULT_0000.json",
        release_result(
            0,
            (
                "lost_response_reconciled_ambiguous_identity"
                if terminal_mode == "ambiguous"
                else "lost_response_reconciled_release_effect"
            ),
        ),
    )
    if terminal_mode == "ambiguous":
        released = {key: None for key in repair.RELEASED_KEYS}
        released.update(
            {
                "schema_version": 1,
                "status": "report_repair_released",
                "campaign_id": repair.CAMPAIGN_ID,
                "submission_sha256": repair.EXPECTED_SUBMISSION_SHA256,
                "attempt": repair.ATTEMPT,
            }
        )
        repair.seal_json(
            submission / "REPORT_REPAIR_0002_RELEASED.json", released
        )
    else:
        repair.seal_json(
            submission / "CALLING_REPORT_REPAIR_0002_RELEASE_0001.json",
            release_calling(1),
        )
    before = {
        path.name: path.read_bytes()
        for path in journal.iterdir()
        if path.is_file()
    }
    with pytest.raises(repair.RepairError, match="release result terminality"):
        repair._classify_repair_phase(
            submission, source_must_be_installed=True
        )
    assert {
        path.name: path.read_bytes()
        for path in journal.iterdir()
        if path.is_file()
    } == before


def test_full_execute_replays_before_release_worker_terminal_without_scheduler(
    repair, tmp_path, monkeypatch
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )
    real_release = repair._release_authorized_job

    def crash_before_release(*_args, **_kwargs):
        raise RuntimeError("authorized-before-release fixture")

    monkeypatch.setattr(repair, "_release_authorized_job", crash_before_release)
    initial_calls = []

    def initial(argv, _cwd, _environment):
        initial_calls.append(list(argv))
        if argv[0] == "/usr/local/bin/sbatch":
            return _command_result(repair, b"444444\n")
        if argv[0] == "/usr/local/bin/squeue":
            submitted = sum(
                call[0] == "/usr/local/bin/squeue" for call in initial_calls
            ) > 3
            return _squeue_result(
                repair, [_repair_row(repair)] if submitted else []
            )
        raise AssertionError(argv)

    with pytest.raises(RuntimeError, match="authorized-before-release"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=initial,
            sleep=lambda _seconds: None,
        )
    monkeypatch.setattr(repair, "_release_authorized_job", real_release)

    transition_calls = []
    squeue_round = 0

    def transition(argv, _cwd, _environment):
        nonlocal squeue_round
        transition_calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            squeue_round += 1
            return _squeue_result(
                repair, [_repair_row(repair)] if squeue_round <= 3 else []
            )
        if argv[0] == "/usr/local/bin/scontrol":
            return _command_result(repair)
        if argv[0] == "/usr/local/bin/sacct":
            return _repair_sacct_result(
                repair,
                state="FAILED",
                exit_code="2:0",
                end="2026-08-29T10:01:03",
            )
        raise AssertionError(argv)

    terminal = repair.execute_report_repair(
        repo,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        allow_initial_submission=False,
        runner=_with_job_control(repair, submission, transition),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_worker_failure"
    assert terminal["reason"] == "repair_worker_terminal_before_release_evidence"
    assert not (
        submission / "REPORT_REPAIR_0002_RELEASED.json"
    ).exists()
    assert sum(call[0] == "/usr/local/bin/scontrol" for call in transition_calls) == 1

    replay_calls = []

    def replay_runner(argv, _cwd, _environment):
        replay_calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            return _squeue_result(repair, [])
        raise AssertionError("terminal replay reached scheduler mutation")

    replay = repair.execute_report_repair(
        repo,
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        allow_initial_submission=False,
        runner=replay_runner,
        sleep=lambda _seconds: None,
    )
    assert repair.exact_json_equal(replay, terminal)
    assert len(replay_calls) == 3
    assert all(call[0] == "/usr/local/bin/squeue" for call in replay_calls)


@pytest.mark.parametrize(
    "mutation", ["control_plane", "raw_environment", "raw_argv", "projection"]
)
def test_worker_rejects_forged_held_job_control_envelope(
    repair, report, tmp_path, mutation
):
    submission = tmp_path / "submission"
    submission.mkdir(mode=0o700)
    source_root = submission / repair.SOURCE_ARCHIVE_NAME
    contract = _scheduler_contract()
    environment = repair._scheduler_environment("/tmp/slurm.conf")

    def runner(argv, _cwd, _environment):
        assert list(argv[:4]) == [
            "/usr/local/bin/scontrol",
            "show",
            "job",
            "-dd",
        ]
        return _held_job_control_result(
            repair,
            submission,
            str(argv[4]),
            scheduler_command="/proc/self/fd/198",
        )

    observation = repair._job_control_observation(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        "444444",
        source_root,
        runner,
        _FakeLocks(),
        scheduler_command="/proc/self/fd/198",
    )
    if mutation == "control_plane":
        observation["scheduler_control_plane"]["unexpected"] = True
    elif mutation == "raw_environment":
        observation["raw"]["environment"]["USER"] = "forged"
    elif mutation == "raw_argv":
        observation["raw"]["argv"] = list(reversed(observation["raw"]["argv"]))
    else:
        observation["projection"]["fields"]["TimeLimit"] = "03:59:59"
        observation["projection_sha256"] = report.stable_hash(
            observation["projection"]
        )
    authorization = {
        "repair_report_job_id": "444444",
        "repair_job_name": repair._repair_name(
            repair.EXPECTED_SUBMISSION_SHA256
        ),
        "scheduler_comment": repair._repair_comment(
            repair.EXPECTED_SUBMISSION_SHA256
        ),
        "repair_source_root": str(source_root),
    }
    with pytest.raises(report.ReportError):
        report._validated_repair_job_control(
            observation,
            submission_root=submission,
            submission_sha256=repair.EXPECTED_SUBMISSION_SHA256,
            repair_authorization=authorization,
            authority_environment=environment,
            contract=contract,
            scheduler_source_argument="/proc/self/fd/198",
        )


@pytest.mark.parametrize("module_fixture", ["repair", "report"])
def test_direct_final_json_partial_is_permanent_untouched_stop(
    request, module_fixture, tmp_path, monkeypatch
):
    module = request.getfixturevalue(module_fixture)
    parent = tmp_path / module_fixture
    parent.mkdir(mode=0o700)
    target = parent / "REPORT_REPAIR_0002_DIRECT_FIXTURE.json"
    value = {"schema_version": 1, "status": "direct_final_fixture"}
    real_write = module.os.write
    injected = False

    def partial_then_crash(descriptor, payload):
        nonlocal injected
        if not injected:
            injected = True
            real_write(descriptor, bytes(payload[:7]))
            raise OSError(errno.EIO, "direct-final killpoint")
        return real_write(descriptor, payload)

    monkeypatch.setattr(module.os, "write", partial_then_crash)
    with pytest.raises(OSError, match="direct-final killpoint"):
        module.seal_json(target, value)
    monkeypatch.setattr(module.os, "write", real_write)
    before_info = target.lstat()
    before = target.read_bytes()
    assert stat.S_IMODE(before_info.st_mode) == 0o600
    assert before
    destructive = []

    def forbidden(*args, **kwargs):
        destructive.append((args, kwargs))
        raise AssertionError("fail-stop recovery attempted destructive cleanup")

    for name in ("unlink", "rmdir", "rename", "link"):
        monkeypatch.setattr(module.os, name, forbidden)
    error_type = module.RepairError if module_fixture == "repair" else module.ReportError
    with pytest.raises(error_type):
        module.seal_json(target, value)
    after_info = target.lstat()
    assert not destructive
    assert target.read_bytes() == before
    assert (after_info.st_dev, after_info.st_ino, after_info.st_mode) == (
        before_info.st_dev,
        before_info.st_ino,
        before_info.st_mode,
    )


def _direct_source_fixture_root(repair, tmp_path):
    submission = tmp_path / "submission"
    _seed_retained_authority_fixture(repair, submission)
    (submission / "journal").mkdir(mode=0o700, exist_ok=True)
    for name in sorted(
        {
            *repair.EXPECTED_ATTEMPT1_CHAIN_SHA256,
            "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
        }
    ):
        path = submission / "journal" / name
        path.write_bytes(b"fixture\n")
        path.chmod(0o444)
    predecessor = submission / "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
    predecessor.write_bytes(
        _sealed_json_payload(_phase_fixture_value(repair, predecessor.name))
    )
    predecessor.chmod(0o444)
    repair_parent = submission / "report-repair"
    repair_parent.mkdir(mode=0o700, exist_ok=True)
    attempt1 = repair_parent / "attempt-0001"
    attempt1.mkdir(mode=0o700)
    source1 = attempt1 / "source"
    source1.mkdir(mode=0o700)
    source1.chmod(0o555)
    return submission


def test_direct_final_source_snapshot_has_no_cleanup_or_install_rename(
    repair, tmp_path, monkeypatch
):
    submission = _direct_source_fixture_root(repair, tmp_path)
    source = _repair_source(repair)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("direct-final source used a destructive install primitive")

    for name in ("unlink", "rmdir", "rename", "link"):
        monkeypatch.setattr(repair.os, name, forbidden)
    locks = _FakeLocks()
    with repair._retained_transition_scope(
        submission, locks, source_must_be_installed=False
    ) as transition:
        root = repair._seal_repair_source_snapshot(
            submission, source, locks, transition=transition
        )
    sealed = repair._load_sealed_repair_source(root)
    assert sealed["repair_source_installation_method"] == (
        repair.SOURCE_ARCHIVE_INSTALL_METHOD
    )
    assert root == submission / repair.SOURCE_ARCHIVE_NAME
    assert stat.S_IMODE(root.lstat().st_mode) == 0o444
    assert root.lstat().st_nlink == 1


@pytest.mark.parametrize("partial_size", [1, 31])
def test_direct_final_source_archive_partial_is_permanent_untouched_stop(
    repair, tmp_path, monkeypatch, partial_size
):
    submission = _direct_source_fixture_root(repair, tmp_path)
    source = _repair_source(repair)
    real_write = repair.os.write
    injected = False

    def write_then_crash(descriptor, payload):
        nonlocal injected
        if not injected:
            injected = True
            real_write(descriptor, bytes(payload[:partial_size]))
            raise RuntimeError("direct source killpoint")
        return real_write(descriptor, payload)

    monkeypatch.setattr(repair.os, "write", write_then_crash)
    locks = _FakeLocks()
    with pytest.raises(RuntimeError, match="direct source killpoint"):
        with repair._retained_transition_scope(
            submission, locks, source_must_be_installed=False
        ) as transition:
            repair._seal_repair_source_snapshot(
                submission, source, locks, transition=transition
            )
    source_archive = submission / repair.SOURCE_ARCHIVE_NAME
    before_info = source_archive.lstat()
    before = source_archive.read_bytes()
    monkeypatch.setattr(repair.os, "write", real_write)
    destructive = []

    def forbidden(*args, **kwargs):
        destructive.append((args, kwargs))
        raise AssertionError("partial source recovery attempted cleanup")

    for name in ("unlink", "rmdir", "rename", "link"):
        monkeypatch.setattr(repair.os, name, forbidden)
    with pytest.raises(repair.RepairError):
        repair._seal_repair_source_snapshot(submission, source, _FakeLocks())
    after_info = source_archive.lstat()
    assert not destructive and source_archive.read_bytes() == before
    assert stat.S_IMODE(after_info.st_mode) == 0o600
    assert (after_info.st_dev, after_info.st_ino) == (
        before_info.st_dev,
        before_info.st_ino,
    )


def test_direct_final_empty_attempt2_source_prefix_is_permanent_stop(
    repair, tmp_path, monkeypatch
):
    submission = _direct_source_fixture_root(repair, tmp_path)
    attempt2 = submission / "report-repair/attempt-0002"
    attempt2.mkdir(mode=0o700)
    before = attempt2.lstat()
    destructive = []

    def forbidden(*args, **kwargs):
        destructive.append((args, kwargs))
        raise AssertionError("empty final attempt prefix was cleaned")

    for name in ("unlink", "rmdir", "rename", "link"):
        monkeypatch.setattr(repair.os, name, forbidden)
    with pytest.raises(
        repair.RepairError,
        match="(?:incomplete direct-final attempt2|attempt namespace differs)",
    ):
        with repair._retained_transition_scope(
            submission, _FakeLocks(), source_must_be_installed=False
        ):
            raise AssertionError("forged attempt2 directory entered transition")
    after = attempt2.lstat()
    assert not destructive and list(attempt2.iterdir()) == []
    assert (after.st_dev, after.st_ino, after.st_mode) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
    )


def legacy_direct_final_report_quartet_uses_no_cleanup_or_install_rename(
    report, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    submission.mkdir(mode=0o700)
    bundle, decision, provenance = _report_payload_fixture(report)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("direct-final report used a destructive install primitive")

    for name in ("unlink", "rmdir", "rename", "link"):
        monkeypatch.setattr(report.os, name, forbidden)
    commit = report._publish_report_locked(
        submission, "a" * 64, bundle, decision, provenance
    )
    root = submission / "report"
    assert len(commit) == 14
    assert stat.S_IMODE(root.lstat().st_mode) == 0o555
    assert {path.name for path in root.iterdir()} == {
        commit["report_bundle"],
        commit["gate_decision"],
        commit["provenance"],
        "REPORT_COMMIT.json",
    }


@pytest.mark.parametrize("stop_index", [0, 3])
def legacy_direct_final_report_partial_quartet_is_permanent_untouched_stop(
    report, tmp_path, monkeypatch, stop_index
):
    submission = tmp_path / "submission"
    submission.mkdir(mode=0o700)
    bundle, decision, provenance = _report_payload_fixture(report)
    real_seal = report.seal_json
    count = 0

    def seal_then_crash(path, value):
        nonlocal count
        digest = real_seal(path, value)
        current = count
        count += 1
        if current == stop_index:
            raise RuntimeError("direct report killpoint")
        return digest

    monkeypatch.setattr(report, "seal_json", seal_then_crash)
    with pytest.raises(RuntimeError, match="direct report killpoint"):
        report._publish_report_locked(
            submission, "a" * 64, bundle, decision, provenance
        )
    root = submission / "report"
    before_root = root.lstat()
    before = {
        path.name: (
            path.lstat().st_dev,
            path.lstat().st_ino,
            path.lstat().st_mode,
            path.read_bytes(),
        )
        for path in root.iterdir()
    }
    monkeypatch.setattr(report, "seal_json", real_seal)
    destructive = []

    def forbidden(*args, **kwargs):
        destructive.append((args, kwargs))
        raise AssertionError("partial report recovery attempted cleanup")

    for name in ("unlink", "rmdir", "rename", "link"):
        monkeypatch.setattr(report.os, name, forbidden)
    with pytest.raises(report.ReportError):
        report._publish_report_locked(
            submission, "a" * 64, bundle, decision, provenance
        )
    after_root = root.lstat()
    assert not destructive
    assert (after_root.st_dev, after_root.st_ino, after_root.st_mode) == (
        before_root.st_dev,
        before_root.st_ino,
        before_root.st_mode,
    )
    assert {
        path.name: (
            path.lstat().st_dev,
            path.lstat().st_ino,
            path.lstat().st_mode,
            path.read_bytes(),
        )
        for path in root.iterdir()
    } == before


@pytest.mark.parametrize(
    "target_name",
    [
        "CALLING_REPORT_REPAIR_0002_SUBMIT.json",
        "REPORT_REPAIR_0002_AUTHORIZED.json",
        "REPORT_REPAIR_0002_COMPLETED.json",
    ],
)
def test_direct_final_journal_partial_blocks_execute_untouched_before_scheduler(
    repair, tmp_path, monkeypatch, target_name
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )
    target = submission / "journal" / target_name
    target.write_bytes(b"partial-direct-final")
    target.chmod(0o600)
    before = target.lstat()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("partial direct journal reached a mutation")

    monkeypatch.setattr(repair.os, "unlink", forbidden)
    monkeypatch.setattr(repair.os, "rename", forbidden)
    monkeypatch.setattr(repair.os, "rmdir", forbidden)
    with pytest.raises(repair.RepairError):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=False,
            runner=forbidden,
            sleep=lambda _seconds: None,
        )
    after = target.lstat()
    assert not calls and target.read_bytes() == b"partial-direct-final"
    assert (after.st_dev, after.st_ino, after.st_mode) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
    )


def _minimal_retained_worker_publication_tree(report, tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir(mode=0o700)
    journal = submission / "journal"
    journal.mkdir(mode=0o700)
    for name in (
        "SUBMISSION_CONTRACT.json",
        "SUBMISSION_RECEIPT.json",
        "SUBMISSION_AUTHORIZATION.json",
    ):
        path = submission / name
        path.write_bytes(_sealed_json_payload({"schema_version": 1, "name": name}))
        path.chmod(0o444)
    repair_parent = submission / "report-repair"
    repair_parent.mkdir(mode=0o700)
    attempt_root = repair_parent / "attempt-0001"
    attempt_root.mkdir(mode=0o700)
    source_root = attempt_root / "source"
    source_root.mkdir(mode=0o700)
    source_file = source_root / "authority.fixture"
    source_file.write_bytes(b"attempt-1\n")
    source_file.chmod(0o444)
    source_root.chmod(0o555)
    tasks = submission / "tasks"
    tasks.mkdir(mode=0o700)
    for index in range(20):
        cell = tasks / f"cell-{index:02d}"
        cell.mkdir(mode=0o700)
        receipt = cell / "WORKER_COMPLETE.json"
        receipt.write_bytes(
            _sealed_json_payload({"schema_version": 1, "cell": index})
        )
        receipt.chmod(0o444)
        launch_artifact = cell / "LAUNCH.json"
        launch_artifact.write_bytes(
            _sealed_json_payload({"schema_version": 1, "cell": index})
        )
        launch_artifact.chmod(0o444)
        waves = cell / "waves"
        waves.mkdir(mode=0o700)
        for wave_index, names in (
            (0, ("START.json", "CONTINUATION_READY.json")),
            (1, ("START.json", "WORKER_COMPLETE.json")),
        ):
            wave = waves / str(wave_index)
            wave.mkdir(mode=0o700)
            for name in (*names, "WORKER_SIGNAL_READY.json"):
                artifact = wave / name
                artifact.write_bytes(
                    _sealed_json_payload(
                        {
                            "schema_version": 1,
                            "cell": index,
                            "wave": wave_index,
                            "name": name,
                        }
                    )
                )
                artifact.chmod(0o444)
    launches = submission / "launches"
    launches.mkdir(mode=0o700)
    for index in range(20):
        launch = launches / f"cell-{index:02d}.json"
        launch.write_bytes(
            _sealed_json_payload({"schema_version": 1, "cell": index})
        )
        launch.chmod(0o444)
    logs = submission / "logs"
    logs.mkdir(mode=0o700)
    attempt1_log = (
        logs
        / f"report-repair-0001-{report.EXPECTED_ATTEMPT1_JOB_ID}.out"
    )
    attempt1_log.write_bytes(b"repair publication cannot be requeued\n")
    attempt1_log.chmod(0o600)
    _seed_retained_authority_fixture(report, submission)
    archive, _digest, _size = _write_source_archive_fixture(
        submission, b"print('retained worker archive')\n"
    )
    report.__EXP23_REPAIR_SOURCE_ARCHIVE_FD__ = os.open(
        archive, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    return submission


def _retain_minimal_scientific_runs(report, binding, submission):
    roots = submission.parent / "scientific-runs"
    roots.mkdir(mode=0o700)
    for index in range(20):
        run = roots / f"cell-{index:02d}"
        run.mkdir(mode=0o700)
        artifact = run / "scientific.fixture"
        artifact.write_bytes(f"scientific-cell-{index:02d}\n".encode("ascii"))
        artifact.chmod(0o444)
        binding.retain_scientific_run_tree(run)


def test_worker_retained_boundary_spans_direct_report_and_completed_append(
    report, tmp_path
):
    submission = _minimal_retained_worker_publication_tree(report, tmp_path)
    locks = _FakeLocks()
    binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=locks
    )
    try:
        _retain_minimal_scientific_runs(report, binding, submission)
        bundle, decision, provenance = _report_payload_fixture(report)
        commit = report._publish_report_locked(
            submission,
            "a" * 64,
            bundle,
            decision,
            provenance,
            repair_installation_method=report.PUBLICATION_ARCHIVE_INSTALL_METHOD,
            repair_locks=locks,
            repair_phase_binding=binding,
        )
        binding.revalidate()
        completed = {
            "schema_version": 1,
            "status": "report_repair_terminal_publication_complete",
        }
        digest = binding.seal_completed(
            submission / "REPORT_REPAIR_0002_COMPLETED.json",
            completed,
        )
        assert digest == hashlib.sha256(_sealed_json_payload(completed)).hexdigest()
        assert len(commit) == 14
        binding.revalidate()
    finally:
        binding.close()


def test_worker_retained_boundary_rejects_authority_drift_before_completed(
    report, tmp_path
):
    submission = _minimal_retained_worker_publication_tree(report, tmp_path)
    locks = _FakeLocks()
    binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=locks
    )
    try:
        _retain_minimal_scientific_runs(report, binding, submission)
        bundle, decision, provenance = _report_payload_fixture(report)
        report._publish_report_locked(
            submission,
            "a" * 64,
            bundle,
            decision,
            provenance,
            repair_installation_method=report.PUBLICATION_ARCHIVE_INSTALL_METHOD,
            repair_locks=locks,
            repair_phase_binding=binding,
        )
        authority = submission / "SUBMISSION_AUTHORIZATION.json"
        authority.chmod(0o600)
        authority.write_bytes(authority.read_bytes() + b" ")
        authority.chmod(0o444)
        with pytest.raises(report.ReportError, match="retained.*changed"):
            binding.seal_completed(
                submission / "REPORT_REPAIR_0002_COMPLETED.json",
                {"schema_version": 1, "status": "forbidden"},
            )
        assert not (
            submission / "REPORT_REPAIR_0002_COMPLETED.json"
        ).exists()
    finally:
        binding.close()


def test_worker_publication_archive_partial_is_permanent_untouched_stop(
    report, tmp_path, monkeypatch
):
    submission = _minimal_retained_worker_publication_tree(report, tmp_path)
    locks = _FakeLocks()
    binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=locks
    )
    _retain_minimal_scientific_runs(report, binding, submission)
    original_create = binding.create_root_file_from_fd
    real_write = report.os.write

    def partial_archive(path, source_fd, *, size, digest, label):
        injected = False

        def write_then_crash(descriptor, payload):
            nonlocal injected
            if not injected:
                injected = True
                real_write(descriptor, bytes(payload[:37]))
                raise RuntimeError("publication archive direct-final killpoint")
            return real_write(descriptor, payload)

        monkeypatch.setattr(report.os, "write", write_then_crash)
        try:
            return original_create(
                path,
                source_fd,
                size=size,
                digest=digest,
                label=label,
            )
        finally:
            monkeypatch.setattr(report.os, "write", real_write)

    monkeypatch.setattr(binding, "create_root_file_from_fd", partial_archive)
    bundle, decision, provenance = _report_payload_fixture(report)
    with pytest.raises(RuntimeError, match="publication archive direct-final"):
        report._publish_report_locked(
            submission,
            "a" * 64,
            bundle,
            decision,
            provenance,
            repair_installation_method=report.PUBLICATION_ARCHIVE_INSTALL_METHOD,
            repair_locks=locks,
            repair_phase_binding=binding,
        )
    binding.close()
    report.__EXP23_REPAIR_SOURCE_ARCHIVE_FD__ = os.open(
        submission / report.SOURCE_ARCHIVE_NAME,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    archives = list(submission.glob("REPORT_REPAIR_0002_PUBLICATION.*.archive"))
    assert len(archives) == 1
    archive = archives[0]
    before = archive.read_bytes()
    before_info = archive.lstat()
    assert stat.S_IMODE(before_info.st_mode) == 0o600
    destructive = []

    def forbidden(*args, **kwargs):
        destructive.append((args, kwargs))
        raise AssertionError("partial publication archive was cleaned or reused")

    for name in ("unlink", "rename", "link", "rmdir"):
        monkeypatch.setattr(report.os, name, forbidden)
    with pytest.raises(report.ReportError):
        report._RepairPublicationPhaseBinding(
            submission, allow_completed_stage=True, locks=_FakeLocks()
        )
    after_info = archive.lstat()
    assert not destructive and archive.read_bytes() == before
    assert (after_info.st_dev, after_info.st_ino, after_info.st_mode) == (
        before_info.st_dev,
        before_info.st_ino,
        before_info.st_mode,
    )


def test_sealed_publication_archive_without_completed_is_permanent_fail_stop(
    report, tmp_path
):
    submission = _minimal_retained_worker_publication_tree(report, tmp_path)
    locks = _FakeLocks()
    binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=locks
    )
    _retain_minimal_scientific_runs(report, binding, submission)
    bundle, decision, provenance = _report_payload_fixture(report)
    report._publish_report_locked(
        submission,
        "a" * 64,
        bundle,
        decision,
        provenance,
        repair_installation_method=report.PUBLICATION_ARCHIVE_INSTALL_METHOD,
        repair_locks=locks,
        repair_phase_binding=binding,
    )
    binding.close()
    archive = next(
        submission.glob("REPORT_REPAIR_0002_PUBLICATION.*.archive")
    )
    before = (
        archive.lstat().st_dev,
        archive.lstat().st_ino,
        archive.lstat().st_mode,
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    report.__EXP23_REPAIR_SOURCE_ARCHIVE_FD__ = os.open(
        submission / report.SOURCE_ARCHIVE_NAME,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    rebound = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=locks
    )
    try:
        for index in range(20):
            rebound.retain_scientific_run_tree(
                submission.parent / f"scientific-runs/cell-{index:02d}"
            )
        with pytest.raises(
            report.ReportError, match="preexisting publication archive.*fail-stop"
        ):
            report._publish_report_locked(
                submission,
                "a" * 64,
                bundle,
                decision,
                provenance,
                repair_installation_method=(
                    report.PUBLICATION_ARCHIVE_INSTALL_METHOD
                ),
                repair_locks=locks,
                repair_phase_binding=rebound,
            )
    finally:
        rebound.close()
    after = (
        archive.lstat().st_dev,
        archive.lstat().st_ino,
        archive.lstat().st_mode,
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    assert after == before


def test_completed_publication_recovery_rejects_byte_identical_archive_clone(
    report, tmp_path
):
    submission = _minimal_retained_worker_publication_tree(report, tmp_path)
    locks = _FakeLocks()
    binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=locks
    )
    _retain_minimal_scientific_runs(report, binding, submission)
    bundle, decision, provenance = _report_payload_fixture(report)
    commit = report._publish_report_locked(
        submission,
        "a" * 64,
        bundle,
        decision,
        provenance,
        repair_installation_method=report.PUBLICATION_ARCHIVE_INSTALL_METHOD,
        repair_locks=locks,
        repair_phase_binding=binding,
    )
    authorization = {
        "repair_report_job_id": "444444",
        "predecessor_failure_evidence": (
            "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
        ),
        "predecessor_failure_evidence_sha256": "b" * 64,
        "_validated_release_sha256": "c" * 64,
        "repair_source_installation_method": (
            report.SOURCE_ARCHIVE_INSTALL_METHOD
        ),
        "report_publication_installation_method": (
            report.PUBLICATION_ARCHIVE_INSTALL_METHOD
        ),
        "expected_reassembly": copy.deepcopy(report.EXPECTED_REPAIR_REASSEMBLY),
    }
    authorization_sha = "d" * 64
    publication_authority = provenance["publication_authority"]
    completed = report._seal_report_repair_completed(
        submission,
        "a" * 64,
        authorization,
        authorization_sha,
        commit,
        publication_authority,
        locks,
        binding,
    )
    evidence = dict(binding.publication_archive_evidence)
    old_identity = dict(evidence["file_identity"])
    commit_row = next(
        row
        for row in evidence["header"]["entries"]
        if row["kind"] == "report_commit"
    )
    archive = submission / str(evidence["archive"])
    archive_payload = archive.read_bytes()
    displaced = tmp_path / "publication-archive.original"
    binding.close()
    archive.rename(displaced)
    archive.write_bytes(archive_payload)
    archive.chmod(0o444)
    clone_identity = report._direct_final_file_identity(archive.lstat())
    assert clone_identity["inode"] != old_identity["inode"]
    clone_evidence = {
        **evidence,
        "file_identity": clone_identity,
    }
    with pytest.raises(
        report.ReportError, match="completed report repair evidence differs"
    ):
        report._validated_completed_value(
            completed,
            submission_sha256="a" * 64,
            authorization=authorization,
            authorization_sha256=authorization_sha,
            commit=commit,
            commit_sha256=commit_row["sha256"],
            publication_authority=publication_authority,
            publication_archive=clone_evidence,
        )


@pytest.mark.parametrize("mutation", ["same_inode_bytes", "replacement_inode"])
def test_worker_retained_publication_archive_rejects_path_or_byte_swap(
    report, tmp_path, mutation
):
    submission = _minimal_retained_worker_publication_tree(report, tmp_path)
    locks = _FakeLocks()
    binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=locks
    )
    try:
        _retain_minimal_scientific_runs(report, binding, submission)
        bundle, decision, provenance = _report_payload_fixture(report)
        report._publish_report_locked(
            submission,
            "a" * 64,
            bundle,
            decision,
            provenance,
            repair_installation_method=report.PUBLICATION_ARCHIVE_INSTALL_METHOD,
            repair_locks=locks,
            repair_phase_binding=binding,
        )
        archive = next(
            submission.glob("REPORT_REPAIR_0002_PUBLICATION.*.archive")
        )
        if mutation == "same_inode_bytes":
            archive.chmod(0o600)
            archive.write_bytes(archive.read_bytes() + b"drift")
            archive.chmod(0o444)
        else:
            displaced = submission / "publication.archive.displaced"
            payload = archive.read_bytes()
            archive.rename(displaced)
            archive.write_bytes(payload)
            archive.chmod(0o444)
        with pytest.raises(report.ReportError, match="retained.*changed"):
            binding.revalidate()
    finally:
        binding.close()


def test_controller_retained_boundary_includes_attempt1_failure_log(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    _seed_direct_authorized_phase(repair, submission)
    logs = submission / "logs"
    log = logs / f"report-repair-0001-{repair.EXPECTED_ATTEMPT1_JOB_ID}.out"
    log.write_bytes(repair.EXPECTED_ATTEMPT1_LOG_BYTES)
    log.chmod(0o600)
    binding = repair._RepairTransitionBinding(
        submission, _FakeLocks(), source_must_be_installed=True
    )
    try:
        log.write_bytes(log.read_bytes() + b"drift")
        with pytest.raises(
            repair.RepairError, match="transition predecessor changed"
        ):
            binding.revalidate()
    finally:
        binding.close()


def _synthetic_transition_authority_expectation(
    repair, monkeypatch, submission
):
    aggregate = hashlib.sha256()
    for index in range(20):
        relative = Path("tasks") / f"cell-{index:02d}" / "WORKER_COMPLETE.json"
        payload = (submission / relative).read_bytes()
        encoded_path = relative.as_posix().encode("utf-8")
        aggregate.update(len(encoded_path).to_bytes(8, "big"))
        aggregate.update(encoded_path)
        aggregate.update(len(payload).to_bytes(8, "big"))
        aggregate.update(payload)
    monkeypatch.setattr(
        repair,
        "EXPECTED_WORKER_MARKER_AGGREGATE_SHA256",
        aggregate.hexdigest(),
    )
    receipts = repair._worker_receipt_map(submission)
    snapshot_file = submission / "source-snapshot/repo/retained-authority.fixture"
    snapshot_inventory = {
        "retained-authority.fixture": hashlib.sha256(
            snapshot_file.read_bytes()
        ).hexdigest()
    }

    def digest(name):
        return hashlib.sha256((submission / name).read_bytes()).hexdigest()

    return {
        "submission_root": str(submission),
        "submission_contract_sha256": digest("SUBMISSION_CONTRACT.json"),
        "submission_receipt_sha256": digest("SUBMISSION_RECEIPT.json"),
        "submission_authorization_sha256": digest(
            "SUBMISSION_AUTHORIZATION.json"
        ),
        "snapshot_root": str(submission / "source-snapshot/repo"),
        "snapshot_inventory": snapshot_inventory,
        "snapshot_inventory_sha256": repair.stable_hash(snapshot_inventory),
        "worker_receipt_map": receipts,
        "worker_receipt_map_sha256": repair.stable_hash(receipts),
        "attempt1_log_sha256": repair.EXPECTED_ATTEMPT1_LOG_SHA256,
        "attempt1_log_size": repair.EXPECTED_ATTEMPT1_LOG_SIZE,
    }


@pytest.mark.parametrize(
    "target",
    ["contract", "receipt", "snapshot", "attempt1_log"],
)
def test_controller_transition_rejects_preexisting_external_authority_drift(
    repair, monkeypatch, tmp_path, target
):
    submission = tmp_path / "submission"
    _seed_direct_authorized_phase(repair, submission)
    locks = _FakeLocks()
    locks._transition_authority_expectation = (
        _synthetic_transition_authority_expectation(
            repair, monkeypatch, submission
        )
    )
    paths = {
        "contract": submission / "SUBMISSION_CONTRACT.json",
        "receipt": submission / "tasks/cell-00/WORKER_COMPLETE.json",
        "snapshot": (
            submission / "source-snapshot/repo/retained-authority.fixture"
        ),
        "attempt1_log": (
            submission
            / "logs"
            / f"report-repair-0001-{repair.EXPECTED_ATTEMPT1_JOB_ID}.out"
        ),
    }
    path = paths[target]
    original_mode = stat.S_IMODE(path.lstat().st_mode)
    path.chmod(0o600)
    if target in {"contract", "receipt"}:
        value = json.loads(path.read_text(encoding="utf-8"))
        value["coherent_authority_drift"] = True
        path.write_bytes(_sealed_json_payload(value))
    else:
        path.write_bytes(path.read_bytes() + b"authority-drift")
    path.chmod(original_mode)
    with pytest.raises(
        repair.RepairError,
        match="retained .* differs|worker receipt aggregate differs",
    ):
        repair._RepairTransitionBinding(
            submission, locks, source_must_be_installed=True
        )


def test_controller_transition_accepts_exact_external_authority_expectation(
    repair, monkeypatch, tmp_path
):
    submission = tmp_path / "submission"
    _seed_direct_authorized_phase(repair, submission)
    locks = _FakeLocks()
    locks._transition_authority_expectation = (
        _synthetic_transition_authority_expectation(
            repair, monkeypatch, submission
        )
    )
    binding = repair._RepairTransitionBinding(
        submission, locks, source_must_be_installed=True
    )
    try:
        binding.revalidate()
    finally:
        binding.close()


def test_controller_transition_requires_all_twenty_expected_receipts(
    repair, monkeypatch, tmp_path
):
    submission = tmp_path / "submission"
    _seed_direct_authorized_phase(repair, submission)
    locks = _FakeLocks()
    expectation = _synthetic_transition_authority_expectation(
        repair, monkeypatch, submission
    )
    expectation["worker_receipt_map"]["files"].pop(
        "tasks/cell-19/WORKER_COMPLETE.json"
    )
    expectation["worker_receipt_map_sha256"] = repair.stable_hash(
        expectation["worker_receipt_map"]
    )
    locks._transition_authority_expectation = expectation
    with pytest.raises(
        repair.RepairError, match="retained worker receipt expectation differs"
    ):
        repair._RepairTransitionBinding(
            submission, locks, source_must_be_installed=True
        )


@pytest.mark.parametrize("extra_kind", ["regular", "directory"])
def test_controller_retained_snapshot_graph_rejects_coherent_clean_path_swap(
    repair, monkeypatch, tmp_path, extra_kind
):
    submission = tmp_path / "submission"
    _seed_direct_authorized_phase(repair, submission)
    snapshot = submission / "source-snapshot"
    snapshot_root = snapshot / "repo"
    clean = submission / "source-snapshot.clean"
    shutil.copytree(snapshot, clean, copy_function=shutil.copy2)
    snapshot_root.chmod(0o700)
    if extra_kind == "regular":
        extra = snapshot_root / "retained-extra.fixture"
        extra.write_bytes(b"captured only\n")
        extra.chmod(0o444)
    else:
        extra = snapshot_root / "retained-extra-dir"
        extra.mkdir(mode=0o700)
        extra.chmod(0o555)
    snapshot_root.chmod(0o555)
    locks = _FakeLocks()
    locks._transition_authority_expectation = (
        _synthetic_transition_authority_expectation(
            repair, monkeypatch, submission
        )
    )
    original_capture = repair._RepairTransitionBinding._capture
    fresh_rows = []

    def capture_extra_then_show_clean_path(binding):
        original_capture(binding)
        displaced = submission / "source-snapshot.displaced"
        snapshot.rename(displaced)
        clean.rename(snapshot)
        try:
            fresh_rows.append(
                repair._secure_snapshot_inventory_for_transition(
                    snapshot / "repo"
                )
            )
        finally:
            snapshot.rename(clean)
            displaced.rename(snapshot)

    monkeypatch.setattr(
        repair._RepairTransitionBinding,
        "_capture",
        capture_extra_then_show_clean_path,
    )
    with pytest.raises(
        repair.RepairError,
        match="retained source snapshot (?:directory|file) coverage differs",
    ):
        repair._RepairTransitionBinding(
            submission, locks, source_must_be_installed=True
        )
    assert fresh_rows == [
        locks._transition_authority_expectation["snapshot_inventory"]
    ]
    assert extra.exists()


@pytest.mark.parametrize("extra_kind", ["regular", "directory"])
def test_worker_retained_snapshot_graph_rejects_extra_captured_entry(
    report, tmp_path, extra_kind
):
    submission = _minimal_retained_worker_publication_tree(report, tmp_path)
    snapshot_root = submission / "source-snapshot/repo"
    snapshot_root.chmod(0o700)
    if extra_kind == "regular":
        extra = snapshot_root / "retained-extra.fixture"
        extra.write_bytes(b"captured only\n")
        extra.chmod(0o444)
    else:
        extra = snapshot_root / "retained-extra-dir"
        extra.mkdir(mode=0o700)
        extra.chmod(0o555)
    snapshot_root.chmod(0o555)
    binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=_FakeLocks()
    )
    inventory = {
        "retained-authority.fixture": hashlib.sha256(
            (snapshot_root / "retained-authority.fixture").read_bytes()
        ).hexdigest()
    }
    contract = {
        "snapshot_root": str(snapshot_root),
        "snapshot_inventory": inventory,
        "snapshot_inventory_sha256": report.stable_hash(inventory),
    }
    try:
        with pytest.raises(
            report.ReportError,
            match="retained repair snapshot (?:directory|file) coverage differs",
        ):
            binding.validate_exact_snapshot_authority(contract)
    finally:
        binding.close()
    assert extra.exists()


@pytest.mark.parametrize("module_fixture", ["repair", "report"])
@pytest.mark.parametrize("location", ["tasks", "cell"])
def test_retained_task_receipt_tree_requires_exact_captured_names(
    request, module_fixture, tmp_path, location
):
    module = request.getfixturevalue(module_fixture)
    if module_fixture == "repair":
        submission = _direct_source_fixture_root(module, tmp_path)
    else:
        submission = _minimal_retained_worker_publication_tree(module, tmp_path)
    parent = (
        submission / "tasks"
        if location == "tasks"
        else submission / "tasks/cell-00"
    )
    extra = parent / "UNDECLARED_AUTHORITY"
    extra.write_bytes(b"preserve\n")
    extra.chmod(0o444)
    if module_fixture == "repair":
        with pytest.raises(module.RepairError, match="auxiliary authority directory"):
            module._RepairTransitionBinding(
                submission, _FakeLocks(), source_must_be_installed=False
            )
    else:
        with pytest.raises(module.ReportError, match="task.*identity differs"):
            module._RepairPublicationPhaseBinding(
                submission, allow_completed_stage=True, locks=_FakeLocks()
            )
    assert extra.read_bytes() == b"preserve\n"


@pytest.mark.parametrize("module_fixture", ["repair", "report"])
@pytest.mark.parametrize("location", ["tasks", "cell"])
def test_retained_task_receipt_directories_require_exact_mode(
    request, module_fixture, tmp_path, location
):
    module = request.getfixturevalue(module_fixture)
    submission = (
        _direct_source_fixture_root(module, tmp_path)
        if module_fixture == "repair"
        else _minimal_retained_worker_publication_tree(module, tmp_path)
    )
    target = (
        submission / "tasks"
        if location == "tasks"
        else submission / "tasks/cell-00"
    )
    target.chmod(0o755)
    error = module.RepairError if module_fixture == "repair" else module.ReportError
    with pytest.raises(
        error, match="(?:auxiliary authority(?: directory)?|task.*identity) differs"
    ):
        if module_fixture == "repair":
            module._RepairTransitionBinding(
                submission, _FakeLocks(), source_must_be_installed=False
            )
        else:
            module._RepairPublicationPhaseBinding(
                submission, allow_completed_stage=True, locks=_FakeLocks()
            )
    assert stat.S_IMODE(target.lstat().st_mode) == 0o755


@pytest.mark.parametrize("module_fixture", ["repair", "report"])
def test_retained_worker_receipt_rejects_hardlink_identity(
    request, module_fixture, monkeypatch, tmp_path
):
    module = request.getfixturevalue(module_fixture)
    if module_fixture == "repair":
        submission = tmp_path / "submission"
        _seed_direct_authorized_phase(module, submission)
        locks = _FakeLocks()
        locks._transition_authority_expectation = (
            _synthetic_transition_authority_expectation(
                module, monkeypatch, submission
            )
        )
    else:
        submission = _minimal_retained_worker_publication_tree(module, tmp_path)
        locks = _FakeLocks()
    receipt = submission / "tasks/cell-00/WORKER_COMPLETE.json"
    peer = submission / "receipt-hardlink-peer"
    os.link(receipt, peer)
    error = module.RepairError if module_fixture == "repair" else module.ReportError
    with pytest.raises(
        error,
        match="(?:worker receipt|task.*(?:receipt|artifact)).*(?:differs|identity)",
    ):
        if module_fixture == "repair":
            module._RepairTransitionBinding(
                submission, locks, source_must_be_installed=True
            )
        else:
            module._RepairPublicationPhaseBinding(
                submission, allow_completed_stage=True, locks=locks
            )
    assert receipt.lstat().st_nlink == peer.lstat().st_nlink == 2


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "conflicting_terminal", "wrong_mode", "hardlink"],
)
def test_worker_retained_wave_namespace_and_identity_are_exact(
    report, tmp_path, mutation
):
    submission = _minimal_retained_worker_publication_tree(report, tmp_path)
    wave = submission / "tasks/cell-00/waves/0"
    signal = wave / "WORKER_SIGNAL_READY.json"
    if mutation == "missing":
        signal.unlink()
    elif mutation == "extra":
        extra = wave / "UNDECLARED.json"
        extra.write_bytes(b"preserve-extra\n")
        extra.chmod(0o444)
    elif mutation == "conflicting_terminal":
        conflict = wave / "WORKER_COMPLETE.json"
        conflict.write_bytes(b"preserve-conflict\n")
        conflict.chmod(0o444)
    elif mutation == "wrong_mode":
        signal.chmod(0o600)
    else:
        peer = tmp_path / "SIGNAL_HARDLINK.json"
        os.link(signal, peer)
    before = {
        path.name: (
            path.lstat().st_dev,
            path.lstat().st_ino,
            path.lstat().st_mode,
            path.read_bytes(),
        )
        for path in wave.iterdir()
        if path.is_file()
    }
    with pytest.raises(report.ReportError, match="task cell-00 wave0"):
        report._RepairPublicationPhaseBinding(
            submission, allow_completed_stage=True, locks=_FakeLocks()
        )
    after = {
        path.name: (
            path.lstat().st_dev,
            path.lstat().st_ino,
            path.lstat().st_mode,
            path.read_bytes(),
        )
        for path in wave.iterdir()
        if path.is_file()
    }
    assert after == before


def test_worker_launch_path_swap_reads_only_retained_fd_then_fails_rebind(
    report, tmp_path
):
    submission = _minimal_retained_worker_publication_tree(report, tmp_path)
    binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=_FakeLocks()
    )
    worker = SimpleNamespace()
    report._bind_worker_retained_scientific_io(worker, binding)
    launch = submission / "launches/cell-00.json"
    displaced = tmp_path / "launch.displaced"
    evil = tmp_path / "launch.evil"
    evil.write_bytes(b'{"attacker":true}\n')
    evil.chmod(0o444)
    launch.rename(displaced)
    evil.rename(launch)
    try:
        with pytest.raises(report.ReportError, match="retained.*changed"):
            worker._open_absolute_regular(launch, "launch")
        assert launch.read_bytes() == b'{"attacker":true}\n'
    finally:
        launch.rename(evil)
        displaced.rename(launch)
    try:
        with pytest.raises(
            report.ReportError, match="retained repair publication source changed"
        ):
            binding.revalidate()
    finally:
        binding.close()


@pytest.mark.parametrize("scope", ["submission", "scientific"])
def test_managed_scope_postcapture_file_never_falls_back_to_path_open(
    report, tmp_path, scope
):
    submission = _minimal_retained_worker_publication_tree(report, tmp_path)
    binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=_FakeLocks()
    )
    run = submission.parent / "managed-run"
    run.mkdir(mode=0o700)
    binding.retain_scientific_run_tree(run)
    if scope == "submission":
        root = submission
        relative = Path("postcapture.fixture")
    else:
        root = run
        relative = Path("postcapture.fixture")
    injected = root / relative
    injected.write_bytes(b"must-not-open\n")
    injected.chmod(0o444)
    token = report._ACTIVE_REPAIR_PUBLICATION_BINDING.set(binding)
    try:
        with pytest.raises(
            report.ReportError, match="absent from the retained authority graph"
        ):
            report._open_relative_regular(root, relative, "postcapture file")
        assert injected.read_bytes() == b"must-not-open\n"
    finally:
        report._ACTIVE_REPAIR_PUBLICATION_BINDING.reset(token)
        binding.close()


def test_attempt2_active_call_graph_never_enters_legacy_cleanup_paths():
    for filename in ("report.py", "report_repair.py"):
        tree = ast.parse((PACKAGE / filename).read_text(encoding="utf-8"))
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_legacy_")
        ):
            legacy_calls = [
                node.func.id
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id.startswith("_legacy_")
            ]
            assert legacy_calls == [], (filename, function.name, legacy_calls)


def test_attempt2_direct_final_writers_have_no_destructive_pathname_syscalls():
    protected = {
        "report.py": {"seal_json", "_publish_report_locked"},
        "report_repair.py": {
            "seal_json",
            "_write_sealed_file",
            "_seal_repair_source_snapshot",
        },
    }
    forbidden = {"link", "unlink", "rename", "replace", "rmdir"}
    for filename, names in protected.items():
        tree = ast.parse((PACKAGE / filename).read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert names <= functions.keys()
        for name in names:
            calls = [
                node.func.attr
                for node in ast.walk(functions[name])
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr in forbidden
            ]
            assert calls == [], (filename, name, calls)


def test_attempt2_release_and_cancel_prefixes_use_one_captured_namespace():
    tree = ast.parse((PACKAGE / "report_repair.py").read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in ("_release_attempt_prefix", "_cancel_attempt_prefix"):
        function = functions[name]
        fresh_existence_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "lexists"
        ]
        assert fresh_existence_calls == [], name


def test_attempt2_entrypoint_graph_has_no_namespace_destructive_calls():
    cases = {
        "report_repair.py": ({"main", "execute_report_repair"}, set()),
        # Event parsing owns a private scientific scratch file.  It is outside
        # every submission-root repair/publication namespace and is the sole
        # reachable pathname deletion in the worker module.
        "report.py": ({"main", "publish_report"}, {"parse_event_files"}),
    }
    forbidden = {"link", "unlink", "rename", "replace", "rmdir", "mkdir"}
    for filename, (roots, allowed) in cases.items():
        tree = ast.parse((PACKAGE / filename).read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        call_graph = {
            name: {
                call.func.id
                for call in ast.walk(function)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id in functions
            }
            for name, function in functions.items()
        }
        reachable = set()
        pending = list(roots)
        while pending:
            name = pending.pop()
            if name in reachable or name not in functions:
                continue
            reachable.add(name)
            pending.extend(call_graph[name])
        destructive = {
            name
            for name in reachable
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "os"
                and call.func.attr in forbidden
                for call in ast.walk(functions[name])
            )
        }
        assert destructive == allowed, (filename, destructive)


def _install_retained_import_fixture(submission):
    snapshot_root = submission / "source-snapshot/repo"
    snapshot_root.chmod(0o700)
    treewm = snapshot_root / "treewm"
    treewm.mkdir(mode=0o700)
    files = {
        treewm / "__init__.py": b"from . import helper\n",
        treewm / "helper.py": (
            b"import builtins\n"
            b"builtins._exp23_retained_helper = 'retained'\n"
            b"VALUE = 'retained-helper'\n"
        ),
        snapshot_root / "explicit.py": (
            b"import builtins\n"
            b"builtins._exp23_retained_explicit = 'retained'\n"
            b"VALUE = 'retained-explicit'\n"
        ),
        snapshot_root / "entry.py": (
            b"import importlib.util\n"
            b"from treewm import helper\n"
            b"VALUE = helper.VALUE\n"
            b"def explicit(path):\n"
            b"    spec = importlib.util.spec_from_file_location('retained_explicit', path)\n"
            b"    module = importlib.util.module_from_spec(spec)\n"
            b"    spec.loader.exec_module(module)\n"
            b"    return module.VALUE\n"
        ),
    }
    for path, payload in files.items():
        path.write_bytes(payload)
        path.chmod(0o444)
    treewm.chmod(0o555)
    snapshot_root.chmod(0o555)
    return snapshot_root, files


@pytest.mark.parametrize("boundary", ["normal_import", "explicit_spec"])
def test_retained_snapshot_importer_never_executes_transient_path_bytes(
    report, tmp_path, monkeypatch, boundary
):
    submission = _minimal_retained_worker_publication_tree(report, tmp_path)
    snapshot_root, files = _install_retained_import_fixture(submission)
    binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=_FakeLocks()
    )
    evil = tmp_path / "evil.py"
    evil.write_bytes(
        b"import builtins\n"
        b"builtins._exp23_transient_evil_executed = True\n"
        b"VALUE = 'evil'\n"
    )
    target = (
        snapshot_root / "treewm/helper.py"
        if boundary == "normal_import"
        else snapshot_root / "explicit.py"
    )
    real_compile = report.__builtins__["compile"] if isinstance(
        report.__builtins__, dict
    ) else report.__builtins__.compile
    injected = False

    def compile_with_transient_swap(source, filename, mode, *args, **kwargs):
        nonlocal injected
        if not injected and Path(filename).absolute() == target.absolute():
            injected = True
            original = target.read_bytes()
            target.chmod(0o600)
            target.write_bytes(evil.read_bytes())
            try:
                return real_compile(source, filename, mode, *args, **kwargs)
            finally:
                target.write_bytes(original)
                target.chmod(0o444)
        return real_compile(source, filename, mode, *args, **kwargs)

    monkeypatch.delattr(
        __import__("builtins"), "_exp23_transient_evil_executed", raising=False
    )
    monkeypatch.setattr(__import__("builtins"), "compile", compile_with_transient_swap)
    try:
        with pytest.raises(report.ReportError, match="retained.*changed"):
            with report._active_repair_publication_binding(binding):
                with report._retained_snapshot_imports(binding, snapshot_root):
                    entry = report._load_module(
                        "retained-entry", snapshot_root / "entry.py", snapshot_root
                    )
                    if boundary == "explicit_spec":
                        assert entry.explicit(snapshot_root / "explicit.py") == (
                            "retained-explicit"
                        )
        assert injected
        assert not hasattr(
            __import__("builtins"), "_exp23_transient_evil_executed"
        )
        if boundary == "normal_import":
            assert getattr(
                __import__("builtins"), "_exp23_retained_helper"
            ) == "retained"
        else:
            assert getattr(
                __import__("builtins"), "_exp23_retained_explicit"
            ) == "retained"
    finally:
        binding.close()


def test_worker_all_authority_and_scientific_regular_opens_use_retained_fds(
    report, tmp_path
):
    submission = _minimal_retained_worker_publication_tree(report, tmp_path)
    binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=_FakeLocks()
    )
    run_root = submission.parent / "retained-worker-run"
    checkpoint_dir = run_root / "checkpoints"
    checkpoint_dir.mkdir(mode=0o700, parents=True)
    checkpoint = checkpoint_dir / "latest.pt"
    checkpoint.write_bytes(b"retained-checkpoint")
    checkpoint.chmod(0o444)
    binding.retain_scientific_run_tree(run_root)
    worker = SimpleNamespace()
    report._bind_worker_retained_scientific_io(worker, binding)
    try:
        launch = submission / "launches/cell-00.json"
        authority_fd, _ = worker._open_absolute_regular(launch, "launch")
        checkpoint_fd, _ = worker._open_run_regular(
            {"run_directory": run_root}, "checkpoints/latest.pt", "checkpoint"
        )
        try:
            assert os.read(authority_fd, 1 << 20) == launch.read_bytes()
            assert os.read(checkpoint_fd, 1 << 20) == b"retained-checkpoint"
        finally:
            os.close(authority_fd)
            os.close(checkpoint_fd)
        for name in (
            "_open_absolute_regular",
            "_open_run_regular",
            "_open_run_directory",
            "_optional_run_artifact_kind",
            "_verify_snapshot_tree",
            "_verify_snapshot_location",
        ):
            assert getattr(worker, name).__module__ == report.__name__
    finally:
        binding.close()


def test_worker_checkpoint_deserialization_reads_retained_inode_during_path_swap(
    report, tmp_path, monkeypatch
):
    submission = _minimal_retained_worker_publication_tree(report, tmp_path)
    binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=_FakeLocks()
    )
    run_root = submission.parent / "checkpoint-run"
    checkpoint_dir = run_root / "checkpoints"
    checkpoint_dir.mkdir(mode=0o700, parents=True)
    checkpoint = checkpoint_dir / "latest.pt"
    checkpoint.write_bytes(b"retained-pickle-bytes")
    checkpoint.chmod(0o444)
    binding.retain_scientific_run_tree(run_root)
    worker = _load("exp23_retained_checkpoint_worker", "worker.py")
    report._bind_worker_retained_scientific_io(worker, binding)
    observed = []

    def fake_torch_load(handle, *, map_location, weights_only):
        assert map_location == "cpu" and weights_only is False
        displaced = tmp_path / "checkpoint.displaced"
        evil = tmp_path / "checkpoint.evil"
        evil.write_bytes(b"attacker-pickle-bytes")
        evil.chmod(0o444)
        checkpoint.rename(displaced)
        evil.rename(checkpoint)
        try:
            payload = handle.read()
            observed.append(payload)
            return {"payload": payload}
        finally:
            checkpoint.rename(evil)
            displaced.rename(checkpoint)

    fake_torch = SimpleNamespace(load=fake_torch_load)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        worker,
        "validate_checkpoint_payload",
        lambda payload, _context: {"validated_payload": payload["payload"]},
    )
    try:
        with pytest.raises(worker.LifecycleError, match="changed while open"):
            worker.resolve_checkpoint({"run_directory": run_root})
        assert observed == [b"retained-pickle-bytes"]
        with pytest.raises(report.ReportError, match="retained scientific directory changed"):
            binding.revalidate()
    finally:
        binding.close()


def test_publication_rejects_run_inode_replacement_before_archive_creation(
    report, tmp_path
):
    submission = _minimal_retained_worker_publication_tree(report, tmp_path)
    locks = _FakeLocks()
    binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=locks
    )
    _retain_minimal_scientific_runs(report, binding, submission)
    run_file = submission.parent / "scientific-runs/cell-00/scientific.fixture"
    displaced = run_file.with_name("scientific.displaced")
    payload = run_file.read_bytes()
    run_file.rename(displaced)
    run_file.write_bytes(payload)
    run_file.chmod(0o444)
    bundle, decision, provenance = _report_payload_fixture(report)
    try:
        with pytest.raises(
            report.ReportError, match="retained scientific (?:file|directory) changed"
        ):
            report._publish_report_locked(
                submission,
                "a" * 64,
                bundle,
                decision,
                provenance,
                repair_installation_method=report.PUBLICATION_ARCHIVE_INSTALL_METHOD,
                repair_locks=locks,
                repair_phase_binding=binding,
            )
        assert not list(
            submission.glob("REPORT_REPAIR_0002_PUBLICATION.*.archive")
        )
    finally:
        binding.close()


def _install_retained_compatible_contract_fixture(report, tmp_path):
    submission = _minimal_retained_worker_publication_tree(report, tmp_path)
    snapshot = submission / "source-snapshot/repo"
    snapshot.chmod(0o700)
    package = (
        snapshot
        / "experiments"
        / "23-treewm-executable-prefix-repair-pilot-v1"
    )
    package.mkdir(mode=0o755, parents=True)
    treewm = snapshot / "treewm"
    treewm.mkdir(mode=0o755)
    treewm_init = treewm / "__init__.py"
    treewm_init.write_bytes(b"")
    treewm_init.chmod(0o444)
    setting_ids = ["antmaze", "scene", "cube", "puzzle3", "puzzle4"]
    compatible = tmp_path / "compatible-contracts"
    data = compatible / "data"
    recipes = compatible / "future-recipes"
    data.mkdir(mode=0o700, parents=True)
    recipes.mkdir(mode=0o700)
    for index, setting_id in enumerate(setting_ids):
        contract = data / f"{setting_id}.json"
        contract.write_bytes(
            _sealed_json_payload(
                {"schema_version": 1, "setting_id": setting_id, "ordinal": index}
            )
        )
        contract.chmod(0o444)
        recipe = recipes / setting_id
        recipe.mkdir(mode=0o700)
        recipe_manifest = recipe / "manifest.json"
        recipe_manifest.write_bytes(
            _sealed_json_payload(
                {"schema_version": 1, "setting_id": setting_id, "recipe": index}
            )
        )
        recipe_manifest.chmod(0o444)
    manifest = package / "manifest.json"
    manifest.write_bytes(
        _sealed_json_payload(
            {
                "schema_version": 1,
                "paths": {"compatible_contract_root": str(compatible)},
                "settings": [{"id": value} for value in setting_ids],
            }
        )
    )
    manifest.chmod(0o444)
    protocol = package / "protocol.sha256"
    protocol.write_bytes(("a" * 64 + "\n").encode("ascii"))
    protocol.chmod(0o444)
    treewm.chmod(0o555)
    package.chmod(0o555)
    snapshot.chmod(0o555)
    return submission, snapshot, package, compatible, setting_ids


def test_external_compatible_contract_reads_use_retained_fds_and_reject_swap(
    report, tmp_path
):
    submission, snapshot, _package, compatible, setting_ids = (
        _install_retained_compatible_contract_fixture(report, tmp_path)
    )
    binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=_FakeLocks()
    )
    finder = report._RetainedSnapshotFinder(binding, snapshot)
    target = compatible / "data" / f"{setting_ids[0]}.json"
    displaced = tmp_path / "compatible.displaced"
    attacker = tmp_path / "compatible.attacker"
    attacker.write_bytes(b'{"attacker":true}\n')
    attacker.chmod(0o444)
    target.rename(displaced)
    attacker.rename(target)
    try:
        with pytest.raises(report.ReportError, match="retained.*changed"):
            finder.retained_json(target)
        assert target.read_bytes() == b'{"attacker":true}\n'
    finally:
        target.rename(attacker)
        displaced.rename(target)
        binding.close()


def test_campaign_protocol_lock_reader_uses_retained_bytes_not_path_read_text(
    report, tmp_path, monkeypatch
):
    submission, snapshot, package, _compatible, _setting_ids = (
        _install_retained_compatible_contract_fixture(report, tmp_path)
    )
    binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=_FakeLocks()
    )
    finder = report._RetainedSnapshotFinder(binding, snapshot)
    module = ModuleType("retained_campaign_protocol_fixture")
    module.PACKAGE_DIR = package
    module.protocol_sha256 = lambda root: "a" * 64
    module.require = report.require

    def forbidden_read_text(*_args, **_kwargs):
        raise AssertionError("repair protocol lock reopened its pathname")

    monkeypatch.setattr(Path, "read_text", forbidden_read_text)
    try:
        verify = finder.retained_verify_protocol_lock(module)
        assert verify(package) == "a" * 64
        binding.revalidate()
    finally:
        binding.close()


def test_compatible_contract_postcapture_member_never_falls_back_to_path_open(
    report, tmp_path
):
    submission, _snapshot, _package, compatible, _setting_ids = (
        _install_retained_compatible_contract_fixture(report, tmp_path)
    )
    binding = report._RepairPublicationPhaseBinding(
        submission, allow_completed_stage=True, locks=_FakeLocks()
    )
    injected = compatible / "data/postcapture.json"
    injected.write_bytes(b'{"must_not_open":true}\n')
    injected.chmod(0o444)
    token = report._ACTIVE_REPAIR_PUBLICATION_BINDING.set(binding)
    try:
        with pytest.raises(
            report.ReportError, match="absent from the retained authority graph"
        ):
            report._authenticated_regular_bytes(
                injected, "postcapture compatible contract", capture=True
            )
        assert injected.read_bytes() == b'{"must_not_open":true}\n'
    finally:
        report._ACTIVE_REPAIR_PUBLICATION_BINDING.reset(token)
        binding.close()
