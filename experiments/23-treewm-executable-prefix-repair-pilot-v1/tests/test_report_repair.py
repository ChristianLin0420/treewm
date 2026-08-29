from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
from types import SimpleNamespace

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
    commit = report._publish_report_locked(
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
            "1",
            str(tmp_path / "submission/report-repair/attempt-0001/source"),
            str(tmp_path / "submission/report-repair/attempt-0001/source/report.py"),
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


def test_repair_slurm_rejects_absent_restart_identity_before_publication(tmp_path):
    completed = subprocess.run(
        [
            "/bin/bash",
            str(PACKAGE / "report_repair.slurm"),
            str(tmp_path / "snapshot"),
            str(tmp_path / "submission"),
            "a" * 64,
            "1",
            str(tmp_path / "submission/report-repair/attempt-0001/source"),
            str(tmp_path / "submission/report-repair/attempt-0001/source/report.py"),
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
        },
    )
    assert completed.returncode == 2
    assert b"cannot be requeued" in completed.stderr


@pytest.mark.parametrize("expected_hash_matches", [True, False])
def test_repair_slurm_verifies_sealed_source_when_executed_from_spool_copy(
    tmp_path, expected_hash_matches
):
    submission = tmp_path / "submission"
    snapshot = tmp_path / "snapshot"
    source = submission / "report-repair/attempt-0001/source"
    journal = submission / "journal"
    spool = tmp_path / "slurm-spool"
    source.mkdir(parents=True)
    journal.mkdir()
    snapshot.mkdir()
    spool.mkdir()
    report_program = source / "report.py"
    report_program.write_text("print(__file__)\n", encoding="ascii")
    report_program.chmod(0o444)
    source.chmod(0o555)
    (journal / "REPORT_REPAIR_0001_AUTHORIZED.json").write_text(
        '{"schema_version":1}\n', encoding="ascii"
    )
    copied_script = spool / "slurm_script"
    copied_script.write_bytes((PACKAGE / "report_repair.slurm").read_bytes())
    copied_script.chmod(0o700)

    completed = subprocess.run(
        [
            "/bin/bash",
            str(copied_script),
            str(snapshot),
            str(submission),
            "a" * 64,
            "1",
            str(source),
            str(report_program),
            (
                hashlib.sha256(report_program.read_bytes()).hexdigest()
                if expected_hash_matches
                else "b" * 64
            ),
            str(report_program.stat().st_size),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "SLURM_JOB_ID": "444444",
            "SLURM_RESTART_COUNT": "0",
        },
    )
    if expected_hash_matches:
        assert completed.returncode == 0, completed.stderr.decode()
        assert completed.stdout.decode("ascii").strip() == str(report_program)
    else:
        assert completed.returncode != 0
        assert completed.stdout == b""
        assert b"sealed repair publisher identity/hash differs" in completed.stderr


class _FakeLocks:
    def bindings(self):
        return (
            {"path": "/tmp/transaction", "device": 1, "inode": 2, "uid": os.getuid(), "mode": 0o600},
            {"path": "/tmp/report-cancel", "device": 1, "inode": 3, "uid": os.getuid(), "mode": 0o600},
        )


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
            "attempt": 1,
            "repair_report_job_id": job_id,
            "worker_handoff": dict(report.REPAIR_WORKER_HANDOFF),
            "publication_allowed": True,
            "scheduler_submission_allowed": False,
        }
    )
    path = submission / "journal/REPORT_REPAIR_0001_AUTHORIZED.json"
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


def test_repair_worker_waits_beyond_sixty_seconds_for_authenticated_release(
    report, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    (submission / "journal").mkdir(parents=True)
    _authorization, digest = _wait_authorization(report, submission)
    monkeypatch.setenv("SLURM_JOB_ID", "444444")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    clock = {"value": 0.0}

    def monotonic():
        return clock["value"]

    def sleep(seconds):
        clock["value"] += float(seconds)
        if clock["value"] >= 61.0 and not os.path.lexists(
            submission / "journal/REPORT_REPAIR_0001_RELEASED.json"
        ):
            report.seal_json(
                submission / "journal/REPORT_REPAIR_0001_RELEASED.json",
                {"schema_version": 1, "status": "fixture_release"},
            )

    report._wait_for_repair_release_evidence(
        submission,
        "a" * 64,
        attempt=1,
        authorization_sha256=digest,
        monotonic=monotonic,
        sleep=sleep,
    )
    assert 61.0 <= clock["value"] < report.REPAIR_RELEASE_EVIDENCE_WAIT_SECONDS


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
            attempt=1,
            authorization_sha256=digest,
            monotonic=lambda: next(times),
            sleep=lambda _seconds: None,
        )
    assert not os.path.lexists(submission / "report")

    (submission / "journal/REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json").write_bytes(
        b"terminal"
    )
    with pytest.raises(report.ReportError, match="cleanup/terminal authority"):
        report._wait_for_repair_release_evidence(
            submission,
            "a" * 64,
            attempt=1,
            authorization_sha256=digest,
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
        repair._cleanup_repair_rows(
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            pre,
            "fixture_ambiguity",
            crash_after_calling,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    assert (journal / "REPORT_REPAIR_0001_CANCEL_AUTHORIZED_0000.json").is_file()
    assert (journal / "CALLING_REPORT_REPAIR_0001_SCANCEL_0000_0000.json").is_file()
    assert not (journal / "REPORT_REPAIR_0001_SCANCEL_RESULT_0000_0000.json").exists()
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
    terminal = repair._cleanup_repair_rows(
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
        (journal / "REPORT_REPAIR_0001_SCANCEL_RESULT_0000_0000.json").read_text()
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
    raw = {
        "argv": argv,
        "environment": {},
        "returncode": 0,
        "stdout": _stream(b""),
        "stderr": _stream(b""),
    }
    forged = {
        "job_id": "123",
        "job_name": f"exp23-launch8-{submission_sha[:16]}-report-repair-0001",
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
            census, submission_sha256=submission_sha, label="fixture"
        )


def test_controller_census_requires_exactly_three_rounds(repair):
    census = _census(repair, [_repair_row(repair)])
    census["rounds"].append({**census["rounds"][-1], "round": 3})
    with pytest.raises(repair.RepairError, match="census envelope"):
        repair._validated_scheduler_census(census)


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
    submission = tmp_path / "submission"
    submission.mkdir()
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


def test_source_snapshot_first_file_crash_is_removed(repair, tmp_path, monkeypatch):
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
        repair._seal_repair_source_snapshot(submission, source)
    attempt = repair._repair_root(submission)
    assert not (attempt / "source").exists()
    assert not list(attempt.glob(".source.tmp.*"))


def test_source_snapshot_noreplace_race_preserves_competing_target(
    repair, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    real_noreplace = repair._rename_directory_noreplace

    def race(staging, target):
        target.mkdir(mode=0o700)
        (target / "competitor").write_text("preserve me", encoding="utf-8")
        return real_noreplace(staging, target)

    monkeypatch.setattr(repair, "_rename_directory_noreplace", race)
    with pytest.raises(FileExistsError):
        repair._seal_repair_source_snapshot(submission, source)
    target = repair._repair_source_root(submission)
    assert (target / "competitor").read_text(encoding="utf-8") == "preserve me"
    assert not list(repair._repair_root(submission).glob(".source.tmp.*"))


def test_source_snapshot_atomically_carries_checkout_authority(repair, tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    source_root = repair._seal_repair_source_snapshot(submission, source)
    assert repair._load_sealed_repair_source(source_root) == source
    assert {path.name for path in source_root.iterdir()} == {
        *repair.SOURCE_NAMES,
        repair.SOURCE_AUTHORITY_NAME,
    }
    authority = source_root / repair.SOURCE_AUTHORITY_NAME
    assert stat.S_IMODE(authority.lstat().st_mode) == 0o444
    assert json.loads(authority.read_text()) == source


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


def _pre_rename_source_staging(repair, submission, source):
    repair_parent = submission / "report-repair"
    attempt_root = repair._repair_root(submission)
    repair_parent.mkdir(mode=0o700)
    attempt_root.mkdir(mode=0o700)
    staging = attempt_root / ".source.tmp.killpoint"
    staging.mkdir(mode=0o700)
    for name in repair.SOURCE_NAMES:
        repair._write_sealed_file(staging / name, (PACKAGE / name).read_bytes())
    authority_payload = (
        json.dumps(source, sort_keys=True, indent=2, allow_nan=False) + "\n"
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
    staging = attempt_root / ".source.tmp.killpoint"
    staging.mkdir(mode=0o700)
    return staging


def test_source_snapshot_recovers_complete_0555_pre_rename_staging(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)

    source_root = repair._seal_repair_source_snapshot(submission, source)

    assert not os.path.lexists(staging)
    assert repair._load_sealed_repair_source(source_root) == source
    assert stat.S_IMODE(source_root.lstat().st_mode) == 0o555


@pytest.mark.parametrize("forgery", ["extra", "missing", "wrong", "authority"])
def test_source_snapshot_rejects_forged_0555_pre_rename_staging(
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
    else:
        target = staging / repair.SOURCE_AUTHORITY_NAME
        target.unlink()
        repair._write_sealed_file(target, b'{"schema_version":1}\n')
    staging.chmod(0o555)

    with pytest.raises(repair.RepairError, match="repair source staging|sealed repair"):
        repair._seal_repair_source_snapshot(submission, source)

    assert os.path.lexists(staging)
    assert stat.S_IMODE(staging.lstat().st_mode) == 0o555
    assert not os.path.lexists(repair._repair_source_root(submission))


@pytest.mark.parametrize("partial_name", ["source", "authority"])
def test_source_snapshot_removes_0600_partial_pre_seal_staging(
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

    source_root = repair._seal_repair_source_snapshot(submission, source)

    assert not os.path.lexists(staging)
    assert repair._load_sealed_repair_source(source_root) == source


@pytest.mark.parametrize(
    "killpoint", ["authority_invalidated", "during_chmod_pass", "after_first_unlink"]
)
def test_source_snapshot_cleanup_two_phase_killpoints_are_restartable(
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

    source_root = repair._seal_repair_source_snapshot(submission, source)

    assert not os.path.lexists(staging)
    assert repair._load_sealed_repair_source(source_root) == source
    assert repair._seal_repair_source_snapshot(submission, source) == source_root


def test_source_snapshot_rejects_missing_0700_coverage_with_completed_authority(
    repair, tmp_path
):
    submission = tmp_path / "submission"
    submission.mkdir()
    source = _repair_source(repair)
    staging = _pre_rename_source_staging(repair, submission, source)
    staging.chmod(0o700)
    (staging / repair.SOURCE_NAMES[0]).unlink()

    with pytest.raises(repair.RepairError, match="authority coverage"):
        repair._seal_repair_source_snapshot(submission, source)

    assert stat.S_IMODE(
        (staging / repair.SOURCE_AUTHORITY_NAME).lstat().st_mode
    ) == 0o444
    assert not os.path.lexists(repair._repair_source_root(submission))


@pytest.mark.parametrize(
    "forgery", ["unknown", "hardlink", "symlink", "special", "mode"]
)
def test_source_snapshot_rejects_unsafe_0700_partial_staging(
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
        repair._seal_repair_source_snapshot(submission, source)

    assert os.path.lexists(staging)
    assert stat.S_IMODE(staging.lstat().st_mode) == 0o700
    assert not os.path.lexists(repair._repair_source_root(submission))


def _configure_minimal_controller(repair, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    submission = tmp_path / "submission"
    repo.mkdir()
    (submission / "journal").mkdir(parents=True)
    (submission / "logs").mkdir()
    contract = {
        "snapshot_root": str(submission / "source-snapshot/repo"),
        "snapshot_inventory_sha256": "3" * 64,
        "package_protocol_sha256": repair.EXPECTED_ORIGINAL_PROTOCOL,
        "git_provenance": _actual_git_provenance(),
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"},
    }
    repair.seal_json(submission / "SUBMISSION_CONTRACT.json", contract)
    receipt = {"report_job_id": repair.EXPECTED_ORIGINAL_REPORT_JOB_ID}
    receipt_map = {"schema_version": 1, "files": {}}
    source = _repair_source(repair)
    failure = {"schema_version": 1, "status": "fixture_failure"}
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
    monkeypatch.setattr(repair, "_verified_live_repair_source", lambda _root: source)
    monkeypatch.setattr(
        repair,
        "_seal_repair_source_snapshot",
        lambda root, _source: repair._repair_source_root(root),
    )
    monkeypatch.setattr(
        repair, "_load_sealed_repair_source", lambda _root: source
    )
    monkeypatch.setattr(repair, "_validate_sealed_repair_source", lambda *_args: None)
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
        (package / name).write_bytes(payload)

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
    repo = tmp_path / "repo"
    submission = tmp_path / "submission"
    repo.mkdir()
    (submission / "journal").mkdir(parents=True)
    (submission / "logs").mkdir()
    contract = {
        "snapshot_root": str(submission / "source-snapshot/repo"),
        "snapshot_inventory_sha256": "3" * 64,
        "package_protocol_sha256": repair.EXPECTED_ORIGINAL_PROTOCOL,
        "git_provenance": _actual_git_provenance(),
        "scheduler_control_plane_contract": {"slurm_conf": "/tmp/slurm.conf"},
    }
    repair.seal_json(submission / "SUBMISSION_CONTRACT.json", contract)
    receipt = {"report_job_id": repair.EXPECTED_ORIGINAL_REPORT_JOB_ID}
    receipt_map = {"schema_version": 1, "files": {}}
    source = _repair_source(repair)
    failure = {"schema_version": 1, "status": "fixture_failure"}

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
    monkeypatch.setattr(repair, "_verified_live_repair_source", lambda _root: source)
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

    calls = []
    squeue_count = 0

    def runner(argv, _cwd, _environment):
        nonlocal squeue_count
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/sbatch":
            journal = submission / "journal"
            assert (journal / "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json").is_file()
            assert (journal / "CALLING_REPORT_REPAIR_0001_SUBMIT.json").is_file()
            assert not (journal / "REPORT_REPAIR_0001_SUBMITTED.json").exists()
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
            assert (journal / "REPORT_REPAIR_0001_AUTHORIZED.json").is_file()
            assert (journal / "CALLING_REPORT_REPAIR_0001_RELEASE_0000.json").is_file()
            assert not (journal / "REPORT_REPAIR_0001_RELEASE_RESULT_0000.json").exists()
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
    assert (journal / "REPORT_REPAIR_0001_SUBMITTED.json").is_file()
    assert (journal / "REPORT_REPAIR_0001_AUTHORIZED.json").is_file()
    assert (journal / "REPORT_REPAIR_0001_RELEASE_RESULT_0000.json").is_file()
    assert (journal / "REPORT_REPAIR_0001_RELEASED.json").is_file()
    assert [call[0] for call in calls] == [
        *(["/usr/local/bin/squeue"] * 3),
        "/usr/local/bin/sbatch",
        *(["/usr/local/bin/squeue"] * 3),
        *(["/usr/local/bin/squeue"] * 3),
        "/usr/local/bin/scontrol",
        *(["/usr/local/bin/squeue"] * 3),
    ]

    # A validated, already-published report wins without another scheduler read
    # or mutation.  The deep prefix validators still replay the sealed submit,
    # authorization, and release chain before returning.
    (submission / "report").mkdir()
    monkeypatch.setattr(
        repair,
        "_validated_repaired_report_tree",
        lambda *_args: {"status": "rejected", "schema_version": 1},
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


def test_failure_seal_before_submit_calling_resumes_with_exactly_one_sbatch(
    repair, tmp_path, monkeypatch
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )

    def seal_source(root, _source_value):
        target = repair._repair_source_root(root)
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(repair, "_seal_repair_source_snapshot", seal_source)
    real_seal = repair.seal_json

    def crash_after_failure(path, value):
        digest = real_seal(path, value)
        if path.name == "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json":
            raise RuntimeError("failure-to-calling killpoint")
        return digest

    monkeypatch.setattr(repair, "seal_json", crash_after_failure)

    def no_scheduler_before_calling(*_args):
        raise AssertionError("scheduler call preceded durable submit calling")

    with pytest.raises(RuntimeError, match="failure-to-calling killpoint"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=no_scheduler_before_calling,
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    assert (journal / "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json").is_file()
    assert not (journal / "CALLING_REPORT_REPAIR_0001_SUBMIT.json").exists()

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
    assert (journal / "CALLING_REPORT_REPAIR_0001_SUBMIT.json").is_file()
    calling = json.loads(
        (journal / "CALLING_REPORT_REPAIR_0001_SUBMIT.json").read_text()
    )
    assert calling["scheduler_pre_submit_census"]["settled_rows"] == []
    assert calling["scheduler_pre_submit_census_sha256"] == repair.stable_hash(
        calling["scheduler_pre_submit_census"]
    )


def test_source_seal_before_failure_recovers_from_its_own_checkout_authority(
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

    def crash_before_failure(*_args, **_kwargs):
        raise RuntimeError("source-to-failure killpoint")

    monkeypatch.setattr(repair, "_build_failure_evidence", crash_before_failure)
    with pytest.raises(RuntimeError, match="source-to-failure killpoint"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=lambda *_args: (_ for _ in ()).throw(
                AssertionError("scheduler call preceded failure evidence")
            ),
            sleep=lambda _seconds: None,
        )
    source_root = repair._repair_source_root(submission)
    assert repair._load_sealed_repair_source(source_root) == source
    assert not (submission / "journal/REPORT_REPAIR_0001_ORIGINAL_FAILURE.json").exists()

    monkeypatch.setattr(
        repair,
        "_verified_live_repair_source",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("recovery rebound sealed source to live checkout")
        ),
    )
    monkeypatch.setattr(
        repair,
        "_build_failure_evidence",
        lambda *_args, **_kwargs: {
            "schema_version": 1,
            "status": "fixture_failure",
        },
    )
    squeue_round = 0

    def recover(argv, _cwd, _environment):
        nonlocal squeue_round
        if argv[0] == "/usr/local/bin/squeue":
            squeue_round += 1
            return _squeue_result(repair, [])
        assert argv[0] == "/usr/local/bin/sbatch"
        raise RuntimeError("stop after source-authorized calling")

    with pytest.raises(RuntimeError, match="source-authorized calling"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=False,
            runner=recover,
            sleep=lambda _seconds: None,
        )
    calling = json.loads(
        (
            submission / "journal/CALLING_REPORT_REPAIR_0001_SUBMIT.json"
        ).read_text()
    )
    assert calling["repair_source_commit"] == source["repair_source_commit"]
    assert calling["repair_source_files"] == source["repair_source_files"]
    assert report_programs[-1] == source_root / "report.py"


@pytest.mark.parametrize("case", ["broad_only", "exact_held", "round_drift"])
def test_failure_only_recovery_requires_fresh_empty_settled_owner_census(
    repair, tmp_path, monkeypatch, case
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )

    def seal_source(root, _source_value):
        target = repair._repair_source_root(root)
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(repair, "_seal_repair_source_snapshot", seal_source)
    real_seal = repair.seal_json

    def crash_after_failure(path, value):
        digest = real_seal(path, value)
        if path.name == "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json":
            raise RuntimeError("failure-to-calling killpoint")
        return digest

    monkeypatch.setattr(repair, "seal_json", crash_after_failure)
    with pytest.raises(RuntimeError, match="failure-to-calling killpoint"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=lambda *_args: (_ for _ in ()).throw(
                AssertionError("scheduler call preceded durable failure evidence")
            ),
            sleep=lambda _seconds: None,
        )
    monkeypatch.setattr(repair, "seal_json", real_seal)

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
    assert not (journal / "CALLING_REPORT_REPAIR_0001_SUBMIT.json").exists()
    assert not (journal / "REPORT_REPAIR_0001_SUBMITTED.json").exists()


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
        submission / "journal/CALLING_REPORT_REPAIR_0001_SUBMIT.json"
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
        submission / "journal/REPORT_REPAIR_0001_SUBMITTED.json"
    ).exists()


@pytest.mark.parametrize(
    "name",
    [
        "REPORT_REPAIR_0001_TERMINAL_SUBMIT_FAILURE.json",
        "REPORT_REPAIR_0001_TERMINAL_RELEASE_DENIED.json",
        "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
        "REPORT_REPAIR_0001_CANCEL_AUTHORIZED_0000.json",
        "CALLING_REPORT_REPAIR_0001_SCANCEL_0000_0000.json",
        "REPORT_REPAIR_0001_SCANCEL_RESULT_0000_0000.json",
        "REPORT_REPAIR_0001_CANCEL_TERMINAL_0000.json",
        "CALLING_REPORT_REPAIR_0002_SUBMIT.json",
        "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json",
        "REPORT_REPAIR_0001_SUBMITTED.json",
        "REPORT_REPAIR_0001_AUTHORIZED.json",
        "CALLING_REPORT_REPAIR_0001_RELEASE_0000.json",
        "REPORT_REPAIR_0001_RELEASE_RESULT_0000.json",
        "REPORT_REPAIR_0001_RELEASED.json",
        "CALLING_REPORT_REPAIR_0002_SUBMIT.json",
        "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json",
    ],
)
def test_no_submit_calling_with_repair_successor_or_stop_never_calls_scheduler(
    repair, tmp_path, monkeypatch, name
):
    repo, submission, _contract, _source = _configure_minimal_controller(
        repair, tmp_path, monkeypatch
    )
    (submission / "journal" / name).write_bytes(b"hostile")

    def forbidden(*_args):
        raise AssertionError("split repair state must not call scheduler")

    with pytest.raises(
        repair.RepairError,
        match="successor/stop state|forbidden generation|mandatory predecessor chain",
    ):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=True,
            runner=forbidden,
            sleep=lambda _seconds: None,
        )
    assert not (submission / "journal/CALLING_REPORT_REPAIR_0001_SUBMIT.json").exists()


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
    assert (journal / "CALLING_REPORT_REPAIR_0001_SUBMIT.json").is_file()
    (journal / "REPORT_REPAIR_0002_CANCEL_AUTHORIZED_0000.json").write_bytes(
        b"forbidden successor generation"
    )

    def forbidden(*_args):
        raise AssertionError("forbidden generation must precede scheduler recovery")

    with pytest.raises(repair.RepairError, match="forbidden generation"):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=False,
            runner=forbidden,
            sleep=lambda _seconds: None,
        )
    assert not (journal / "REPORT_REPAIR_0001_SUBMITTED.json").exists()


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
    (journal / "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json").write_bytes(
        b"orphan worker terminal"
    )

    def forbidden(*_args):
        raise AssertionError("orphan worker terminal must precede scheduler recovery")

    with pytest.raises(
        repair.RepairError,
        match="positive successor lacks submitted evidence|mandatory predecessor chain",
    ):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=False,
            runner=forbidden,
            sleep=lambda _seconds: None,
        )
    assert not (journal / "REPORT_REPAIR_0001_SUBMITTED.json").exists()
    assert not (journal / "REPORT_REPAIR_0001_AUTHORIZED.json").exists()


@pytest.mark.parametrize(
    "name",
    [
        "REPORT_REPAIR_0001_AUTHORIZED.json",
        "CALLING_REPORT_REPAIR_0001_RELEASE_0000.json",
        "REPORT_REPAIR_0001_RELEASE_RESULT_0000.json",
        "REPORT_REPAIR_0001_RELEASED.json",
        "REPORT_REPAIR_0001_TERMINAL_RELEASE_DENIED.json",
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
    assert (journal / "CALLING_REPORT_REPAIR_0001_SUBMIT.json").is_file()
    assert not (journal / "REPORT_REPAIR_0001_SUBMITTED.json").exists()
    (journal / name).write_bytes(b"impossible positive successor")

    def forbidden(*_args):
        raise AssertionError("positive successor must fail before scheduler recovery")

    with pytest.raises(
        repair.RepairError, match="positive successor lacks submitted evidence"
    ):
        repair.execute_report_repair(
            repo,
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            allow_initial_submission=False,
            runner=forbidden,
            sleep=lambda _seconds: None,
        )
    assert not (journal / "REPORT_REPAIR_0001_SUBMITTED.json").exists()


@pytest.mark.parametrize(
    "name",
    [
        "CALLING_REPORT_REPAIR_0001_RELEASE_0000.json",
        "REPORT_REPAIR_0001_RELEASE_RESULT_0000.json",
        "REPORT_REPAIR_0001_RELEASED.json",
        "REPORT_REPAIR_0001_TERMINAL_RELEASE_DENIED.json",
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
    assert (journal / "REPORT_REPAIR_0001_SUBMITTED.json").is_file()
    assert not (journal / "REPORT_REPAIR_0001_AUTHORIZED.json").exists()
    (journal / name).write_bytes(b"impossible release successor")

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
    assert not (journal / "REPORT_REPAIR_0001_AUTHORIZED.json").exists()


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
        repair._release_authorized_job(
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            authorization,
            "a" * 64,
            crash_after_calling,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    calling = submission / "journal/CALLING_REPORT_REPAIR_0001_RELEASE_0000.json"
    assert calling.is_file()
    assert not (
        submission / "journal/REPORT_REPAIR_0001_RELEASE_RESULT_0000.json"
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

    released = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        authorization,
        "a" * 64,
        recover,
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
        repair._release_authorized_job(
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            authorization,
            "a" * 64,
            crash_after_release_calling,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )

    ambiguous_rows = [_repair_row(repair, job_id="555555")]
    if not different_only:
        ambiguous_rows.insert(0, _repair_row(repair, job_id="444444"))
    real_seal = repair.seal_json

    def crash_after_ambiguous_result(path, value):
        digest = real_seal(path, value)
        if path.name == "REPORT_REPAIR_0001_RELEASE_RESULT_0000.json":
            raise RuntimeError("after ambiguous result")
        return digest

    monkeypatch.setattr(repair, "seal_json", crash_after_ambiguous_result)

    def observe_ambiguity(argv, _cwd, _environment):
        assert argv[0] == "/usr/local/bin/squeue"
        return _squeue_result(repair, ambiguous_rows)

    with pytest.raises(RuntimeError, match="after ambiguous result"):
        repair._release_authorized_job(
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            authorization,
            "a" * 64,
            observe_ambiguity,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    assert (journal / "REPORT_REPAIR_0001_CANCEL_AUTHORIZED_0000.json").is_file()
    assert (journal / "REPORT_REPAIR_0001_RELEASE_RESULT_0000.json").is_file()
    assert not (journal / "REPORT_REPAIR_0001_RELEASED.json").exists()

    monkeypatch.setattr(repair, "seal_json", real_seal)
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

    terminal = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        authorization,
        "a" * 64,
        resume_cleanup,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"
    assert sum(call[0] == "/usr/local/bin/scancel" for call in calls) == 1
    assert all(call[0] != "/usr/local/bin/scontrol" for call in calls)
    assert not (journal / "REPORT_REPAIR_0001_RELEASED.json").exists()


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

    terminal = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        runner,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"
    assert sum(call[0] == "/usr/local/bin/scontrol" for call in calls) == 1
    assert sum(call[0] == "/usr/local/bin/scancel" for call in calls) == 1
    journal = submission / "journal"
    assert (journal / "REPORT_REPAIR_0001_CANCEL_AUTHORIZED_0000.json").is_file()
    assert not (journal / "REPORT_REPAIR_0001_RELEASED.json").exists()


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

    outcome = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        runner,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert outcome["status"] == (
        "report_repair_release_effect_awaiting_unambiguous_namespace"
    )
    assert outcome["accounting_classification"] == "active"
    assert sum(call[0] == "/usr/local/bin/scontrol" for call in calls) == 1
    assert all(call[0] != "/usr/local/bin/scancel" for call in calls)
    assert not (submission / "journal/REPORT_REPAIR_0001_RELEASED.json").exists()
    assert not list(
        (submission / "journal").glob(
            "REPORT_REPAIR_0001_CANCEL_AUTHORIZED_*.json"
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

    outcome = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        runner,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert outcome["status"] == "report_repair_release_effect_awaiting_accounting"
    assert outcome["accounting_classification"] == expected_classification
    assert sum(call[0] == "/usr/local/bin/scontrol" for call in calls) == 1
    assert not (submission / "journal/REPORT_REPAIR_0001_RELEASED.json").exists()


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
        repair._release_authorized_job(
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            authorization,
            "a" * 64,
            lose_release_response,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )

    real_seal = repair.seal_json

    def crash_after_effect_result(path, value):
        digest = real_seal(path, value)
        if path.name == "REPORT_REPAIR_0001_RELEASE_RESULT_0000.json":
            raise RuntimeError("after release effect result")
        return digest

    monkeypatch.setattr(repair, "seal_json", crash_after_effect_result)

    def released_absent(argv, _cwd, _environment):
        assert argv[0] == "/usr/local/bin/squeue"
        return _squeue_result(repair, [])

    with pytest.raises(RuntimeError, match="after release effect result"):
        repair._release_authorized_job(
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            authorization,
            "a" * 64,
            released_absent,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    monkeypatch.setattr(repair, "seal_json", real_seal)
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

    terminal = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        authorization,
        "a" * 64,
        recycled_held,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"
    assert sum(call[0] == "/usr/local/bin/scancel" for call in calls) == 1
    assert all(call[0] != "/usr/local/bin/scontrol" for call in calls)
    assert not (submission / "journal/REPORT_REPAIR_0001_RELEASED.json").exists()


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

    terminal0 = repair._cleanup_repair_rows(
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

    terminal1 = repair._cleanup_repair_rows(
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
        submission / "journal/REPORT_REPAIR_0001_CANCEL_AUTHORIZED_0001.json"
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
        repair._cleanup_repair_rows(
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
        submission / "journal/REPORT_REPAIR_0001_CANCEL_AUTHORIZED_0000.json"
    )
    authority = json.loads(authority_path.read_text())
    authority["job_ids"] = ["999999"]
    authority_path.chmod(0o600)
    authority_path.write_text(json.dumps(authority, sort_keys=True, indent=2) + "\n")
    authority_path.chmod(0o444)

    def forbidden(*_args):
        raise AssertionError("scheduler must not be called for forged cleanup evidence")

    with pytest.raises(repair.RepairError, match="cleanup"):
        repair._cleanup_repair_rows(
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
        repair._cleanup_repair_rows(
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
        submission / "journal/REPORT_REPAIR_0001_CANCEL_AUTHORIZED_0000.json"
    ).is_file()
    assert not list(
        (submission / "journal").glob("CALLING_REPORT_REPAIR_0001_SCANCEL_*.json")
    )
    monkeypatch.setattr(repair, "_cancel_attempt_prefix", real_prefix)
    squeue_round = 0

    def resume(argv, _cwd, _environment):
        nonlocal squeue_round
        if argv[0] == "/usr/local/bin/scancel":
            return _command_result(repair)
        squeue_round += 1
        return _squeue_result(repair, [])

    terminal = repair._cleanup_repair_rows(
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
    assert (journal / "REPORT_REPAIR_0001_AUTHORIZED.json").is_file()
    assert not (journal / "CALLING_REPORT_REPAIR_0001_RELEASE_0000.json").exists()
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
    assert not (submission / "journal/REPORT_REPAIR_0001_AUTHORIZED.json").exists()
    assert not (submission / "journal/REPORT_REPAIR_0001_RELEASED.json").exists()
    authority = json.loads(
        (
            submission
            / "journal/REPORT_REPAIR_0001_CANCEL_AUTHORIZED_0000.json"
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
    assert (submission / "journal/CALLING_REPORT_REPAIR_0001_SUBMIT.json").is_file()
    assert not (submission / "journal/REPORT_REPAIR_0001_SUBMITTED.json").exists()

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
    assert not (submission / "journal/REPORT_REPAIR_0001_AUTHORIZED.json").exists()


def _fixture_repair_authority(report, submission, job_id="444444"):
    return {
        "original_report_job_id": "33311218",
        "repair_report_job_id": job_id,
        "original_failure_evidence": "journal/REPORT_REPAIR_0001_ORIGINAL_FAILURE.json",
        "original_failure_evidence_sha256": "1" * 64,
        "worker_receipt_map_sha256": "2" * 64,
        "snapshot_root": str(submission / "source-snapshot/repo"),
        "snapshot_inventory_sha256": "3" * 64,
        "original_package_protocol_sha256": "4" * 64,
        "repair_source_root": str(submission / "report-repair/attempt-0001/source"),
        "repair_source_commit": "5" * 40,
        "repair_package_protocol_sha256": "6" * 64,
        "repair_source_files_sha256": "7" * 64,
        "_validated_release_sha256": "8" * 64,
    }


def test_repair_publication_v2_authority_preserves_exact_14_key_commit(
    report, tmp_path, monkeypatch
):
    submission = tmp_path / "submission"
    submission.mkdir()
    authorization_sha = "9" * 64
    repair_authority = _fixture_repair_authority(report, submission)
    expected = report._repair_publication_authority(
        repair_authority, authorization_sha, 1
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
        repair_attempt=1,
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

    other = tmp_path / "forged"
    other.mkdir()
    forged = json.loads(json.dumps(provenance))
    forged["publication_authority"]["repair_report_job_id"] = "555555"
    with pytest.raises(report.ReportError, match="provenance authority"):
        report.publish_report(
            other,
            "a" * 64,
            {"schema_version": 1},
            decision,
            forged,
            repair_attempt=1,
            repair_authorization_sha256=authorization_sha,
        )
    assert not (other / "report").exists()


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
    assert not (submission / "journal/REPORT_REPAIR_0001_SUBMITTED.json").exists()
    assert not (submission / "journal/REPORT_REPAIR_0001_AUTHORIZED.json").exists()
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

    terminal = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        absent,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_release_denied"
    assert terminal["publication_allowed"] is False
    assert all(call[0] != "/usr/local/bin/scontrol" for call in calls)
    assert not list(
        (submission / "journal").glob(
            "CALLING_REPORT_REPAIR_0001_RELEASE_*.json"
        )
    )

    repeat_calls = []

    def still_absent(argv, _cwd, _environment):
        repeat_calls.append(list(argv))
        assert argv[0] == "/usr/local/bin/squeue"
        return _squeue_result(repair, [])

    repeated = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        still_absent,
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

    cleaned = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        delayed_visible,
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

    terminal = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        runner,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_cleanup_complete"
    assert sum(call[0] == "/usr/local/bin/scancel" for call in calls) == 1
    assert all(call[0] != "/usr/local/bin/scontrol" for call in calls)
    assert not list(
        (submission / "journal").glob(
            "CALLING_REPORT_REPAIR_0001_RELEASE_*.json"
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
        repair._release_authorized_job(
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            {"repair_report_job_id": "444444"},
            "a" * 64,
            warning,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    assert (journal / "CALLING_REPORT_REPAIR_0001_RELEASE_0000.json").is_file()
    assert not (journal / "REPORT_REPAIR_0001_RELEASE_RESULT_0000.json").exists()
    assert not (journal / "REPORT_REPAIR_0001_RELEASED.json").exists()

    calls = []

    def reconcile(argv, _cwd, _environment):
        calls.append(list(argv))
        if argv[0] == "/usr/local/bin/squeue":
            return _squeue_result(repair, [])
        if argv[0] == "/usr/local/bin/sacct":
            return _repair_sacct_result(repair)
        raise AssertionError(argv)

    released = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        reconcile,
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
        repair._release_authorized_job(
            submission,
            repair.EXPECTED_SUBMISSION_SHA256,
            contract,
            {"repair_report_job_id": "444444"},
            "a" * 64,
            crash_before_post_release_census,
            _FakeLocks(),
            sleep=lambda _seconds: None,
        )
    journal = submission / "journal"
    assert (journal / "CALLING_REPORT_REPAIR_0001_RELEASE_0000.json").is_file()
    assert (journal / "REPORT_REPAIR_0001_RELEASE_RESULT_0000.json").is_file()
    assert not (journal / "REPORT_REPAIR_0001_RELEASED.json").exists()

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

    released = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        recover,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert released["status"] == "report_repair_released"
    assert len(released["release_attempts"]) == 1
    assert all(call[0] in {"/usr/local/bin/squeue", "/usr/local/bin/sacct"} for call in calls)
    assert (not visible_running) == any(
        call[0] == "/usr/local/bin/sacct" for call in calls
    )
    assert not (journal / "REPORT_REPAIR_0001_TERMINAL_RELEASE_DENIED.json").exists()


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

    terminal = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        "a" * 64,
        runner,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert terminal["status"] == "report_repair_terminal_worker_failure"
    assert terminal["reason"] == "repair_worker_terminal_before_release_evidence"
    assert terminal["publication_allowed"] is False
    journal = submission / "journal"
    assert (journal / "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json").is_file()
    assert not (journal / "REPORT_REPAIR_0001_RELEASED.json").exists()
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

    released = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        authorization_sha,
        release_runner,
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

    released = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        authorization_sha,
        release_runner,
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
    assert (journal / "REPORT_REPAIR_0001_CANCEL_AUTHORIZED_0000.json").is_file()
    assert (journal / "CALLING_REPORT_REPAIR_0001_SCANCEL_0000_0000.json").is_file()
    assert not (journal / "REPORT_REPAIR_0001_CANCEL_TERMINAL_0000.json").exists()

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
        (journal / "REPORT_REPAIR_0001_SCANCEL_RESULT_0000_0000.json").read_text()
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

    released = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        authorization_sha,
        release_runner,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    assert released["status"] == "report_repair_released"
    (submission / "journal/REPORT_REPAIR_0001_TERMINAL_RELEASE_DENIED.json").write_bytes(
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

    released = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        {"repair_report_job_id": "444444"},
        authorization_sha,
        release_runner,
        _FakeLocks(),
        sleep=lambda _seconds: None,
    )
    calling_path = submission / "journal/CALLING_REPORT_REPAIR_0001_RELEASE_0000.json"
    result_path = submission / "journal/REPORT_REPAIR_0001_RELEASE_RESULT_0000.json"
    released_path = submission / "journal/REPORT_REPAIR_0001_RELEASED.json"
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
        "attempt": 1,
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
            repair._release_authorized_job(
                submission,
                repair.EXPECTED_SUBMISSION_SHA256,
                contract,
                authorization,
                "a" * 64,
                still_held,
                _FakeLocks(),
                sleep=lambda _seconds: None,
            )

    journal = submission / "journal"
    assert len(list(journal.glob("REPORT_REPAIR_0001_RELEASE_RESULT_*.json"))) == 3
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

    outcome = repair._release_authorized_job(
        submission,
        repair.EXPECTED_SUBMISSION_SHA256,
        contract,
        authorization,
        "a" * 64,
        final_observation,
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
        repair._cleanup_repair_rows(
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
    assert (journal / "CALLING_REPORT_REPAIR_0001_SCANCEL_0000_0000.json").is_file()
    assert not (journal / "REPORT_REPAIR_0001_SCANCEL_RESULT_0000_0000.json").exists()

    def reconcile(argv, _cwd, _environment):
        assert argv[0] == "/usr/local/bin/squeue"
        return _squeue_result(repair, [])

    terminal = repair._cleanup_repair_rows(
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
    assert not (submission / "journal/REPORT_REPAIR_0001_AUTHORIZED.json").exists()


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
    (submission / "journal/REPORT_REPAIR_0001_CANCEL_AUTHORIZED_0000.json").write_bytes(
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
    }
    receipt = {"report_job_id": repair.EXPECTED_ORIGINAL_REPORT_JOB_ID}
    receipt_map = {"schema_version": 1, "files": {}}
    source = _repair_source(repair)
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
        calling_sha256="b" * 64,
        submitted_sha256="c" * 64,
        job_id="444444",
        census=census,
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
            calling_sha256="b" * 64,
            submitted_sha256="c" * 64,
        )


@pytest.mark.parametrize(
    "name",
    [
        "REPORT_REPAIR_0001_TERMINAL_SUBMIT_FAILURE.json",
        "REPORT_REPAIR_0001_TERMINAL_RELEASE_DENIED.json",
        "REPORT_REPAIR_0001_CANCEL_AUTHORIZED_0000.json",
        "CALLING_REPORT_REPAIR_0001_SCANCEL_0000_0000.json",
        "REPORT_REPAIR_0001_SCANCEL_RESULT_0000_0000.json",
        "REPORT_REPAIR_0001_CANCEL_TERMINAL_0000.json",
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
        journal / "REPORT_REPAIR_0001_RELEASE_RESULT_0000.json",
        {
            "schema_version": 1,
            "mode": "lost_response_reconciled_ambiguous_identity",
        },
    )
    assert report._durable_repair_stop_prefix_exists(tmp_path)
