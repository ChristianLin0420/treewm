"""Focused fail-closed tests for Exp23 submission, cancellation, and reporting."""

from __future__ import annotations

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
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return {
        "lexical_executable": str(python),
        "venv_site_packages": str(venv / "lib" / version / "site-packages"),
        "base_site_packages": str(base / "lib" / version / "site-packages"),
        "resolved_executable_sha256": submit.file_sha256(target),
        "resolved_executable_size": target.stat().st_size,
    }


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
        "campaign_id": "treewm-executable-prefix-repair-pilot-v1",
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
