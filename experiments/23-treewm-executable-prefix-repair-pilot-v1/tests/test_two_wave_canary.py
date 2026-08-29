from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


worker = _load("exp23_test_canary_worker", "canary_worker.py")
controller = _load("exp23_test_canary_controller", "two_wave_canary.py")
submit = _load("exp23_test_canary_submit", "submit.py")


def _authorization(root: Path) -> dict:
    source_sha256 = {
        name: hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()
        for name in controller.CANARY_SOURCE_FILES
    }
    controller_lock = {
        "path": str(root / ".CANARY_CONTROLLER.lock"),
        "device": 1,
        "inode": 2,
        "uid": os.getuid(),
        "mode": 0o600,
    }
    scheduler_control_plane = {"schema_version": 1, "mode": "fixture"}
    authorization = {
        "schema_version": 1,
        "status": "authorized_two_wave_gpu_canary",
        "campaign_id": worker.CAMPAIGN_ID,
        "canary_token": "0123456789abcdef",
        "state_root": str(root.absolute()),
        "controller_identity_sha256": "0" * 64,
        "controller_lock": controller_lock,
        "worker_sha256": source_sha256["canary_worker.py"],
        "source_sha256": source_sha256,
        "package_protocol_sha256": "a" * 64,
        "job_ids": {"wave0": "7000", "wave1": "8000", "report": "9000"},
        "job_names": {
            "wave0": "exp23-launch8-canary-0123456789abcdef-wave0",
            "wave1": "exp23-launch8-canary-0123456789abcdef-wave1",
            "report": "exp23-launch8-canary-0123456789abcdef-report",
        },
        "dependencies": {
            "wave0": "none",
            "wave1": "afterok:7000",
            "report": "afterok:8000",
        },
        "scheduler_comment": "treewm-exp23-canary:0123456789abcdef",
        "scheduler_executables": {
            "submit": "/fixture/bin/sbatch",
            "control": "/fixture/bin/scontrol",
        },
        "scheduler_control_plane": scheduler_control_plane,
        "scheduler_control_plane_sha256": worker.stable_hash(
            scheduler_control_plane
        ),
        "accepted_submission_records_sha256": "0" * 64,
        "accepted_scheduler_evidence_sha256": "0" * 64,
        "within_wave_requeue": False,
    }
    authorization["controller_identity_sha256"] = _json_payload_sha256(
        _controller_identity(root, authorization)
    )
    records, evidence = _accepted_scheduler_fixtures(root, authorization)
    authorization["accepted_submission_records_sha256"] = worker.stable_hash(records)
    authorization["accepted_scheduler_evidence_sha256"] = worker.stable_hash(evidence)
    return authorization


def _json_payload_sha256(value: dict) -> str:
    return hashlib.sha256((worker.canonical_json(value) + "\n").encode("ascii")).hexdigest()


def _controller_identity(root: Path, authorization: dict) -> dict:
    return {
        "schema_version": 1,
        "status": "canary_controller_claimed",
        "campaign_id": worker.CAMPAIGN_ID,
        "scientific": False,
        "state_root": str(root),
        "canary_token": authorization["canary_token"],
        "job_names": authorization["job_names"],
        "scheduler_comment": authorization["scheduler_comment"],
        "controller_lock": authorization["controller_lock"],
        "source_sha256": authorization["source_sha256"],
        "package_protocol_sha256": authorization["package_protocol_sha256"],
        "scheduler_control_plane": authorization["scheduler_control_plane"],
        "scheduler_control_plane_sha256": authorization[
            "scheduler_control_plane_sha256"
        ],
    }


def _calling_payload(root: Path, authorization: dict, role: str) -> dict:
    return {
        "schema_version": 1,
        "status": "canary_scheduler_calling",
        "campaign_id": worker.CAMPAIGN_ID,
        "state_root": str(root),
        "canary_token": authorization["canary_token"],
        "role": role,
        "job_name": authorization["job_names"][role],
        "scheduler_comment": authorization["scheduler_comment"],
        "command": worker._expected_submission_commands(root, authorization)[role],
        "controller_lock": authorization["controller_lock"],
    }


def _accepted_scheduler_fixtures(
    root: Path, authorization: dict
) -> tuple[dict, dict]:
    observation = {"schema_version": 1, "mode": "fixture"}
    records = {
        role: {
            "command": command,
            "returncode": 0,
            "stdout": f"{authorization['job_ids'][role]}\n",
            "stderr": "",
            "reconciled_job_ids": [authorization["job_ids"][role]],
            "scheduler_control_plane": observation,
            "calling_sha256": _json_payload_sha256(
                _calling_payload(root, authorization, role)
            ),
        }
        for role, command in worker._expected_submission_commands(
            root, authorization
        ).items()
    }
    jobs = authorization["job_ids"]
    names = authorization["job_names"]
    comment = authorization["scheduler_comment"]
    control = authorization["scheduler_executables"]["control"]
    hold_stdout = (
        f"JobId={jobs['wave0']} JobName={names['wave0']} Comment={comment} "
        "JobState=PENDING Reason=JobHeldUser\n"
    )
    evidence = {
        "wave0_accepted_hold": {
            "command": [control, "show", "job", jobs["wave0"], "--oneliner"],
            "returncode": 0,
            "stdout": hold_stdout,
            "stderr": "",
            "state": "PENDING",
            "reason": "JobHeldUser",
            "scheduler_control_plane": observation,
        }
    }
    for role, predecessor in (("wave1", "wave0"), ("report", "wave1")):
        dependency = f"afterok:{jobs[predecessor]}(unfulfilled)"
        evidence[f"{role}_accepted_dependency"] = {
            "command": [control, "show", "job", jobs[role], "--oneliner"],
            "returncode": 0,
            "stdout": (
                f"JobId={jobs[role]} JobName={names[role]} Comment={comment} "
                f"JobState=PENDING Dependency={dependency} "
                "KillOInInvalidDependent=Yes\n"
            ),
            "stderr": "",
            "dependency": dependency,
            "role": role,
            "kill_on_invalid_dependency": "Yes",
            "scheduler_control_plane": observation,
        }
    return records, evidence


def _seal_controller_contract(
    root: Path,
    authorization: dict,
    *,
    records: dict | None = None,
    evidence: dict | None = None,
) -> None:
    source = root / "source"
    source.mkdir(mode=0o700)
    for name in controller.CANARY_SOURCE_FILES:
        destination = source / name
        destination.write_bytes((PACKAGE / name).read_bytes())
        destination.chmod(0o444)
    source.chmod(0o555)
    worker.seal_json(
        root / "CANARY_CONTROLLER_IDENTITY.json",
        _controller_identity(root, authorization),
    )
    for role, filename in (
        ("wave0", "CANARY_WAVE0_CALLING.json"),
        ("wave1", "CANARY_WAVE1_CALLING.json"),
        ("report", "CANARY_REPORT_CALLING.json"),
    ):
        worker.seal_json(root / filename, _calling_payload(root, authorization, role))
    default_records, default_evidence = _accepted_scheduler_fixtures(
        root, authorization
    )
    records = default_records if records is None else records
    evidence = default_evidence if evidence is None else evidence
    worker.seal_json(
        root / "CANARY_WAVE0_SUBMITTED.json",
        {
            "schema_version": 1,
            "status": "canary_wave0_submitted_held",
            "job_id": authorization["job_ids"]["wave0"],
            "accepted_submission_record": records["wave0"],
            "accepted_hold": evidence["wave0_accepted_hold"],
        },
    )
    for role in ("wave1", "report"):
        worker.seal_json(
            root / f"CANARY_{role.upper()}_SUBMITTED.json",
            {
                "schema_version": 1,
                "status": f"canary_{role}_submitted",
                "job_id": authorization["job_ids"][role],
                "accepted_submission_record": records[role],
                "accepted_dependency": evidence[f"{role}_accepted_dependency"],
            },
        )
    worker.seal_json(root / worker.AUTH_NAME, authorization)
    auth_sha256 = worker.sha256_file(root / worker.AUTH_NAME)
    receipt = {
        "schema_version": 1,
        "status": "two_wave_gpu_canary_ready_to_release",
        "campaign_id": worker.CAMPAIGN_ID,
        "scientific": False,
        "state_root": str(root),
        "canary_token": authorization["canary_token"],
        "controller_identity_sha256": authorization[
            "controller_identity_sha256"
        ],
        "controller_lock": authorization["controller_lock"],
        "authorization_sha256": auth_sha256,
        "job_ids": authorization["job_ids"],
        "job_names": authorization["job_names"],
        "dependencies": authorization["dependencies"],
        "scheduler_comment": authorization["scheduler_comment"],
        "scheduler_executables": authorization["scheduler_executables"],
        "scheduler_control_plane": authorization["scheduler_control_plane"],
        "scheduler_control_plane_sha256": authorization[
            "scheduler_control_plane_sha256"
        ],
        "source_sha256": authorization["source_sha256"],
        "package_protocol_sha256": authorization["package_protocol_sha256"],
        "accepted_submission_records_sha256": authorization[
            "accepted_submission_records_sha256"
        ],
        "accepted_scheduler_evidence_sha256": authorization[
            "accepted_scheduler_evidence_sha256"
        ],
        "wave0_accepted_hold": evidence["wave0_accepted_hold"],
        "wave1_accepted_dependency": evidence["wave1_accepted_dependency"],
        "report_accepted_dependency": evidence["report_accepted_dependency"],
        "accepted_submission_records": records,
    }
    worker.seal_json(root / worker.RECEIPT_NAME, receipt)
    worker.seal_json(
        root / worker.READY_TO_RELEASE_NAME,
        {
            "schema_version": 1,
            "status": "canary_ready_to_release",
            "campaign_id": worker.CAMPAIGN_ID,
            "authorization_sha256": auth_sha256,
            "receipt_sha256": worker.sha256_file(root / worker.RECEIPT_NAME),
            "job_ids": authorization["job_ids"],
            "dependencies": authorization["dependencies"],
            "accepted_submission_records_sha256": authorization[
                "accepted_submission_records_sha256"
            ],
            "accepted_scheduler_evidence_sha256": authorization[
                "accepted_scheduler_evidence_sha256"
            ],
        },
    )
    ready_to_release_sha256 = worker.sha256_file(
        root / worker.READY_TO_RELEASE_NAME
    )
    jobs = authorization["job_ids"]
    names = authorization["job_names"]
    comment = authorization["scheduler_comment"]
    control = authorization["scheduler_executables"]["control"]
    release_stdout = (
        f"JobId={jobs['wave0']} JobName={names['wave0']} Comment={comment} "
        "JobState=RUNNING Reason=None\n"
    )
    release_calling_sha256 = worker.seal_json(
        root / "CANARY_WAVE0_RELEASE_CALLING.json",
        {
            "schema_version": 1,
            "status": "canary_scheduler_calling",
            "campaign_id": worker.CAMPAIGN_ID,
            "state_root": str(root),
            "canary_token": authorization["canary_token"],
            "role": "wave0_release",
            "job_name": names["wave0"],
            "scheduler_comment": comment,
            "command": [control, "release", jobs["wave0"]],
            "controller_lock": authorization["controller_lock"],
            "authorization_sha256": auth_sha256,
            "receipt_sha256": worker.sha256_file(root / worker.RECEIPT_NAME),
        },
    )
    worker.seal_json(
        root / "CANARY_WAVE0_RELEASED.json",
        {
            "schema_version": 1,
            "status": "canary_wave0_released",
            "campaign_id": worker.CAMPAIGN_ID,
            "authorization_sha256": auth_sha256,
            "receipt_sha256": worker.sha256_file(root / worker.RECEIPT_NAME),
            "ready_to_release_sha256": ready_to_release_sha256,
            "calling_sha256": release_calling_sha256,
            "wave0_job_id": jobs["wave0"],
            "wave0_release": {
                "release_command": [control, "release", jobs["wave0"]],
                "release_returncode": 0,
                "release_stdout": "",
                "release_stderr": "",
                "show_command": [
                    control,
                    "show",
                    "job",
                    jobs["wave0"],
                    "--oneliner",
                ],
                "show_returncode": 0,
                "show_stdout": release_stdout,
                "show_stderr": "",
                "state": "RUNNING",
                "reason": "None",
                "release_scheduler_control_plane": {
                    "schema_version": 1,
                    "mode": "fixture",
                },
                "show_scheduler_control_plane": {
                    "schema_version": 1,
                    "mode": "fixture",
                },
            },
        },
    )


def _rewrite_sealed_json(path: Path, value: dict) -> None:
    path.chmod(0o600)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)


def test_default_canary_description_is_read_only(capsys: pytest.CaptureFixture[str]) -> None:
    assert controller.main([]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value == controller.description()
    assert value["status"] == "real_gpu_two_wave_canary_available_not_run"
    assert value["scheduler_mutation_on_describe"] is False
    assert value["preflight_invocation"] is False


def test_real_canary_confirmation_fails_before_state_or_scheduler(
    tmp_path: Path,
) -> None:
    state = tmp_path / "exp23-launch8-two-wave-canary-denied"
    with pytest.raises(controller.CanarySubmissionError, match="confirmation"):
        controller.submit_real_canary(REPO, state, "wrong")
    assert not os.path.lexists(state)


@pytest.mark.parametrize("attempt_kind", ("failed", "accepted"))
def test_historical_canary_root_is_forbidden_even_if_prior_root_is_absent(
    tmp_path: Path, attempt_kind: str,
) -> None:
    repo = (tmp_path / "repo").absolute()
    (repo / "outputs").mkdir(parents=True)
    failed_root = (
        repo
        / controller.CANARY_PARENT_RELATIVE
        / "exp23-launch8-two-wave-canary-failed"
    )
    manifest = {
        "paths": {
            "run_root": str(repo / "outputs" / "scientific"),
            "transaction_lock": str(repo / "outputs" / "scientific.lock"),
        },
        "superseded_launches": [],
        "launch_contract": {
            "real_gpu_two_wave_canary": {
                "failed_attempts": ([
                    {
                        "state_root": str(failed_root),
                        "canary_token": "e09ce7d5a0cef1b0",
                        "job_ids_by_role": {
                            "wave0": ["33285485"],
                            "wave1": ["33285486"],
                            "report": (
                                ["33285487"] if attempt_kind == "accepted" else []
                            ),
                        },
                    }
                ] if attempt_kind == "failed" else []),
                "accepted_attempts": ([
                    {
                        "state_root": str(failed_root),
                        "canary_token": "e09ce7d5a0cef1b0",
                        "job_ids_by_role": {
                            "wave0": ["33285485"],
                            "wave1": ["33285486"],
                            "report": ["33285487"],
                        },
                    }
                ] if attempt_kind == "accepted" else []),
                }
            },
    }
    assert not os.path.lexists(failed_root)
    with pytest.raises(
        controller.CanarySubmissionError,
        match="historical-canary namespace",
    ):
        controller._prepare_state(repo, failed_root, manifest)
    assert not os.path.lexists(failed_root)


@pytest.mark.parametrize("attempt_kind", ("failed", "accepted"))
def test_historical_canary_token_is_rejected_before_state_or_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attempt_kind: str
) -> None:
    state = (
        tmp_path / "exp23-launch8-two-wave-canary-failed-token"
    ).absolute()
    manifest = {
        "campaign_id": worker.CAMPAIGN_ID,
        "status": "sealed_launch_ready_unsubmitted",
        "formal_validation": False,
        "paths": {"python": sys.executable},
        "launch_contract": {
            "real_gpu_two_wave_canary": {
                "failed_attempts": ([
                    {
                        "state_root": str(
                            tmp_path
                            / "exp23-launch8-two-wave-canary-prior-attempt"
                        ),
                        "canary_token": "e09ce7d5a0cef1b0",
                        "job_ids_by_role": {
                            "wave0": ["33285485"],
                            "wave1": ["33285486"],
                            "report": [],
                        },
                    }
                ] if attempt_kind == "failed" else []),
                "accepted_attempts": ([
                    {
                        "state_root": str(
                            tmp_path
                            / "exp23-launch8-two-wave-canary-accepted-attempt"
                        ),
                        "canary_token": "e09ce7d5a0cef1b0",
                        "job_ids_by_role": {
                            "wave0": ["33285485"],
                            "wave1": ["33285486"],
                            "report": ["33285487"],
                        },
                    }
                ] if attempt_kind == "accepted" else []),
            }
        },
    }
    fake_campaign = SimpleNamespace(
        load_contract=lambda _repo: (copy.deepcopy(manifest), {}),
        verify_protocol_lock=lambda _package: "a" * 64,
    )
    monkeypatch.setattr(controller, "_load_submit", lambda _repo: submit)
    monkeypatch.setattr(controller, "_load_campaign", lambda _repo: fake_campaign)
    original_read_json = submit.read_json
    monkeypatch.setattr(
        submit,
        "read_json",
        lambda path: (
            copy.deepcopy(manifest)
            if Path(path).name == "manifest.json"
            else original_read_json(path)
        ),
    )
    monkeypatch.setattr(
        controller.secrets, "token_hex", lambda _size: "e09ce7d5a0cef1b0"
    )
    monkeypatch.setattr(
        controller,
        "_prepare_state",
        lambda *_args: pytest.fail("state preparation preceded failed-token check"),
    )
    original_flags = controller.sys.flags

    class IsolatedFlags:
        isolated = 1
        no_site = 1
        dont_write_bytecode = 1
        safe_path = 1

        def __getattr__(self, name):
            return getattr(original_flags, name)

    monkeypatch.setattr(controller.sys, "flags", IsolatedFlags())
    monkeypatch.setattr(controller.sys, "dont_write_bytecode", True)
    for key in list(controller.os.environ):
        if key.startswith("TREEWM_") or key in {"RANK", "WORLD_SIZE", "LOCAL_RANK"}:
            monkeypatch.delenv(key)
    for name in list(controller.sys.modules):
        if name == "treewm" or name.startswith("treewm."):
            monkeypatch.delitem(controller.sys.modules, name)
    with pytest.raises(
        controller.CanarySubmissionError,
        match="reuses a historical canary token",
    ):
        controller.submit_real_canary(
            tmp_path,
            state,
            controller.CONFIRMATION,
            scheduler_runner=lambda *_args: pytest.fail("scheduler was called"),
        )
    assert not os.path.lexists(state)


@pytest.mark.parametrize("attempt_kind", ("failed", "accepted"))
def test_historical_canary_root_recovery_is_rejected_before_any_root_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attempt_kind: str
) -> None:
    repo = tmp_path.absolute()
    failed_root = (
        repo
        / controller.CANARY_PARENT_RELATIVE
        / "exp23-launch8-two-wave-canary-failed-recovery"
    )
    manifest = {
        "campaign_id": worker.CAMPAIGN_ID,
        "status": "sealed_launch_ready_unsubmitted",
        "formal_validation": False,
        "paths": {"python": sys.executable},
        "launch_contract": {
            "real_gpu_two_wave_canary": {
                "failed_attempts": ([
                    {
                        "state_root": str(failed_root),
                        "canary_token": "e09ce7d5a0cef1b0",
                        "job_ids_by_role": {
                            "wave0": ["33285485"],
                            "wave1": ["33285486"],
                            "report": [],
                        },
                    }
                ] if attempt_kind == "failed" else []),
                "accepted_attempts": ([
                    {
                        "state_root": str(failed_root),
                        "canary_token": "e09ce7d5a0cef1b0",
                        "job_ids_by_role": {
                            "wave0": ["33285485"],
                            "wave1": ["33285486"],
                            "report": ["33285487"],
                        },
                    }
                ] if attempt_kind == "accepted" else []),
            }
        },
    }
    fake_campaign = SimpleNamespace(
        load_contract=lambda _repo: (copy.deepcopy(manifest), {}),
        verify_protocol_lock=lambda _package: pytest.fail(
            "protocol access followed failed-root classification"
        ),
    )
    monkeypatch.setattr(controller, "_load_submit", lambda _repo: submit)
    monkeypatch.setattr(controller, "_load_campaign", lambda _repo: fake_campaign)
    original_read_json = submit.read_json
    root_reads: list[Path] = []

    def guarded_read(path):
        candidate = Path(path)
        if candidate.name == "manifest.json":
            return copy.deepcopy(manifest)
        root_reads.append(candidate)
        return original_read_json(candidate)

    monkeypatch.setattr(submit, "read_json", guarded_read)
    original_is_dir = Path.is_dir
    original_is_symlink = Path.is_symlink
    root_stats: list[tuple[str, Path]] = []

    def guarded_is_dir(path):
        if path == failed_root:
            root_stats.append(("is_dir", path))
        return original_is_dir(path)

    def guarded_is_symlink(path):
        if path == failed_root:
            root_stats.append(("is_symlink", path))
        return original_is_symlink(path)

    monkeypatch.setattr(Path, "is_dir", guarded_is_dir)
    monkeypatch.setattr(Path, "is_symlink", guarded_is_symlink)
    monkeypatch.setattr(
        controller,
        "_CanaryControllerLock",
        lambda _root, **_kwargs: pytest.fail(
            "failed canary controller lock was opened"
        ),
    )
    original_flags = controller.sys.flags

    class IsolatedFlags:
        isolated = 1
        no_site = 1
        dont_write_bytecode = 1
        safe_path = 1

        def __getattr__(self, name):
            return getattr(original_flags, name)

    monkeypatch.setattr(controller.sys, "flags", IsolatedFlags())
    monkeypatch.setattr(controller.sys, "dont_write_bytecode", True)
    for key in list(controller.os.environ):
        if key.startswith("TREEWM_") or key in {"RANK", "WORLD_SIZE", "LOCAL_RANK"}:
            monkeypatch.delenv(key)
    for name in list(controller.sys.modules):
        if name == "treewm" or name.startswith("treewm."):
            monkeypatch.delitem(controller.sys.modules, name)
    with pytest.raises(
        controller.CanarySubmissionError,
        match="historical canary state root cannot be recovered or read",
    ):
        controller.recover_or_cancel_real_canary(
            repo,
            failed_root,
            controller.CONFIRMATION,
            scheduler_runner=lambda *_args: pytest.fail("scheduler was called"),
        )
    assert root_stats == []
    assert root_reads == []
    assert not os.path.lexists(failed_root)


def test_failed_canary_alias_symlink_recovery_never_follows_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path.absolute()
    parent = repo / controller.CANARY_PARENT_RELATIVE
    parent.mkdir(parents=True)
    failed_root = parent / "exp23-launch8-two-wave-canary-failed-target"
    failed_root.mkdir()
    alias_root = parent / "exp23-launch8-two-wave-canary-fresh-alias"
    alias_root.symlink_to(failed_root, target_is_directory=True)
    manifest = {
        "campaign_id": worker.CAMPAIGN_ID,
        "status": "sealed_launch_ready_unsubmitted",
        "formal_validation": False,
        "paths": {"python": sys.executable},
        "launch_contract": {
            "real_gpu_two_wave_canary": {
                "failed_attempts": [
                    {
                        "state_root": str(failed_root),
                        "canary_token": "e09ce7d5a0cef1b0",
                        "job_ids_by_role": {
                            "wave0": ["33285485"],
                            "wave1": ["33285486"],
                            "report": [],
                        },
                    }
                ],
                "accepted_attempts": [],
            }
        },
    }
    fake_campaign = SimpleNamespace(
        load_contract=lambda _repo: (copy.deepcopy(manifest), {}),
        verify_protocol_lock=lambda _package: "a" * 64,
    )
    monkeypatch.setattr(controller, "_load_submit", lambda _repo: submit)
    monkeypatch.setattr(controller, "_load_campaign", lambda _repo: fake_campaign)
    monkeypatch.setattr(
        submit,
        "read_json",
        lambda path: (
            copy.deepcopy(manifest)
            if Path(path).name == "manifest.json"
            else pytest.fail("failed canary alias caused a state-root read")
        ),
    )
    original_is_dir = Path.is_dir

    def guarded_is_dir(path):
        if path == alias_root:
            pytest.fail("failed canary alias target was followed")
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", guarded_is_dir)
    monkeypatch.setattr(
        controller,
        "_CanaryControllerLock",
        lambda _root, **_kwargs: pytest.fail("failed canary alias lock was opened"),
    )
    original_flags = controller.sys.flags

    class IsolatedFlags:
        isolated = 1
        no_site = 1
        dont_write_bytecode = 1
        safe_path = 1

        def __getattr__(self, name):
            return getattr(original_flags, name)

    monkeypatch.setattr(controller.sys, "flags", IsolatedFlags())
    monkeypatch.setattr(controller.sys, "dont_write_bytecode", True)
    for key in list(controller.os.environ):
        if key.startswith("TREEWM_") or key in {"RANK", "WORLD_SIZE", "LOCAL_RANK"}:
            monkeypatch.delenv(key)
    for name in list(controller.sys.modules):
        if name == "treewm" or name.startswith("treewm."):
            monkeypatch.delitem(controller.sys.modules, name)
    with pytest.raises(
        controller.CanarySubmissionError,
        match="canary recovery state root differs",
    ):
        controller.recover_or_cancel_real_canary(
            repo,
            alias_root,
            controller.CONFIRMATION,
            scheduler_runner=lambda *_args: pytest.fail("scheduler was called"),
        )
    assert alias_root.is_symlink()


def test_recovery_controller_lock_is_existing_and_metadata_read_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "exp23-launch8-two-wave-canary-lock-recovery"
    root.mkdir()
    lock_path = root / ".CANARY_CONTROLLER.lock"
    with pytest.raises(
        controller.CanarySubmissionError,
        match="canary controller lock is unavailable",
    ):
        with controller._CanaryControllerLock(root, create=False):
            pytest.fail("missing recovery lock was acquired")
    assert not os.path.lexists(lock_path)

    lock_path.write_bytes(b"controller-lock-sentinel\n")
    lock_path.chmod(0o644)

    def identity():
        info = lock_path.lstat()
        return {
            "mode": info.st_mode & 0o7777,
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
            "sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        }

    writable_before = identity()
    with pytest.raises(
        controller.CanarySubmissionError,
        match="canary controller lock identity differs",
    ):
        with controller._CanaryControllerLock(root, create=False):
            pytest.fail("writable recovery lock was acquired")
    assert identity() == writable_before

    lock_path.chmod(0o600)
    immutable_before = identity()
    with controller._CanaryControllerLock(root, create=False) as held:
        assert held.binding()["mode"] == 0o600
    assert identity() == immutable_before


@pytest.mark.parametrize(
    "reuse_kind,expected_error",
        (
            ("token", "reuses a historical canary token"),
            (
                "prior-live-job",
                "recovery result lacks recycled-ID cleanup evidence",
            ),
            (
                "residual-chain-job",
                "residual recovery lacks recycled-ID cleanup evidence",
            ),
        (
            "authorization-job",
            "durable prefix reuses a historical canary job ID",
        ),
            (
                "history-job",
                "cancellation history lacks recycled-ID cleanup evidence",
            ),
            (
                "prior-history-job",
                "cancellation history lacks recycled-ID cleanup evidence",
            ),
            (
                "residual-history-job",
                "cancellation history lacks recycled-ID cleanup evidence",
            ),
    ),
)
def test_canary_recovery_rejects_failed_identity_before_scheduler_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reuse_kind: str,
    expected_error: str,
) -> None:
    repo = tmp_path.absolute()
    root = (
        repo
        / controller.CANARY_PARENT_RELATIVE
        / f"exp23-launch8-two-wave-canary-recovery-{reuse_kind}"
    )
    root.mkdir(parents=True, mode=0o700)
    failed_token = "e09ce7d5a0cef1b0"
    fresh_token = "0123456789abcdef"
    token = failed_token if reuse_kind == "token" else fresh_token
    observation = {"schema_version": 1, "mode": "fixture"}
    manifest = {
        "campaign_id": worker.CAMPAIGN_ID,
        "status": "sealed_launch_ready_unsubmitted",
        "formal_validation": False,
        "paths": {"python": sys.executable},
        "execution": {
            "scheduler_control_plane": {},
            "sbatch": "/fixture/sbatch",
            "squeue": "/fixture/squeue",
            "scontrol": "/fixture/scontrol",
            "scancel": "/fixture/scancel",
        },
        "launch_contract": {
            "real_gpu_two_wave_canary": {
                "failed_attempts": [
                    {
                        "state_root": str(
                            repo
                            / controller.CANARY_PARENT_RELATIVE
                            / "exp23-launch8-two-wave-canary-prior"
                        ),
                        "canary_token": failed_token,
                        "job_ids_by_role": {
                            "wave0": ["33285485"],
                            "wave1": ["33285486"],
                            "report": [],
                        },
                    }
                ],
                "accepted_attempts": [],
            }
        },
    }
    protocol_sha256 = "a" * 64
    lock_path = root / ".CANARY_CONTROLLER.lock"
    lock_path.touch(mode=0o600)
    lock_path.chmod(0o600)
    lock_info = lock_path.lstat()
    lock_binding = {
        "path": str(lock_path),
        "device": lock_info.st_dev,
        "inode": lock_info.st_ino,
        "uid": os.getuid(),
        "mode": 0o600,
    }
    source_sha256 = {name: "b" * 64 for name in controller.CANARY_SOURCE_FILES}
    names = {
        role: f"exp23-launch8-canary-{token}-{role}"
        for role in ("wave0", "wave1", "report")
    }
    identity = {
        "schema_version": 1,
        "status": "canary_controller_claimed",
        "campaign_id": worker.CAMPAIGN_ID,
        "scientific": False,
        "state_root": str(root),
        "canary_token": token,
        "job_names": names,
        "scheduler_comment": f"treewm-exp23-canary:{token}",
        "controller_lock": lock_binding,
        "source_sha256": source_sha256,
        "package_protocol_sha256": protocol_sha256,
        "scheduler_control_plane": observation,
        "scheduler_control_plane_sha256": submit.stable_hash(observation),
    }
    prior_path = root / "CANARY_RECOVERY_CANCELLED.json"
    residual_path = root / "CANARY_RECOVERY_RECONCILED_0000.json"
    authorization_path = root / "CANARY_AUTHORIZATION.json"
    if reuse_kind in {
        "prior-job",
        "prior-live-job",
        "residual-chain-job",
        "residual-live-job",
        "prior-history-job",
        "residual-history-job",
    }:
        prior_path.write_text("{}\n", encoding="utf-8")
        prior_path.chmod(0o444)
    if reuse_kind == "residual-history-job":
        residual_path.write_text("{}\n", encoding="utf-8")
        residual_path.chmod(0o444)
    if reuse_kind == "authorization-job":
        authorization_path.write_text("{}\n", encoding="utf-8")
        authorization_path.chmod(0o444)
    fake_campaign = SimpleNamespace(
        load_contract=lambda _repo: (copy.deepcopy(manifest), {}),
        verify_protocol_lock=lambda _package: protocol_sha256,
    )
    monkeypatch.setattr(controller, "_load_submit", lambda _repo: submit)
    monkeypatch.setattr(controller, "_load_campaign", lambda _repo: fake_campaign)
    reads: list[Path] = []

    def guarded_read(path):
        candidate = Path(path)
        reads.append(candidate)
        if candidate.name == "manifest.json":
            return copy.deepcopy(manifest)
        if candidate.name == "CANARY_CONTROLLER_IDENTITY.json":
            return copy.deepcopy(identity)
        if candidate == prior_path and reuse_kind in {
            "prior-job",
            "prior-live-job",
            "residual-chain-job",
            "residual-live-job",
        }:
            return {"fixture": "validated below"}
        if candidate == authorization_path and reuse_kind == "authorization-job":
            return {
                "job_ids": {
                    "wave0": "33285485",
                    "wave1": "8000",
                    "report": "9000",
                }
            }
        pytest.fail(f"unexpected recovery read: {candidate}")

    monkeypatch.setattr(submit, "read_json", guarded_read)

    monkeypatch.setattr(
        submit,
        "file_sha256",
        lambda _path: (
            "b" * 64
            if reuse_kind != "token"
            else pytest.fail("source bytes were read after a denied token")
        ),
    )
    monkeypatch.setattr(submit, "_scheduler_contract", lambda _value: {})
    monkeypatch.setattr(
        submit,
        "_scheduler_fallback_config",
        lambda _value: {"source_control_plane": observation},
    )
    monkeypatch.setattr(submit, "_regular_nonsymlink", lambda *_args: None)
    monkeypatch.setattr(controller.os, "access", lambda *_args: True)
    monkeypatch.setattr(
        controller,
        "_validated_canary_abort",
        lambda *_args, **_kwargs: (
            {
                "wave0": ["33285485"] if reuse_kind == "abort-job" else [],
                "wave1": [],
                "report": [],
            },
            (
                [{"job_ids": ["33285485"]}]
                if reuse_kind.endswith("history-job")
                else []
            ),
        ),
    )
    prior_fixture = {
        "claimed_job_ids": [],
        "live_verified_job_ids": [],
        "live_verified_job_ids_by_role": {
            "wave0": [], "wave1": [], "report": []
        },
        "cancelled_live_job_ids": [],
        "controller_identity_sha256": "c" * 64,
        "durable_prefix_sha256": "d" * 64,
        "cancel_attempt_history": [],
    }
    if reuse_kind == "prior-job":
        prior_fixture["claimed_job_ids"] = ["33285485"]
    if reuse_kind == "prior-live-job":
        prior_fixture["live_verified_job_ids"] = ["33285485"]
        prior_fixture["live_verified_job_ids_by_role"]["wave0"] = ["33285485"]
        prior_fixture["cancelled_live_job_ids"] = ["33285485"]
    if reuse_kind in {
        "prior-job",
        "prior-live-job",
        "residual-chain-job",
        "residual-live-job",
    }:
        monkeypatch.setattr(
            controller,
            "_validated_canary_recovery_result",
            lambda *_args, **_kwargs: copy.deepcopy(prior_fixture),
        )
    if reuse_kind == "residual-chain-job":
        monkeypatch.setattr(
            submit,
            "_validated_residual_recovery_chain",
            lambda *_args, **_kwargs: [
                    {
                        "live_verified_job_ids": ["33285485"],
                        "live_verified_job_ids_by_role": {
                            "wave0": ["33285485"],
                            "wave1": [],
                            "report": [],
                        },
                        "cancelled_live_job_ids": ["33285485"],
                    "cancel_attempt_history": [],
                }
            ],
        )
    elif reuse_kind == "residual-live-job":
        monkeypatch.setattr(
            submit,
            "_validated_residual_recovery_chain",
            lambda *_args, **_kwargs: [],
        )
    if reuse_kind in {"authorization-job", "live-job"}:
        monkeypatch.setattr(
            controller,
            "_load_canary_worker",
            lambda _root: SimpleNamespace(
                _expected_submission_commands=lambda *_args: {}
            ),
        )
    if reuse_kind == "live-job":
        monkeypatch.setattr(
            controller,
            "_validated_canary_postsubmission_prefix",
            lambda *_args, **_kwargs: "d" * 64,
        )
    census_calls: list[str] = []

    def failed_live_census(**_kwargs):
        census_calls.append(reuse_kind)
        return (
            [{"round": index} for index in range(3)],
            {"wave0": ["33285485"], "wave1": [], "report": []},
        )

    if reuse_kind in {"live-job", "residual-live-job"}:
        monkeypatch.setattr(
            submit,
            "_recovery_census_rounds",
            failed_live_census,
        )
    original_flags = controller.sys.flags

    class IsolatedFlags:
        isolated = 1
        no_site = 1
        dont_write_bytecode = 1
        safe_path = 1

        def __getattr__(self, name):
            return getattr(original_flags, name)

    monkeypatch.setattr(controller.sys, "flags", IsolatedFlags())
    monkeypatch.setattr(controller.sys, "dont_write_bytecode", True)
    for key in list(controller.os.environ):
        if key.startswith("TREEWM_") or key in {"RANK", "WORLD_SIZE", "LOCAL_RANK"}:
            monkeypatch.delenv(key)
    for name in list(controller.sys.modules):
        if name == "treewm" or name.startswith("treewm."):
            monkeypatch.delitem(controller.sys.modules, name)
    def state_inventory():
        result = {}
        for path in sorted(root.rglob("*")):
            info = path.lstat()
            payload = path.read_bytes() if path.is_file() else b""
            result[path.relative_to(root).as_posix()] = {
                "mode": info.st_mode & 0o7777,
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "ctime_ns": info.st_ctime_ns,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        return result

    before = state_inventory()
    scheduler_calls: list[list[str]] = []

    def scheduler_runner(command, *_args):
        scheduler_calls.append(list(command))
        pytest.fail("scheduler was called for a failed canary identity")

    with pytest.raises(controller.CanarySubmissionError, match=expected_error):
        controller.recover_or_cancel_real_canary(
            repo,
            root,
            controller.CONFIRMATION,
            scheduler_runner=scheduler_runner,
        )
    after = state_inventory()
    assert scheduler_calls == []
    assert before == after
    expected_reads = [
        repo / controller.PACKAGE_RELATIVE / "manifest.json",
        root / "CANARY_CONTROLLER_IDENTITY.json",
    ]
    if reuse_kind in {
        "prior-job",
        "prior-live-job",
        "residual-chain-job",
        "residual-live-job",
    }:
        expected_reads.append(prior_path)
    if reuse_kind == "authorization-job":
        expected_reads.append(authorization_path)
    assert reads == expected_reads
    assert census_calls == (
        [reuse_kind] if reuse_kind in {"live-job", "residual-live-job"} else []
    )


@pytest.mark.parametrize(
    (
        "duplicate_role,job_ids,failed_job_ids,expected_error,"
        "expected_submit_roles,expected_show_roles,expected_cancel_ids"
    ),
    (
        (
            "wave1",
            {"wave0": "7000", "wave1": "7000", "report": "9000"},
            [],
            "assigned one job ID to multiple roles",
            ["wave0", "wave1"],
            ["wave0"],
            [],
        ),
        (
            "report",
            {"wave0": "7000", "wave1": "8000", "report": "8000"},
            [],
            "assigned one job ID to multiple roles",
            ["wave0", "wave1", "report"],
            ["wave0", "wave1"],
            [],
        ),
        (
            "failed-wave0",
            {"wave0": "33285485", "wave1": "8000", "report": "9000"},
            ["33285485", "33285486"],
            "reused a historical canary job ID",
            ["wave0"],
            [],
            ["33285485"],
        ),
        (
            "accepted-wave0",
            {"wave0": "6000", "wave1": "8000", "report": "9000"},
            [],
            "reused a historical canary job ID",
            ["wave0"],
            [],
            ["6000"],
        ),
        (
            "accepted-wave1",
            {"wave0": "7000", "wave1": "6001", "report": "9000"},
            [],
            "reused a historical canary job ID",
            ["wave0", "wave1"],
            ["wave0"],
            ["6001", "7000"],
        ),
        (
            "accepted-report",
            {"wave0": "7000", "wave1": "8000", "report": "6002"},
            [],
            "reused a historical canary job ID",
            ["wave0", "wave1", "report"],
            ["wave0", "wave1"],
            ["6002", "7000", "8000"],
        ),
    ),
)
def test_canary_duplicate_cross_role_job_id_never_reaches_release_or_scancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    duplicate_role: str,
    job_ids: dict[str, str],
    failed_job_ids: list[str],
    expected_error: str,
    expected_submit_roles: list[str],
    expected_show_roles: list[str],
    expected_cancel_ids: list[str],
) -> None:
    state = (
        tmp_path / f"exp23-launch8-two-wave-canary-duplicate-{duplicate_role}"
    ).absolute()
    observation = {"schema_version": 1, "mode": "fixture"}
    protocol_sha256 = "a" * 64
    manifest = {
        "campaign_id": worker.CAMPAIGN_ID,
        "status": "sealed_launch_ready_unsubmitted",
        "formal_validation": False,
        "paths": {
            "python": sys.executable,
            "run_root": str(tmp_path / "scientific"),
            "transaction_lock": str(tmp_path / "scientific.lock"),
        },
        "execution": {
            "scheduler_control_plane": {},
            "sbatch": "/fixture/sbatch",
            "squeue": "/fixture/squeue",
            "scontrol": "/fixture/scontrol",
            "scancel": "/fixture/scancel",
        },
        "superseded_launches": [],
        "launch_contract": {
            "real_gpu_two_wave_canary": {
                "failed_attempts": (
                    [
                        {
                            "state_root": str(
                                tmp_path
                                / "exp23-launch8-two-wave-canary-old-attempt"
                            ),
                            "canary_token": "e09ce7d5a0cef1b0",
                            "job_ids_by_role": {
                                "wave0": failed_job_ids[:1],
                                "wave1": failed_job_ids[1:],
                                "report": [],
                            },
                        }
                    ]
                    if failed_job_ids
                    else []
                ),
                "accepted_attempts": (
                    []
                    if failed_job_ids
                    else [
                        {
                            "state_root": str(
                                tmp_path
                                / "exp23-launch8-two-wave-canary-accepted-anchor"
                            ),
                            "canary_token": "abcdef0123456789",
                            "job_ids_by_role": {
                                "wave0": ["6000"],
                                "wave1": ["6001"],
                                "report": ["6002"],
                            },
                        }
                    ]
                ),
            }
        },
    }
    fake_campaign = SimpleNamespace(
        load_contract=lambda _repo: (copy.deepcopy(manifest), {}),
        verify_protocol_lock=lambda _package: protocol_sha256,
    )
    monkeypatch.setattr(controller, "_load_submit", lambda _repo: submit)
    monkeypatch.setattr(controller, "_load_campaign", lambda _repo: fake_campaign)
    original_read_json = submit.read_json
    monkeypatch.setattr(
        submit,
        "read_json",
        lambda path: (
            copy.deepcopy(manifest)
            if Path(path).name == "manifest.json"
            else original_read_json(path)
        ),
    )
    monkeypatch.setattr(submit, "_scheduler_contract", lambda _value: {})
    monkeypatch.setattr(
        submit,
        "_scheduler_fallback_config",
        lambda _value: {"source_control_plane": observation},
    )
    monkeypatch.setattr(
        submit, "_scheduler_control_plane_observation", lambda _value: observation
    )
    monkeypatch.setattr(submit, "_scheduler_environment", lambda _value: {})
    monkeypatch.setattr(submit, "_regular_nonsymlink", lambda *_args: None)
    monkeypatch.setattr(controller.os, "access", lambda *_args: True)
    monkeypatch.setattr(controller.secrets, "token_hex", lambda _size: "0123456789abcdef")

    def prepare(
        _repo: Path,
        root: Path,
        _manifest: dict,
        *,
        hold_controller_lock: bool = False,
    ):
        root.mkdir(mode=0o700)
        created = root.lstat()
        lock = controller._CanaryControllerLock(root, create=True)
        lock.__enter__()
        (root / "logs").mkdir(mode=0o700)
        source = root / "source"
        source.mkdir(mode=0o700)
        hashes = {}
        for name in controller.CANARY_SOURCE_FILES:
            payload = (PACKAGE / name).read_bytes()
            destination = source / name
            destination.write_bytes(payload)
            destination.chmod(0o444)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        source.chmod(0o555)
        if hold_controller_lock:
            return root, hashes, lock, (created.st_dev, created.st_ino)
        lock.__exit__(None, None, None)
        return root, hashes

    monkeypatch.setattr(controller, "_prepare_state", prepare)
    original_flags = controller.sys.flags

    class IsolatedFlags:
        isolated = 1
        no_site = 1
        dont_write_bytecode = 1
        safe_path = 1

        def __getattr__(self, name):
            return getattr(original_flags, name)

    monkeypatch.setattr(controller.sys, "flags", IsolatedFlags())
    monkeypatch.setattr(controller.sys, "dont_write_bytecode", True)
    for key in list(controller.os.environ):
        if key.startswith("TREEWM_") or key in {"RANK", "WORLD_SIZE", "LOCAL_RANK"}:
            monkeypatch.delenv(key)
    for name in list(controller.sys.modules):
        if name == "treewm" or name.startswith("treewm."):
            monkeypatch.delitem(controller.sys.modules, name)

    calls: list[list[str]] = []
    active_by_name: dict[str, str] = {}
    shown_roles: list[str] = []
    expected_user = submit.pwd.getpwuid(os.getuid()).pw_name
    comment = "treewm-exp23-canary:0123456789abcdef"
    names = {
        role: f"exp23-launch8-canary-0123456789abcdef-{role}"
        for role in ("wave0", "wave1", "report")
    }

    def runner(command, _cwd, _environment, inherited_fds):
        values = list(command)
        calls.append(values)
        assert inherited_fds
        executable = Path(values[0]).name
        if executable == "squeue":
            name = next(
                item.split("=", 1)[1]
                for item in values
                if item.startswith("--name=")
            )
            job_id = active_by_name.get(name)
            stdout = (
                f"{job_id}|{name}|{expected_user}|PENDING|{comment}\n"
                if job_id is not None
                else ""
            )
            return subprocess.CompletedProcess(values, 0, stdout=stdout, stderr="")
        if executable == "sbatch":
            name = next(
                item.split("=", 1)[1]
                for item in values
                if item.startswith("--job-name=")
            )
            role = next(role for role, expected in names.items() if name == expected)
            active_by_name[name] = job_ids[role]
            return subprocess.CompletedProcess(
                values, 0, stdout=f"{job_ids[role]}\n", stderr=""
            )
        if executable == "scontrol":
            assert values[1:3] == ["show", "job"]
            role = ("wave0", "wave1", "report")[len(shown_roles)]
            shown_roles.append(role)
            if role == "wave0":
                suffix = "JobState=PENDING Reason=JobHeldUser"
            else:
                predecessor = "wave0" if role == "wave1" else "wave1"
                suffix = (
                    "JobState=PENDING "
                    f"Dependency=afterok:{job_ids[predecessor]}(unfulfilled) "
                    "KillOInInvalidDependent=Yes"
                )
            return subprocess.CompletedProcess(
                values,
                0,
                stdout=(
                    f"JobId={job_ids[role]} JobName={names[role]} "
                    f"Comment={comment} {suffix}\n"
                ),
                stderr="",
            )
        if executable == "scancel":
            assert values[1:] == expected_cancel_ids
            for name, active_id in list(active_by_name.items()):
                if active_id in expected_cancel_ids:
                    del active_by_name[name]
            return subprocess.CompletedProcess(values, 0, stdout="", stderr="")
        pytest.fail(f"ambiguous duplicate ID reached scheduler mutation: {values}")

    with pytest.raises(
        (submit.SubmissionError, controller.CanarySubmissionError)
    ) as captured:
        controller.submit_real_canary(
            REPO,
            state,
            controller.CONFIRMATION,
            scheduler_runner=runner,
        )
    assert expected_error in str(captured.value), (
        repr(captured.value.__context__),
        repr(captured.value.__cause__),
    )
    submitted = [row for row in calls if Path(row[0]).name == "sbatch"]
    submitted_roles = [
        next(
            role
            for role in ("wave0", "wave1", "report")
            if any(item == f"--job-name={names[role]}" for item in row)
        )
        for row in submitted
    ]
    assert submitted_roles == expected_submit_roles
    assert shown_roles == expected_show_roles
    assert not any(
        Path(row[0]).name == "scontrol" and row[1:2] == ["release"]
        for row in calls
    )
    scancel_calls = [row[1:] for row in calls if Path(row[0]).name == "scancel"]
    assert scancel_calls == ([expected_cancel_ids] if expected_cancel_ids else [])
    assert not os.path.lexists(state / "CANARY_AUTHORIZATION.json")
    assert not os.path.lexists(state / "CANARY_SUBMISSION_RECEIPT.json")
    assert not os.path.lexists(state / "CANARY_READY_TO_RELEASE.json")
    assert not os.path.lexists(state / "CANARY_WAVE0_RELEASED.json")
    historical_ids = set(failed_job_ids) or {"6000", "6001", "6002"}
    recycled_ids = sorted(
        historical_ids.intersection(expected_cancel_ids), key=int
    )
    if recycled_ids:
        marker_path = state / "CANARY_HISTORICAL_ID_RECYCLED_0000.json"
        marker = worker.read_json(marker_path, "historical-ID cleanup marker")
        assert marker["phase"] == "initial_submit_abort"
        assert marker["historical_recycled_job_ids"] == recycled_ids
        assert marker["authorization_allowed"] is False
        assert marker["release_allowed"] is False
        assert marker["resume_allowed"] is False
        assert marker["result_allowed"] is False
        abort = worker.read_json(state / "CANARY_ABORTED.json", "canary abort")
        assert abort["historical_numeric_id_recycled"] is True
        assert abort["historical_recycled_job_ids"] == recycled_ids
        assert abort["historical_recycled_evidence_sha256"] == [
            {
                "name": marker_path.name,
                "sha256": hashlib.sha256(marker_path.read_bytes()).hexdigest(),
            }
        ]
        cancel_calling = worker.read_json(
            state / "CANARY_RECOVERY_CANCEL_CALLING_0000.json",
            "historical-ID cancel calling",
        )
        assert cancel_calling["historical_recycled_evidence"] == {
            "name": marker_path.name,
            "sha256": hashlib.sha256(marker_path.read_bytes()).hexdigest(),
        }
        assert marker["cancel_history_length_before"] == 0
    else:
        assert not os.path.lexists(
            state / "CANARY_HISTORICAL_ID_RECYCLED_0000.json"
        )


@pytest.mark.parametrize("settled", (True, False))
def test_calling_only_recycled_id_recovery_requires_settled_cleanup_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settled: bool,
) -> None:
    for name in list(sys.modules):
        if name == "treewm" or name.startswith("treewm."):
            monkeypatch.delitem(sys.modules, name)
    repo = (tmp_path / "repo").absolute()
    package = repo / controller.PACKAGE_RELATIVE
    package.mkdir(parents=True)
    executables: dict[str, str] = {}
    for name in ("sbatch", "scontrol", "squeue", "scancel"):
        path = tmp_path / name
        path.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
        path.chmod(0o755)
        executables[name] = str(path)
    historical_root = (
        repo
        / controller.CANARY_PARENT_RELATIVE
        / "exp23-launch8-two-wave-canary-accepted-anchor"
    )
    root = (
        repo
        / controller.CANARY_PARENT_RELATIVE
        / f"exp23-launch8-two-wave-canary-recycled-{'settled' if settled else 'drift'}"
    )
    root.mkdir(parents=True, mode=0o700)
    source = root / "source"
    source.mkdir(mode=0o700)
    source_sha256 = {}
    for name in controller.CANARY_SOURCE_FILES:
        payload = (PACKAGE / name).read_bytes()
        (package / name).write_bytes(payload)
        destination = source / name
        destination.write_bytes(payload)
        destination.chmod(0o444)
        source_sha256[name] = hashlib.sha256(payload).hexdigest()
    source.chmod(0o555)
    manifest = {
        "campaign_id": worker.CAMPAIGN_ID,
        "status": "sealed_launch_ready_unsubmitted",
        "formal_validation": False,
        "paths": {"python": sys.executable},
        "execution": {
            **executables,
            "scheduler_control_plane": {},
        },
        "launch_contract": {
            "real_gpu_two_wave_canary": {
                "failed_attempts": [],
                "accepted_attempts": [
                    {
                        "state_root": str(historical_root),
                        "canary_token": "abcdef0123456789",
                        "job_ids_by_role": {
                            "wave0": ["6000"],
                            "wave1": ["6001"],
                            "report": ["6002"],
                        },
                    }
                ],
            }
        },
    }
    (package / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    lock_path = root / ".CANARY_CONTROLLER.lock"
    lock_path.touch(mode=0o600)
    lock_path.chmod(0o600)
    lock_info = lock_path.lstat()
    lock_binding = {
        "path": str(lock_path),
        "device": lock_info.st_dev,
        "inode": lock_info.st_ino,
        "uid": os.getuid(),
        "mode": 0o600,
    }
    token = "0123456789abcdef"
    observation = {"schema_version": 1, "mode": "fixture"}
    names = {
        role: f"exp23-launch8-canary-{token}-{role}"
        for role in ("wave0", "wave1", "report")
    }
    identity = {
        "schema_version": 1,
        "status": "canary_controller_claimed",
        "campaign_id": worker.CAMPAIGN_ID,
        "scientific": False,
        "state_root": str(root),
        "canary_token": token,
        "job_names": names,
        "scheduler_comment": f"treewm-exp23-canary:{token}",
        "controller_lock": lock_binding,
        "source_sha256": source_sha256,
        "package_protocol_sha256": "a" * 64,
        "scheduler_control_plane": observation,
        "scheduler_control_plane_sha256": submit.stable_hash(observation),
    }
    worker.seal_json(root / "CANARY_CONTROLLER_IDENTITY.json", identity)
    calling_authorization = {
        "canary_token": token,
        "job_ids": {"wave0": "6000", "wave1": "8000", "report": "9000"},
        "job_names": names,
        "scheduler_comment": identity["scheduler_comment"],
        "scheduler_executables": {
            "submit": executables["sbatch"],
            "control": executables["scontrol"],
        },
        "scheduler_control_plane": observation,
        "scheduler_control_plane_sha256": submit.stable_hash(observation),
        "controller_lock": lock_binding,
    }
    worker.seal_json(
        root / "CANARY_WAVE0_CALLING.json",
        _calling_payload(root, calling_authorization, "wave0"),
    )
    fake_campaign = SimpleNamespace(
        load_contract=lambda _repo: (copy.deepcopy(manifest), {}),
        verify_protocol_lock=lambda _package: "a" * 64,
    )
    monkeypatch.setattr(controller, "_load_submit", lambda _repo: submit)
    monkeypatch.setattr(controller, "_load_campaign", lambda _repo: fake_campaign)
    monkeypatch.setattr(submit, "_scheduler_contract", lambda _value: {})
    monkeypatch.setattr(
        submit,
        "_scheduler_fallback_config",
        lambda _value: {"source_control_plane": observation},
    )
    monkeypatch.setattr(
        submit, "_scheduler_control_plane_observation", lambda _value: observation
    )
    monkeypatch.setattr(submit, "_scheduler_environment", lambda _value: {})
    monkeypatch.setattr(submit.time, "sleep", lambda _seconds: None)
    original_flags = controller.sys.flags

    class IsolatedFlags:
        isolated = 1
        no_site = 1
        dont_write_bytecode = 1
        safe_path = 1

        def __getattr__(self, name):
            return getattr(original_flags, name)

    monkeypatch.setattr(controller.sys, "flags", IsolatedFlags())
    monkeypatch.setattr(controller.sys, "dont_write_bytecode", True)
    for key in list(controller.os.environ):
        if key.startswith("TREEWM_") or key in {"RANK", "WORLD_SIZE", "LOCAL_RANK"}:
            monkeypatch.delenv(key)
    expected_user = submit.pwd.getpwuid(os.getuid()).pw_name
    calls: list[list[str]] = []
    wave0_queries = 0
    active = True

    def runner(command, _cwd, _environment, inherited_fds):
        nonlocal wave0_queries, active
        values = list(command)
        calls.append(values)
        assert inherited_fds
        if Path(values[0]).name == "squeue":
            name = next(
                item.split("=", 1)[1]
                for item in values
                if item.startswith("--name=")
            )
            if name == names["wave0"]:
                wave0_queries += 1
            visible = active and name == names["wave0"] and (
                settled or wave0_queries in {1, 3}
            )
            stdout = (
                f"6000|{name}|{expected_user}|PENDING|{identity['scheduler_comment']}\n"
                if visible
                else ""
            )
            return subprocess.CompletedProcess(values, 0, stdout=stdout, stderr="")
        assert Path(values[0]).name == "scancel"
        assert values[1:] == ["6000"]
        active = False
        return subprocess.CompletedProcess(values, 0, stdout="", stderr="")

    if not settled:
        with pytest.raises(
            submit.SubmissionError,
            match="did not settle",
        ):
            controller.recover_or_cancel_real_canary(
                repo,
                root,
                controller.CONFIRMATION,
                scheduler_runner=runner,
            )
        assert not any(Path(row[0]).name == "scancel" for row in calls)
        assert not os.path.lexists(
            root / "CANARY_HISTORICAL_ID_RECYCLED_0000.json"
        )
        assert not os.path.lexists(root / "CANARY_RECOVERY_CANCELLED.json")
        return

    original_append = submit._append_recovery_cancel_attempt

    def crash_after_marker(*_args, **_kwargs):
        raise submit.SubmissionError("fixture crash after recycled-ID marker")

    monkeypatch.setattr(
        submit, "_append_recovery_cancel_attempt", crash_after_marker
    )
    with pytest.raises(
        submit.SubmissionError,
        match="fixture crash after recycled-ID marker",
    ):
        controller.recover_or_cancel_real_canary(
            repo,
            root,
            controller.CONFIRMATION,
            scheduler_runner=runner,
        )
    first_marker_path = root / "CANARY_HISTORICAL_ID_RECYCLED_0000.json"
    assert os.path.lexists(first_marker_path)
    assert not os.path.lexists(root / "CANARY_RECOVERY_CANCEL_CALLING_0000.json")
    monkeypatch.setattr(
        submit, "_append_recovery_cancel_attempt", original_append
    )
    original_exclusive = submit.exclusive_json

    def crash_before_terminal(path, value, *args, **kwargs):
        if Path(path).name == "CANARY_RECOVERY_CANCELLED.json":
            raise submit.SubmissionError(
                "fixture crash after recycled-ID cancel result"
            )
        return original_exclusive(path, value, *args, **kwargs)

    monkeypatch.setattr(submit, "exclusive_json", crash_before_terminal)
    with pytest.raises(
        submit.SubmissionError,
        match="fixture crash after recycled-ID cancel result",
    ):
        controller.recover_or_cancel_real_canary(
            repo,
            root,
            controller.CONFIRMATION,
            scheduler_runner=runner,
        )
    assert os.path.lexists(root / "CANARY_RECOVERY_CANCEL_CALLING_0000.json")
    assert os.path.lexists(root / "CANARY_RECOVERY_CANCEL_RESULT_0000.json")
    assert not os.path.lexists(root / "CANARY_RECOVERY_CANCELLED.json")
    monkeypatch.setattr(submit, "exclusive_json", original_exclusive)
    result = controller.recover_or_cancel_real_canary(
        repo,
        root,
        controller.CONFIRMATION,
        scheduler_runner=runner,
    )
    assert result["status"] == "canary_recovered_terminal_after_cancel_attempts"
    assert result["claimed_job_ids"] == []
    assert result["live_verified_job_ids"] == []
    assert result["historical_numeric_id_recycled"] is True
    assert result["historical_recycled_job_ids"] == ["6000"]
    assert result["historical_recycled_job_ids_by_role"] == {
        "wave0": ["6000"], "wave1": [], "report": []
    }
    assert [Path(row[0]).name for row in calls].count("scancel") == 1
    marker_path = root / "CANARY_HISTORICAL_ID_RECYCLED_0001.json"
    marker = worker.read_json(
        marker_path,
        "recycled cleanup marker",
    )
    assert marker["phase"] == "first_recovery"
    assert marker["supersedes_unconsumed_evidence"] == {
        "name": first_marker_path.name,
        "sha256": hashlib.sha256(first_marker_path.read_bytes()).hexdigest(),
    }
    assert marker["historical_recycled_job_ids_by_role"] == {
        "wave0": ["6000"], "wave1": [], "report": []
    }
    calling = worker.read_json(
        root / "CANARY_RECOVERY_CANCEL_CALLING_0000.json",
        "recycled cleanup calling",
    )
    assert calling["historical_recycled_evidence"] == {
        "name": marker_path.name,
        "sha256": hashlib.sha256(marker_path.read_bytes()).hexdigest(),
    }
    # Coherently rehashing the entire cancel-result/terminal suffix cannot
    # detach the pre-cancel recycled-ID marker from the mutation intent.
    calling_path = root / "CANARY_RECOVERY_CANCEL_CALLING_0000.json"
    calling.pop("historical_recycled_evidence")
    _rewrite_sealed_json(calling_path, calling)
    calling_sha256 = hashlib.sha256(calling_path.read_bytes()).hexdigest()
    cancel_result_path = root / "CANARY_RECOVERY_CANCEL_RESULT_0000.json"
    cancel_result = worker.read_json(cancel_result_path, "cancel result")
    cancel_result["calling_sha256"] = calling_sha256
    _rewrite_sealed_json(cancel_result_path, cancel_result)
    result_sha256 = hashlib.sha256(cancel_result_path.read_bytes()).hexdigest()
    terminal_path = root / "CANARY_RECOVERY_CANCELLED.json"
    terminal = worker.read_json(terminal_path, "recovery terminal")
    terminal["cancel_calling_sha256"] = calling_sha256
    terminal["cancel_attempt_history"][0]["calling_sha256"] = calling_sha256
    terminal["cancel_attempt_history"][0]["result_sha256"] = result_sha256
    _rewrite_sealed_json(terminal_path, terminal)
    calls_before = list(calls)
    with pytest.raises(
        controller.CanarySubmissionError,
        match="recycled-ID fields differ",
    ):
        controller.recover_or_cancel_real_canary(
            repo,
            root,
            controller.CONFIRMATION,
            scheduler_runner=runner,
        )
    assert calls == calls_before


def test_canary_worker_authenticates_exact_graph_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "state").absolute()
    root.mkdir(mode=0o700)
    authorization = _authorization(root)
    _seal_controller_contract(root, authorization)
    monkeypatch.setattr(worker, "__file__", str(root / "source/canary_worker.py"))
    assert worker._authorization(root) == authorization

    checkpoint = root / "wave0_checkpoint.pt"
    checkpoint.write_bytes(b"sealed checkpoint fixture")
    checkpoint.chmod(0o444)
    ready = {
        "schema_version": 1,
        "status": "wave0_ready",
        "campaign_id": worker.CAMPAIGN_ID,
        "canary_token": authorization["canary_token"],
        "wave0_job_id": "7000",
        "checkpoint_sha256": worker.sha256_file(checkpoint),
        "expected_resumed_result": 123.0,
        "cuda_device_name": "fixture",
        "within_wave_requeue": False,
    }
    worker.seal_json(root / worker.READY_NAME, ready)
    worker.seal_json(
        root / worker.COMPLETE_NAME,
        {
            "schema_version": 1,
            "status": "wave1_complete",
            "campaign_id": worker.CAMPAIGN_ID,
            "canary_token": authorization["canary_token"],
            "wave0_job_id": "7000",
            "wave1_job_id": "8000",
            "ready_sha256": worker.sha256_file(root / worker.READY_NAME),
            "checkpoint_sha256": worker.sha256_file(checkpoint),
            "resumed_result": 123.0,
            "within_wave_requeue": False,
        },
    )
    monkeypatch.setenv("SLURM_JOB_ID", "9000")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    assert worker._report(root, authorization) == 0
    report = worker.read_json(root / worker.REPORT_NAME, "canary report")
    assert report["status"] == "two_wave_gpu_canary_passed"
    assert report["job_ids"] == authorization["job_ids"]
    assert report["authorization_sha256"] == worker.sha256_file(
        root / worker.AUTH_NAME
    )
    assert report["receipt_sha256"] == worker.sha256_file(root / worker.RECEIPT_NAME)
    assert report["ready_to_release_sha256"] == worker.sha256_file(
        root / worker.READY_TO_RELEASE_NAME
    )
    assert report["wave0_release_sha256"] == worker.sha256_file(
        root / "CANARY_WAVE0_RELEASED.json"
    )
    assert report["wave0_ready_sha256"] == worker.sha256_file(
        root / worker.READY_NAME
    )
    assert report["checkpoint_sha256"] == worker.sha256_file(checkpoint)
    assert report["accepted_submission_records_sha256"] == authorization[
        "accepted_submission_records_sha256"
    ]
    assert report["accepted_scheduler_evidence_sha256"] == authorization[
        "accepted_scheduler_evidence_sha256"
    ]


def test_canary_authorization_rejects_dependency_and_worker_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for mutation, pattern in (("dependency", "graph"), ("worker", "source bytes")):
        root = (tmp_path / mutation).absolute()
        root.mkdir(mode=0o700)
        authorization = _authorization(root)
        if mutation == "dependency":
            authorization["dependencies"]["wave1"] = "afterok:9999"
        else:
            authorization["worker_sha256"] = "f" * 64
            authorization["source_sha256"]["canary_worker.py"] = "f" * 64
        _seal_controller_contract(root, authorization)
        monkeypatch.setattr(worker, "__file__", str(root / "source/canary_worker.py"))
        with pytest.raises(worker.CanaryError, match=pattern):
            worker._authorization(root)


@pytest.mark.parametrize(
    "mutation",
    (
        "held-state",
        "dependent-policy",
        "submission-command",
        "control-plane",
        "control-plane-bool",
    ),
)
def test_canary_authorization_rejects_rehashed_forged_scheduler_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = (tmp_path / mutation).absolute()
    root.mkdir(mode=0o700)
    authorization = _authorization(root)
    records, evidence = _accepted_scheduler_fixtures(root, authorization)
    records = copy.deepcopy(records)
    evidence = copy.deepcopy(evidence)
    if mutation == "held-state":
        evidence["wave0_accepted_hold"]["reason"] = "None"
    elif mutation == "dependent-policy":
        evidence["wave1_accepted_dependency"]["kill_on_invalid_dependency"] = "No"
    elif mutation == "submission-command":
        records["wave0"]["command"].remove("--hold")
    elif mutation == "control-plane":
        records["report"]["scheduler_control_plane"]["mode"] = "forged"
    else:
        coerced = {"schema_version": True, "mode": "fixture"}
        for record in records.values():
            record["scheduler_control_plane"] = coerced
        for row in evidence.values():
            row["scheduler_control_plane"] = coerced
        authorization["scheduler_control_plane"] = coerced
        authorization["scheduler_control_plane_sha256"] = worker.stable_hash(
            coerced
        )
    authorization["accepted_submission_records_sha256"] = worker.stable_hash(records)
    authorization["accepted_scheduler_evidence_sha256"] = worker.stable_hash(evidence)
    _seal_controller_contract(
        root, authorization, records=records, evidence=evidence
    )
    monkeypatch.setattr(worker, "__file__", str(root / "source/canary_worker.py"))
    with pytest.raises(worker.CanaryError, match="canary"):
        worker._authorization(root)


@pytest.mark.parametrize("role", ("wave1", "report"))
def test_canary_authorization_rejects_array_dependency_for_scalar_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    root = (tmp_path / f"array-predecessor-{role}").absolute()
    root.mkdir(mode=0o700)
    authorization = _authorization(root)
    records, evidence = _accepted_scheduler_fixtures(root, authorization)
    records = copy.deepcopy(records)
    evidence = copy.deepcopy(evidence)
    predecessor = "wave0" if role == "wave1" else "wave1"
    scalar = f"afterok:{authorization['job_ids'][predecessor]}(unfulfilled)"
    array = f"afterok:{authorization['job_ids'][predecessor]}_*(unfulfilled)"
    row = evidence[f"{role}_accepted_dependency"]
    assert row["dependency"] == scalar
    row["dependency"] = array
    row["stdout"] = row["stdout"].replace(scalar, array)
    authorization["accepted_submission_records_sha256"] = worker.stable_hash(records)
    authorization["accepted_scheduler_evidence_sha256"] = worker.stable_hash(evidence)
    _seal_controller_contract(
        root, authorization, records=records, evidence=evidence
    )
    monkeypatch.setattr(worker, "__file__", str(root / "source/canary_worker.py"))
    with pytest.raises(worker.CanaryError, match="accepted dependency differs"):
        worker._authorization(root)


def test_canary_report_rejects_rehashed_forged_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "forged-release").absolute()
    root.mkdir(mode=0o700)
    authorization = _authorization(root)
    _seal_controller_contract(root, authorization)
    monkeypatch.setattr(worker, "__file__", str(root / "source/canary_worker.py"))
    release_path = root / "CANARY_WAVE0_RELEASED.json"
    release = worker.read_json(release_path, "fixture release")
    release_path.unlink()
    release["wave0_release"]["release_command"][-1] = "9999"
    worker.seal_json(release_path, release)
    monkeypatch.setenv("SLURM_JOB_ID", "9000")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    with pytest.raises(worker.CanaryError, match="released wave-zero scheduler identity"):
        worker._report(root, authorization)
    assert not os.path.lexists(root / worker.REPORT_NAME)


@pytest.mark.parametrize(
    "filename,mutation,pattern",
    [
        (
            "CANARY_AUTHORIZATION.json",
            lambda value: {
                **value,
                "job_names": {**value["job_names"], "wave0": "forged"},
            },
            "authorization durable semantics",
        ),
        (
            "CANARY_SUBMISSION_RECEIPT.json",
            lambda value: {**value, "dependencies": {**value["dependencies"], "wave1": "afterok:1"}},
            "durable receipt",
        ),
        (
            "CANARY_READY_TO_RELEASE.json",
            lambda value: {**value, "receipt_sha256": "f" * 64},
            "ready-to-release",
        ),
        (
            "CANARY_WAVE0_RELEASE_CALLING.json",
            lambda value: {**value, "command": ["/fixture/bin/scontrol", "release", "1"]},
            "release intent",
        ),
        (
            "CANARY_WAVE0_RELEASED.json",
            lambda value: {
                **value,
                "wave0_release": {
                    **value["wave0_release"],
                    "release_command": ["/fixture/bin/scontrol", "release", "1"],
                },
            },
            "released wave-zero scheduler identity",
        ),
    ],
)
def test_canary_recovery_validates_every_durable_postsubmission_prefix(
    tmp_path: Path, filename: str, mutation, pattern: str
) -> None:
    root = (tmp_path / filename.replace(".json", "").lower()).absolute()
    root.mkdir(mode=0o700)
    authorization = _authorization(root)
    _seal_controller_contract(root, authorization)
    identity = worker.read_json(root / "CANARY_CONTROLLER_IDENTITY.json", "identity")
    prefix = controller._validated_canary_postsubmission_prefix(
        submit,
        worker,
        root,
        identity=identity,
        protocol_sha256=authorization["package_protocol_sha256"],
        scheduler_executables=authorization["scheduler_executables"],
        scheduler_control_plane_sha256=authorization[
            "scheduler_control_plane_sha256"
        ],
    )
    assert all(prefix.values())
    path = root / filename
    value = worker.read_json(path, filename)
    path.chmod(0o600)
    path.write_text(
        json.dumps(mutation(value), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    with pytest.raises((controller.CanarySubmissionError, worker.CanaryError), match=pattern):
        controller._validated_canary_postsubmission_prefix(
            submit,
            worker,
            root,
            identity=identity,
            protocol_sha256=authorization["package_protocol_sha256"],
            scheduler_executables=authorization["scheduler_executables"],
            scheduler_control_plane_sha256=authorization[
                "scheduler_control_plane_sha256"
            ],
        )


def test_canary_recovery_accepts_auth_only_prefix_and_rejects_gap(tmp_path: Path) -> None:
    root = (tmp_path / "auth-only").absolute()
    root.mkdir(mode=0o700)
    authorization = _authorization(root)
    _seal_controller_contract(root, authorization)
    identity = worker.read_json(root / "CANARY_CONTROLLER_IDENTITY.json", "identity")
    for filename in (
        "CANARY_SUBMISSION_RECEIPT.json",
        "CANARY_READY_TO_RELEASE.json",
        "CANARY_WAVE0_RELEASE_CALLING.json",
        "CANARY_WAVE0_RELEASED.json",
    ):
        (root / filename).unlink()
    prefix = controller._validated_canary_postsubmission_prefix(
        submit,
        worker,
        root,
        identity=identity,
        protocol_sha256=authorization["package_protocol_sha256"],
        scheduler_executables=authorization["scheduler_executables"],
        scheduler_control_plane_sha256=authorization[
            "scheduler_control_plane_sha256"
        ],
    )
    assert prefix["authorization"]
    assert all(
        prefix[key] is None
        for key in ("receipt", "ready_to_release", "release_calling", "released")
    )
    assert prefix["submitted_job_ids_by_role"] == {
        "wave0": ["7000"],
        "wave1": ["8000"],
        "report": ["9000"],
    }
    assert set(prefix["calling_intent_sha256_by_role"]) == {
        "wave0",
        "wave1",
        "report",
    }
    worker.seal_json(root / "CANARY_READY_TO_RELEASE.json", {"forged": True})
    with pytest.raises(controller.CanarySubmissionError, match="prefix has a gap"):
        controller._validated_canary_postsubmission_prefix(
            submit,
            worker,
            root,
            identity=identity,
            protocol_sha256=authorization["package_protocol_sha256"],
            scheduler_executables=authorization["scheduler_executables"],
            scheduler_control_plane_sha256=authorization[
                "scheduler_control_plane_sha256"
            ],
        )


def test_canary_recovery_semantically_validates_submitted_prefix_before_auth(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "submitted-only").absolute()
    root.mkdir(mode=0o700)
    authorization = _authorization(root)
    _seal_controller_contract(root, authorization)
    identity = worker.read_json(root / "CANARY_CONTROLLER_IDENTITY.json", "identity")
    for filename in (
        "CANARY_AUTHORIZATION.json",
        "CANARY_SUBMISSION_RECEIPT.json",
        "CANARY_READY_TO_RELEASE.json",
        "CANARY_WAVE0_RELEASE_CALLING.json",
        "CANARY_WAVE0_RELEASED.json",
    ):
        (root / filename).unlink()
    prefix = controller._validated_canary_postsubmission_prefix(
        submit,
        worker,
        root,
        identity=identity,
        protocol_sha256=authorization["package_protocol_sha256"],
        scheduler_executables=authorization["scheduler_executables"],
        scheduler_control_plane_sha256=authorization[
            "scheduler_control_plane_sha256"
        ],
    )
    assert all(
        prefix[key] is None
        for key in ("authorization", "receipt", "ready_to_release", "release_calling", "released")
    )
    assert prefix["submitted_job_ids_by_role"] == {
        "wave0": ["7000"],
        "wave1": ["8000"],
        "report": ["9000"],
    }
    assert set(prefix["calling_intent_sha256_by_role"]) == {
        "wave0",
        "wave1",
        "report",
    }
    path = root / "CANARY_WAVE0_SUBMITTED.json"
    value = worker.read_json(path, "wave0 submitted")
    value["accepted_submission_record"]["command"][0] = "/forged/sbatch"
    path.chmod(0o600)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    path.chmod(0o444)
    with pytest.raises(worker.CanaryError, match="submission record identity"):
        controller._validated_canary_postsubmission_prefix(
            submit,
            worker,
            root,
            identity=identity,
            protocol_sha256=authorization["package_protocol_sha256"],
            scheduler_executables=authorization["scheduler_executables"],
            scheduler_control_plane_sha256=authorization[
                "scheduler_control_plane_sha256"
            ],
        )


@pytest.mark.parametrize(
    ("missing", "pattern"),
    (
        ("CANARY_WAVE1_SUBMITTED.json", "submitted journals are not a contiguous prefix"),
        ("CANARY_WAVE1_CALLING.json", "calling records are not a contiguous prefix"),
    ),
)
def test_canary_postsubmission_validator_rejects_gapped_role_prefix(
    tmp_path: Path, missing: str, pattern: str
) -> None:
    root = (tmp_path / missing).absolute()
    root.mkdir(mode=0o700)
    authorization = _authorization(root)
    _seal_controller_contract(root, authorization)
    identity = worker.read_json(root / "CANARY_CONTROLLER_IDENTITY.json", "identity")
    (root / missing).unlink()
    with pytest.raises(controller.CanarySubmissionError, match=pattern):
        controller._validated_canary_postsubmission_prefix(
            submit,
            worker,
            root,
            identity=identity,
            protocol_sha256=authorization["package_protocol_sha256"],
            scheduler_executables=authorization["scheduler_executables"],
            scheduler_control_plane_sha256=authorization[
                "scheduler_control_plane_sha256"
            ],
        )


@pytest.mark.parametrize("bad_id", (7000, True))
@pytest.mark.parametrize(
    "surface",
    (
        "submitted_job_id",
        "submitted_reconciled_id",
        "authorization_job_id",
        "receipt_job_id",
        "ready_job_id",
        "release_calling_job_id",
        "released_job_id",
    ),
)
def test_canary_durable_graph_rejects_non_string_job_ids_without_writes(
    tmp_path: Path, surface: str, bad_id: object
) -> None:
    root = (tmp_path / f"{surface}-{bad_id!r}").absolute()
    root.mkdir(mode=0o700)
    authorization = _authorization(root)
    _seal_controller_contract(root, authorization)
    identity = worker.read_json(root / "CANARY_CONTROLLER_IDENTITY.json", "identity")
    targets = {
        "submitted_job_id": root / "CANARY_WAVE0_SUBMITTED.json",
        "submitted_reconciled_id": root / "CANARY_WAVE0_SUBMITTED.json",
        "authorization_job_id": root / worker.AUTH_NAME,
        "receipt_job_id": root / worker.RECEIPT_NAME,
        "ready_job_id": root / worker.READY_TO_RELEASE_NAME,
        "release_calling_job_id": root / "CANARY_WAVE0_RELEASE_CALLING.json",
        "released_job_id": root / "CANARY_WAVE0_RELEASED.json",
    }
    target = targets[surface]
    value = worker.read_json(target, f"{surface} fixture")
    if surface == "submitted_job_id":
        value["job_id"] = bad_id
    elif surface == "submitted_reconciled_id":
        value["accepted_submission_record"]["reconciled_job_ids"] = [bad_id]
    elif surface in {"authorization_job_id", "receipt_job_id", "ready_job_id"}:
        value["job_ids"]["wave0"] = bad_id
    elif surface == "release_calling_job_id":
        value["command"][-1] = bad_id
    else:
        value["wave0_job_id"] = bad_id
    _rewrite_sealed_json(target, value)
    before = {
        entry.name: hashlib.sha256(entry.read_bytes()).hexdigest()
        for entry in root.iterdir()
        if entry.is_file()
    }
    with pytest.raises(RuntimeError):
        controller._validated_canary_postsubmission_prefix(
            submit,
            worker,
            root,
            identity=identity,
            protocol_sha256=authorization["package_protocol_sha256"],
            scheduler_executables=authorization["scheduler_executables"],
            scheduler_control_plane_sha256=authorization[
                "scheduler_control_plane_sha256"
            ],
        )
    after = {
        entry.name: hashlib.sha256(entry.read_bytes()).hexdigest()
        for entry in root.iterdir()
        if entry.is_file()
    }
    assert after == before


@pytest.mark.parametrize("bad_id", (7000, True))
@pytest.mark.parametrize(
    "surface",
    (
        "claimed_role",
        "claimed_flat",
        "authority_role",
        "authority_flat",
    ),
)
def test_canary_initial_abort_rejects_non_string_job_ids_without_scheduler_or_writes(
    tmp_path: Path, surface: str, bad_id: object
) -> None:
    root = (tmp_path / f"abort-{surface}-{bad_id!r}").absolute()
    root.mkdir(mode=0o700)
    controller_lock = {
        "path": str(root / ".CANARY_CONTROLLER.lock"),
        "device": 1,
        "inode": 2,
        "uid": os.getuid(),
        "mode": 0o600,
    }
    value = {
        "schema_version": 1,
        "status": "two_wave_gpu_canary_aborted",
        "campaign_id": worker.CAMPAIGN_ID,
        "state_root": str(root),
        "canary_token": "0123456789abcdef",
        "controller_lock": controller_lock,
        "error": "fixture abort",
        "known_job_ids": ["7000"],
        "job_ids_by_role": {"wave0": ["7000"], "wave1": [], "report": []},
        "cancellation_authority_job_ids": ["7000"],
        "cancellation_authority_job_ids_by_role": {
            "wave0": ["7000"],
            "wave1": [],
            "report": [],
        },
        "reconciliation_errors": [],
        "cancellation": None,
        "cancellation_error": "fixture no call",
        "cancel_attempt_history": [],
        "historical_numeric_id_recycled": False,
        "historical_recycled_job_ids": [],
        "historical_recycled_job_ids_by_role": {
            "wave0": [], "wave1": [], "report": []
        },
        "historical_recycled_evidence_sha256": [],
    }
    if surface == "claimed_role":
        value["job_ids_by_role"]["wave0"] = [bad_id]
    elif surface == "claimed_flat":
        value["known_job_ids"] = [bad_id]
    elif surface == "authority_role":
        value["cancellation_authority_job_ids_by_role"]["wave0"] = [bad_id]
    else:
        value["cancellation_authority_job_ids"] = [bad_id]
    worker.seal_json(root / "CANARY_ABORTED.json", value)
    before = {
        entry.name: hashlib.sha256(entry.read_bytes()).hexdigest()
        for entry in root.iterdir()
        if entry.is_file()
    }
    with pytest.raises(controller.CanarySubmissionError):
        controller._validated_canary_abort(
            submit,
            root,
            canary_token="0123456789abcdef",
            controller_lock=controller_lock,
            scancel="/fixture/scancel",
            expected_control_plane={"schema_version": 1},
            scheduler_fallback={},
            historical_job_ids=set(),
        )
    after = {
        entry.name: hashlib.sha256(entry.read_bytes()).hexdigest()
        for entry in root.iterdir()
        if entry.is_file()
    }
    assert after == before


def test_canary_terminal_recovery_census_cannot_be_forged_empty(tmp_path: Path) -> None:
    root = (tmp_path / "terminal-recovery").absolute()
    root.mkdir(mode=0o700)
    authorization = _authorization(root)
    _seal_controller_contract(root, authorization)
    identity = worker.read_json(root / "CANARY_CONTROLLER_IDENTITY.json", "identity")
    roles = authorization["job_names"]
    attempts = []
    rounds = []
    cursor = 0
    for round_index in range(3):
        spans = {}
        for role in ("wave0", "wave1", "report"):
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
                    "stdout": "",
                    "stderr": "",
                    "control_plane": {"schema_version": 1},
                    "canonical_boundary_error": None,
                }
            )
            spans[role] = {"start": cursor, "stop": cursor + 1}
            cursor += 1
        rounds.append(
            {
                "round": round_index,
                "job_ids_by_role": {"wave0": [], "wave1": [], "report": []},
                "scheduler_attempt_spans_by_role": spans,
            }
        )
    calling_hashes = {
        role: submit.file_sha256(root / f"CANARY_{role.upper()}_CALLING.json")
        for role in ("wave0", "wave1", "report")
    }
    durable_prefix = controller._validated_canary_postsubmission_prefix(
        submit,
        worker,
        root,
        identity=identity,
        protocol_sha256=authorization["package_protocol_sha256"],
        scheduler_executables=authorization["scheduler_executables"],
        scheduler_control_plane_sha256=authorization[
            "scheduler_control_plane_sha256"
        ],
    )
    claimed_by_role = {
        role: [authorization["job_ids"][role]] for role in ("wave0", "wave1", "report")
    }
    value = {
        "schema_version": 1,
        "status": "canary_recovered_terminal_no_active_jobs",
        "campaign_id": worker.CAMPAIGN_ID,
        "state_root": str(root),
        "canary_token": authorization["canary_token"],
        "controller_identity_sha256": submit.file_sha256(
            root / "CANARY_CONTROLLER_IDENTITY.json"
        ),
        "package_protocol_sha256": authorization["package_protocol_sha256"],
        "source_sha256": authorization["source_sha256"],
        "claimed_job_ids": ["7000", "8000", "9000"],
        "claimed_job_ids_by_role": claimed_by_role,
        "live_verified_job_ids": [],
        "live_verified_job_ids_by_role": {"wave0": [], "wave1": [], "report": []},
        "calling_intent_sha256_by_role": calling_hashes,
        "pre_cancel_census_rounds": rounds,
        "cancelled_live_job_ids": [],
        "cancel_calling_sha256": None,
        "cancellation": None,
        "cancel_attempt_history": [],
        "post_cancel_census_rounds": [],
        "post_cancel_active_job_ids_by_role": {"wave0": [], "wave1": [], "report": []},
        "durable_prefix_sha256": durable_prefix,
        "scheduler_control_plane_observations": attempts,
        "scheduler_calls": 9,
        "controller_lock": authorization["controller_lock"],
        "new_jobs_created": 0,
        "historical_numeric_id_recycled": False,
        "historical_recycled_job_ids": [],
        "historical_recycled_job_ids_by_role": {
            "wave0": [], "wave1": [], "report": []
        },
        "historical_recycled_evidence_sha256": [],
    }
    assert controller._validated_canary_recovery_result(
        submit,
        root,
        value,
        identity=identity,
        protocol_sha256=authorization["package_protocol_sha256"],
        role_names=roles,
        squeue="/fixture/squeue",
        scancel="/fixture/scancel",
        controller_lock=authorization["controller_lock"],
        scheduler_executables=authorization["scheduler_executables"],
        scheduler_control_plane_sha256=authorization[
            "scheduler_control_plane_sha256"
        ],
        expected_control_plane={"schema_version": 1},
        scheduler_fallback={},
        historical_job_ids=set(),
        recycled_cleanup_records=[],
    ) == value
    immutable_before = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    for bad_id in (7000, True):
        forged_surfaces = []
        claimed_role = copy.deepcopy(value)
        claimed_role["claimed_job_ids_by_role"]["wave0"] = [bad_id]
        forged_surfaces.append(claimed_role)
        claimed_flat = copy.deepcopy(value)
        claimed_flat["claimed_job_ids"][0] = bad_id
        forged_surfaces.append(claimed_flat)
        live_role = copy.deepcopy(value)
        live_role["live_verified_job_ids_by_role"]["wave0"] = [bad_id]
        live_role["live_verified_job_ids"] = [bad_id]
        live_role["cancelled_live_job_ids"] = [bad_id]
        forged_surfaces.append(live_role)
        post_role = copy.deepcopy(value)
        post_role["post_cancel_active_job_ids_by_role"]["wave0"] = [bad_id]
        forged_surfaces.append(post_role)
        census_role = copy.deepcopy(value)
        census_role["pre_cancel_census_rounds"][0]["job_ids_by_role"][
            "wave0"
        ] = [bad_id]
        forged_surfaces.append(census_role)
        for forged_surface in forged_surfaces:
            with pytest.raises((controller.CanarySubmissionError, submit.SubmissionError)):
                controller._validated_canary_recovery_result(
                    submit,
                    root,
                    forged_surface,
                    identity=identity,
                    protocol_sha256=authorization["package_protocol_sha256"],
                    role_names=roles,
                    squeue="/fixture/squeue",
                    scancel="/fixture/scancel",
                    controller_lock=authorization["controller_lock"],
                    scheduler_executables=authorization["scheduler_executables"],
                    scheduler_control_plane_sha256=authorization[
                        "scheduler_control_plane_sha256"
                    ],
                    expected_control_plane={"schema_version": 1},
                    scheduler_fallback={},
                    historical_job_ids=set(),
                    recycled_cleanup_records=[],
                )
    boolean_jobs = copy.deepcopy(value)
    boolean_jobs["new_jobs_created"] = False
    with pytest.raises(controller.CanarySubmissionError):
        controller._validated_canary_recovery_result(
            submit,
            root,
            boolean_jobs,
            identity=identity,
            protocol_sha256=authorization["package_protocol_sha256"],
            role_names=roles,
            squeue="/fixture/squeue",
            scancel="/fixture/scancel",
            controller_lock=authorization["controller_lock"],
            scheduler_executables=authorization["scheduler_executables"],
            scheduler_control_plane_sha256=authorization[
                "scheduler_control_plane_sha256"
            ],
            expected_control_plane={"schema_version": 1},
            scheduler_fallback={},
            historical_job_ids=set(),
            recycled_cleanup_records=[],
        )
    immutable_after = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert immutable_after == immutable_before
    for forged in (
        {
            **copy.deepcopy(value),
            "claimed_job_ids": [],
            "claimed_job_ids_by_role": {"wave0": [], "wave1": [], "report": []},
        },
        {
            **copy.deepcopy(value),
            "claimed_job_ids": ["7999", "8000", "9000"],
            "claimed_job_ids_by_role": {
                "wave0": ["7999"],
                "wave1": ["8000"],
                "report": ["9000"],
            },
        },
        {
            **copy.deepcopy(value),
            "calling_intent_sha256_by_role": {
                role: digest
                for role, digest in calling_hashes.items()
                if role != "report"
            },
        },
        {
            **copy.deepcopy(value),
            "calling_intent_sha256_by_role": {},
        },
    ):
        with pytest.raises(
            controller.CanarySubmissionError,
            match="claimed IDs differ|calling hashes differ",
        ):
            controller._validated_canary_recovery_result(
                submit,
                root,
                forged,
                identity=identity,
                protocol_sha256=authorization["package_protocol_sha256"],
                role_names=roles,
                squeue="/fixture/squeue",
                scancel="/fixture/scancel",
                controller_lock=authorization["controller_lock"],
                scheduler_executables=authorization["scheduler_executables"],
                scheduler_control_plane_sha256=authorization[
                    "scheduler_control_plane_sha256"
                ],
                expected_control_plane={"schema_version": 1},
                scheduler_fallback={},
                historical_job_ids=set(),
                recycled_cleanup_records=[],
            )
    forged = copy.deepcopy(value)
    forged["pre_cancel_census_rounds"][0]["job_ids_by_role"]["wave0"] = ["7000"]
    with pytest.raises(submit.SubmissionError, match="not derived"):
        controller._validated_canary_recovery_result(
            submit,
            root,
            forged,
            identity=identity,
            protocol_sha256=authorization["package_protocol_sha256"],
            role_names=roles,
            squeue="/fixture/squeue",
            scancel="/fixture/scancel",
            controller_lock=authorization["controller_lock"],
            scheduler_executables=authorization["scheduler_executables"],
            scheduler_control_plane_sha256=authorization[
                "scheduler_control_plane_sha256"
            ],
            expected_control_plane={"schema_version": 1},
            scheduler_fallback={},
            historical_job_ids=set(),
            recycled_cleanup_records=[],
        )
    bool_coerced = copy.deepcopy(value)
    for attempt in bool_coerced["scheduler_control_plane_observations"]:
        attempt["control_plane"]["schema_version"] = True
    with pytest.raises(submit.SubmissionError, match="canonical binding"):
        controller._validated_canary_recovery_result(
            submit,
            root,
            bool_coerced,
            identity=identity,
            protocol_sha256=authorization["package_protocol_sha256"],
            role_names=roles,
            squeue="/fixture/squeue",
            scancel="/fixture/scancel",
            controller_lock=authorization["controller_lock"],
            scheduler_executables=authorization["scheduler_executables"],
            scheduler_control_plane_sha256=authorization[
                "scheduler_control_plane_sha256"
            ],
            expected_control_plane={"schema_version": 1},
            scheduler_fallback={},
            historical_job_ids=set(),
            recycled_cleanup_records=[],
        )


def test_real_canary_recovery_entrypoint_consumes_lost_cancel_and_rechecks_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The production action always starts in a fresh isolated interpreter. Other
    # package tests may have imported TreeWM earlier in this shared pytest process;
    # remove those modules for this functional entrypoint emulation and let
    # monkeypatch restore them afterward.
    for name in list(sys.modules):
        if name == "treewm" or name.startswith("treewm."):
            monkeypatch.delitem(sys.modules, name)
    repo = (tmp_path / "repo").absolute()
    package = repo / controller.PACKAGE_RELATIVE
    root = (
        repo
        / controller.CANARY_PARENT_RELATIVE
        / "exp23-launch8-two-wave-canary-entrypoint"
    )
    source = root / "source"
    package.mkdir(parents=True)
    source.mkdir(parents=True)
    source_sha256 = {}
    for name in controller.CANARY_SOURCE_FILES:
        payload = (PACKAGE / name).read_bytes()
        (package / name).write_bytes(payload)
        (source / name).write_bytes(payload)
        source_sha256[name] = hashlib.sha256(payload).hexdigest()
        (source / name).chmod(0o444)
    source.chmod(0o555)
    root.chmod(0o700)
    executables = {}
    for name in ("sbatch", "scontrol", "squeue", "scancel"):
        path = tmp_path / name
        path.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
        path.chmod(0o755)
        executables[name] = str(path)
    manifest = {
        "campaign_id": worker.CAMPAIGN_ID,
        "status": "sealed_launch_ready_unsubmitted",
        "formal_validation": False,
        "paths": {"python": sys.executable},
        "launch_contract": {
            "real_gpu_two_wave_canary": {
                "failed_attempts": [
                    {
                        "state_root": str(
                            repo
                            / controller.CANARY_PARENT_RELATIVE
                            / "exp23-launch8-two-wave-canary-prior-anchor"
                        ),
                        "canary_token": "abcdef0123456789",
                        "job_ids_by_role": {
                            "wave0": ["6000"],
                            "wave1": ["6001"],
                            "report": ["6002"],
                        },
                    }
                ],
                "accepted_attempts": [],
            }
        },
        "execution": {
            **executables,
            "scheduler_control_plane": json.loads(
                (PACKAGE / "manifest.json").read_text(encoding="utf-8")
            )["execution"]["scheduler_control_plane"],
        },
    }
    (package / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    lock_path = root / ".CANARY_CONTROLLER.lock"
    lock_path.touch(mode=0o600)
    lock_path.chmod(0o600)
    lock_info = lock_path.stat()
    controller_lock = {
        "path": str(lock_path),
        "device": lock_info.st_dev,
        "inode": lock_info.st_ino,
        "uid": os.getuid(),
        "mode": 0o600,
    }
    token = "0123456789abcdef"
    names = {
        "wave0": f"exp23-launch8-canary-{token}-wave0",
        "wave1": f"exp23-launch8-canary-{token}-wave1",
        "report": f"exp23-launch8-canary-{token}-report",
    }
    protocol_sha256 = "a" * 64
    observation = {"schema_version": 1, "mode": "fixture"}
    identity = {
        "schema_version": 1,
        "status": "canary_controller_claimed",
        "campaign_id": worker.CAMPAIGN_ID,
        "scientific": False,
        "state_root": str(root),
        "canary_token": token,
        "job_names": names,
        "scheduler_comment": f"treewm-exp23-canary:{token}",
        "controller_lock": controller_lock,
        "source_sha256": source_sha256,
        "package_protocol_sha256": protocol_sha256,
        "scheduler_control_plane": observation,
        "scheduler_control_plane_sha256": submit.stable_hash(observation),
    }
    worker.seal_json(root / "CANARY_CONTROLLER_IDENTITY.json", identity)
    fake_campaign = SimpleNamespace(
        load_contract=lambda _repo: (manifest, {}),
        verify_protocol_lock=lambda _package: protocol_sha256,
    )
    monkeypatch.setattr(controller, "_load_submit", lambda _repo: submit)
    monkeypatch.setattr(controller, "_load_campaign", lambda _repo: fake_campaign)
    original_flags = controller.sys.flags

    class IsolatedFlags:
        isolated = 1
        no_site = 1
        dont_write_bytecode = 1
        safe_path = 1

        def __getattr__(self, name):
            return getattr(original_flags, name)

    monkeypatch.setattr(
        controller.sys,
        "flags",
        IsolatedFlags(),
    )
    monkeypatch.setattr(controller.sys, "dont_write_bytecode", True)
    monkeypatch.setattr(
        submit,
        "_scheduler_fallback_config",
        lambda _control: {"source_control_plane": observation},
    )
    monkeypatch.setattr(
        submit, "_scheduler_control_plane_observation", lambda _control: observation
    )
    monkeypatch.setattr(submit.time, "sleep", lambda _seconds: None)
    journal_authorization = {
        "canary_token": token,
        "job_ids": {"wave0": "7000", "wave1": "8000", "report": "999999992"},
        "job_names": names,
        "scheduler_comment": identity["scheduler_comment"],
        "scheduler_executables": {
            "submit": executables["sbatch"],
            "control": executables["scontrol"],
        },
        "scheduler_control_plane": observation,
        "scheduler_control_plane_sha256": submit.stable_hash(observation),
        "controller_lock": controller_lock,
    }
    records, evidence = _accepted_scheduler_fixtures(root, journal_authorization)
    for role in ("wave0", "wave1", "report"):
        worker.seal_json(
            root / f"CANARY_{role.upper()}_CALLING.json",
            _calling_payload(root, journal_authorization, role),
        )
    worker.seal_json(
        root / "CANARY_WAVE0_SUBMITTED.json",
        {
            "schema_version": 1,
            "status": "canary_wave0_submitted_held",
            "job_id": "7000",
            "accepted_submission_record": records["wave0"],
            "accepted_hold": evidence["wave0_accepted_hold"],
        },
    )
    worker.seal_json(
        root / "CANARY_WAVE1_SUBMITTED.json",
        {
            "schema_version": 1,
            "status": "canary_wave1_submitted",
            "job_id": "8000",
            "accepted_submission_record": records["wave1"],
            "accepted_dependency": evidence["wave1_accepted_dependency"],
        },
    )
    cancel_context = {
        "schema_version": 1,
        "status": "canary_scheduler_calling",
        "campaign_id": worker.CAMPAIGN_ID,
        "state_root": str(root),
        "canary_token": token,
        "role": "recovery_cancel",
        "controller_lock": controller_lock,
    }
    submit.exclusive_json(
        root / "CANARY_RECOVERY_CANCEL_CALLING_0000.json",
        {
            **cancel_context,
            "attempt_index": 0,
            "job_ids": ["7000"],
            "command": [executables["scancel"], "7000"],
        },
    )
    initial_history = submit._validated_recovery_cancel_history(
        root,
        calling_prefix="CANARY_RECOVERY_CANCEL_CALLING",
        result_prefix="CANARY_RECOVERY_CANCEL_RESULT",
        context=cancel_context,
        scancel=executables["scancel"],
        expected_control_plane=observation,
        fallback={"source_control_plane": observation},
    )
    worker.seal_json(
        root / "CANARY_ABORTED.json",
        {
            "schema_version": 1,
            "status": "two_wave_gpu_canary_aborted",
            "campaign_id": worker.CAMPAIGN_ID,
            "state_root": str(root),
            "canary_token": token,
            "controller_lock": controller_lock,
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
            "reconciliation_errors": ["scancel: lost scheduler response"],
            "cancellation": None,
                "cancellation_error": "lost scheduler response",
                "cancel_attempt_history": initial_history,
                "historical_numeric_id_recycled": False,
                "historical_recycled_job_ids": [],
                "historical_recycled_job_ids_by_role": {
                    "wave0": [], "wave1": [], "report": []
                },
                "historical_recycled_evidence_sha256": [],
            },
        )
    active_job_id: str | None = "7000"
    calls = []
    expected_user = submit.pwd.getpwuid(os.getuid()).pw_name

    def runner(command, _cwd, _environment, inherited_fds):
        nonlocal active_job_id
        values = list(command)
        calls.append(values)
        assert inherited_fds
        if Path(values[0]).name == "squeue":
            name = next(item.split("=", 1)[1] for item in values if item.startswith("--name="))
            stdout = (
                f"{active_job_id}|{names['wave0']}|{expected_user}|PENDING|{identity['scheduler_comment']}\n"
                if active_job_id is not None and name == names["wave0"]
                else ""
            )
            return subprocess.CompletedProcess(values, 0, stdout=stdout, stderr="")
        assert Path(values[0]).name == "scancel" and values[-1] == active_job_id
        active_job_id = None
        return subprocess.CompletedProcess(values, 0, stdout="", stderr="")

    result = controller.recover_or_cancel_real_canary(
        repo,
        root,
        controller.CONFIRMATION,
        scheduler_runner=runner,
    )
    assert result["status"] == "canary_recovered_terminal_after_cancel_attempts"
    assert result["claimed_job_ids"] == ["7000", "7999", "8000"]
    assert result["claimed_job_ids_by_role"] == {
        "wave0": ["7000", "7999"],
        "wave1": ["8000"],
        "report": [],
    }
    assert set(result["calling_intent_sha256_by_role"]) == {
        "wave0",
        "wave1",
        "report",
    }
    assert result["live_verified_job_ids"] == ["7000"]
    assert [row["attempt_index"] for row in result["cancel_attempt_history"]] == [0, 1]
    assert result["cancel_attempt_history"][0]["result_sha256"] is None
    assert result["post_cancel_active_job_ids_by_role"] == {
        "wave0": [], "wave1": [], "report": []
    }
    assert result["scheduler_calls"] == 19
    assert [Path(row[0]).name for row in calls].count("scancel") == 1
    recovery_source = inspect.getsource(controller.recover_or_cancel_real_canary)
    assert "next(iter(claimed" not in recovery_source
    assert 'journal_values[role]["job_id"]' in recovery_source

    active_job_id = "6000"
    prior_calls = len(calls)
    original_exclusive = submit.exclusive_json

    def crash_before_residual_terminal(path, value, *args, **kwargs):
        if Path(path).name == "CANARY_RECOVERY_RECONCILED_0000.json":
            raise submit.SubmissionError(
                "fixture crash after residual recycled-ID cancel result"
            )
        return original_exclusive(path, value, *args, **kwargs)

    monkeypatch.setattr(submit, "exclusive_json", crash_before_residual_terminal)
    with pytest.raises(
        submit.SubmissionError,
        match="fixture crash after residual recycled-ID cancel result",
    ):
        controller.recover_or_cancel_real_canary(
            repo,
            root,
            controller.CONFIRMATION,
            scheduler_runner=runner,
        )
    assert len(calls) - prior_calls == 19
    assert [Path(row[0]).name for row in calls[prior_calls:]].count("scancel") == 1
    residual_marker_path = root / "CANARY_HISTORICAL_ID_RECYCLED_0000.json"
    residual_marker = worker.read_json(
        residual_marker_path, "residual recycled-ID marker"
    )
    assert residual_marker["phase"] == "residual_recovery"
    assert residual_marker["cancel_history_length_before"] == 2
    residual_calling = worker.read_json(
        root / "CANARY_RECOVERY_CANCEL_CALLING_0002.json",
        "residual recycled-ID calling",
    )
    assert residual_calling["historical_recycled_evidence"] == {
        "name": residual_marker_path.name,
        "sha256": hashlib.sha256(residual_marker_path.read_bytes()).hexdigest(),
    }
    assert not os.path.lexists(root / "CANARY_RECOVERY_RECONCILED_0000.json")
    monkeypatch.setattr(submit, "exclusive_json", original_exclusive)
    prior_calls = len(calls)
    resumed = controller.recover_or_cancel_real_canary(
        repo,
        root,
        controller.CONFIRMATION,
        scheduler_runner=runner,
    )
    assert len(calls) - prior_calls == 9
    assert not any(
        Path(row[0]).name == "scancel" for row in calls[prior_calls:]
    )
    chain = resumed["residual_reconciliation_chain"]
    assert len(chain) == 1
    assert chain[0]["previous_terminal_name"] == "CANARY_RECOVERY_CANCELLED.json"
    assert chain[0]["live_verified_job_ids"] == []
    assert chain[0]["status"] == (
        "canary_recovered_residual_terminal_no_active_jobs"
    )

    prior_calls = len(calls)
    reused = controller.recover_or_cancel_real_canary(
        repo,
        root,
        controller.CONFIRMATION,
        scheduler_runner=runner,
    )
    assert len(calls) - prior_calls == 9
    assert len(reused["residual_reconciliation_chain"]) == 1


@pytest.mark.parametrize("mutation", ["missing-hash", "wrong-checkpoint", "extra", "nonfinite"])
def test_canary_report_rejects_forged_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = (tmp_path / mutation).absolute()
    root.mkdir(mode=0o700)
    authorization = _authorization(root)
    _seal_controller_contract(root, authorization)
    monkeypatch.setattr(worker, "__file__", str(root / "source/canary_worker.py"))
    checkpoint = root / "wave0_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint.chmod(0o444)
    ready = {
        "schema_version": 1,
        "status": "wave0_ready",
        "campaign_id": worker.CAMPAIGN_ID,
        "canary_token": authorization["canary_token"],
        "wave0_job_id": "7000",
        "checkpoint_sha256": worker.sha256_file(checkpoint),
        "expected_resumed_result": 123.0,
        "cuda_device_name": "fixture",
        "within_wave_requeue": False,
    }
    worker.seal_json(root / worker.READY_NAME, ready)
    complete = {
        "schema_version": 1,
        "status": "wave1_complete",
        "campaign_id": worker.CAMPAIGN_ID,
        "canary_token": authorization["canary_token"],
        "wave0_job_id": "7000",
        "wave1_job_id": "8000",
        "ready_sha256": worker.sha256_file(root / worker.READY_NAME),
        "checkpoint_sha256": worker.sha256_file(checkpoint),
        "resumed_result": 123.0,
        "within_wave_requeue": False,
    }
    if mutation == "missing-hash":
        del complete["ready_sha256"]
    elif mutation == "wrong-checkpoint":
        complete["checkpoint_sha256"] = "f" * 64
    elif mutation == "extra":
        complete["forged"] = True
    else:
        complete["resumed_result"] = "nan"
    worker.seal_json(root / worker.COMPLETE_NAME, complete)
    monkeypatch.setenv("SLURM_JOB_ID", "9000")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    with pytest.raises(worker.CanaryError, match="completion identity"):
        worker._report(root, authorization)
    assert not os.path.lexists(root / worker.REPORT_NAME)


def test_canary_compute_surface_has_no_scheduler_client_and_no_requeue() -> None:
    worker_source = (PACKAGE / "canary_worker.py").read_text(encoding="utf-8").lower()
    for client in ("scontrol", "squeue", "sbatch", "scancel", "sacct"):
        assert client not in worker_source
    syntax = ast.parse(worker_source)
    top_level_imports = {
        alias.name
        for node in syntax.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in top_level_imports

    for filename in ("canary_gpu.slurm", "canary_report.slurm"):
        source = (PACKAGE / filename).read_text(encoding="utf-8")
        assert "#SBATCH --no-requeue" in source
        assert '"$PYTHON_EXECUTABLE" -I -S -B' in source
        executable = "\n".join(
            line for line in source.splitlines() if not line.startswith("#SBATCH")
        ).lower()
        for client in ("scontrol", "squeue", "sbatch", "scancel", "sacct"):
            assert client not in executable
        completed = subprocess.run(
            ["/bin/bash", "-n", str(PACKAGE / filename)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_exact_canary_isolated_runtime_imports_bound_torch() -> None:
    code = (
        "import runpy;"
        f"d=runpy.run_path({str(PACKAGE / 'canary_worker.py')!r});"
        "d['_bootstrap_runtime']();"
        "t=d['_verified_torch']();"
        "print(t.__version__, t.__file__)"
    )
    completed = subprocess.run(
        [str(worker.PINNED_PYTHON), "-I", "-S", "-B", "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "site-packages/torch/__init__.py" in completed.stdout


def test_canary_controller_orders_full_graph_and_journals_before_release() -> None:
    source = inspect.getsource(controller.submit_real_canary)
    identity = source.index("CANARY_CONTROLLER_IDENTITY.json")
    wave0 = source.index("wave0_id, wave0_record")
    hold = source.index("wave0_hold =")
    wave0_journal = source.index("CANARY_WAVE0_SUBMITTED.json")
    wave1 = source.index("wave1_id, wave1_record")
    wave1_distinct = source.index("wave1_id != wave0_id")
    wave1_dependency = source.index("wave1_dependency =")
    wave1_journal = source.index("CANARY_WAVE1_SUBMITTED.json")
    report = source.index("report_id, report_record")
    report_distinct = source.index("report_id not in {wave0_id, wave1_id}")
    report_dependency = source.index("report_dependency =")
    report_journal = source.index("CANARY_REPORT_SUBMITTED.json")
    authorization = source.index("auth_sha256 = submit.exclusive_json")
    release = source.index("release = submit._release_authorized_wave0")
    receipt = source.index("CANARY_SUBMISSION_RECEIPT.json")
    assert identity < wave0 < hold < wave0_journal < wave1
    assert wave1 < wave1_distinct < wave1_dependency < wave1_journal < report
    assert report < report_distinct < report_dependency < report_journal
    ready = source.index("CANARY_READY_TO_RELEASE.json")
    assert report_journal < authorization < receipt < ready < release
    assert '"--hold"' in source
    assert source.count('"--kill-on-invalid-dep=yes"') == 2


def test_canary_recovery_cli_is_reachable_and_never_submits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state = (tmp_path / "exp23-launch8-two-wave-canary-recovery").absolute()
    calls = []
    recovery_source = inspect.getsource(controller.recover_or_cancel_real_canary)

    def recover(repo_root, state_root, confirmation):
        calls.append((repo_root, state_root, confirmation))
        return {"status": "canary_recovered_cancelled", "new_jobs_created": 0}

    monkeypatch.setattr(controller, "recover_or_cancel_real_canary", recover)
    assert controller.main(
        [
            "--recover-or-cancel-real-gpu-canary",
            "--state-root",
            str(state),
            "--confirmation",
            controller.CONFIRMATION,
        ]
    ) == 0
    assert calls and calls[0][1] == state
    assert json.loads(capsys.readouterr().out)["new_jobs_created"] == 0
    assert "_recovery_census_rounds" in recovery_source
    assert "_append_recovery_cancel_attempt" in recovery_source
    assert "_submit_one" not in recovery_source
    assert "with _CanaryControllerLock(root, create=False)" in recovery_source


def test_scientific_preflights_never_invoke_real_canary() -> None:
    submit_source = (PACKAGE / "submit.py").read_text(encoding="utf-8")
    assert "submit_real_canary" not in submit_source
    assert "--submit-real-gpu-two-wave-canary" not in submit_source
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    canary = manifest["launch_contract"]["real_gpu_two_wave_canary"]
    assert canary["preflight_invocation"] is False
    assert canary["run_during_read_only_preflight"] is False
    assert canary["hard_crash_action"] == "--recover-or-cancel-real-gpu-canary"
    assert canary["failed_attempts"][0]["canary_token"] == "e09ce7d5a0cef1b0"
    assert canary["failed_attempts"][0]["job_ids_by_role"] == {
        "wave0": ["33285485"],
        "wave1": ["33285486"],
        "report": [],
    }
    assert canary["failed_attempts"][0]["reuse_allowed"] is False
    assert canary["failed_attempts"][0]["recovery_allowed"] is False
