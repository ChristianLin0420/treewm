from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any, Callable

import pytest

import engineering_pilot_binder as binder
import runtime


EXP23_TEST_GATE = (
    runtime.REPOSITORY_ROOT
    / runtime.FROZEN_LAUNCH8_PACKAGE_RELATIVE
    / "tests"
    / "test_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "exp24_synthetic_frozen_launch8_fixture", EXP23_TEST_GATE
)
assert SPEC is not None and SPEC.loader is not None
fixture_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture_module)
frozen_gate = fixture_module.gate


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _pretty(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _copy_frozen_source(submission_root: Path) -> Path:
    source_root = submission_root / "source-snapshot" / "repo"
    source_root.mkdir(parents=True)
    for relative in runtime.launch8_verifier_dependency_relatives(
        runtime.REPOSITORY_ROOT
    ):
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runtime.REPOSITORY_ROOT / relative, destination)
        destination.chmod(0o444)
    for directory in sorted(
        (path for path in source_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    source_root.chmod(0o555)
    assert (
        runtime.stable_hash(
            runtime.launch8_verifier_source_inventory(
                source_root, immutable=True
            )
        )
        == runtime.FROZEN_LAUNCH8_VERIFIER_SOURCE_INVENTORY_SHA256
    )
    return source_root


def _add_sealed_file(source_root: Path, relative: Path) -> Path:
    current = source_root
    for part in relative.parent.parts:
        child = current / part
        if not child.exists():
            current.chmod(0o755)
            child.mkdir()
            child.chmod(0o555)
            current.chmod(0o555)
        current = child
    current.chmod(0o755)
    destination = source_root / relative
    destination.write_bytes(b"hostile ignored source-tree entry\n")
    destination.chmod(0o444)
    current.chmod(0o555)
    return destination


def _add_sealed_empty_directory(source_root: Path, relative: Path) -> Path:
    parent = source_root / relative.parent
    parent.chmod(0o755)
    destination = source_root / relative
    destination.mkdir()
    destination.chmod(0o555)
    parent.chmod(0o555)
    return destination


def _synthetic_objects(submission_sha256: str) -> tuple[dict, dict, dict]:
    bundle = fixture_module.valid_bundle()
    decision = frozen_gate.evaluate_bundle(bundle)
    assert decision["status"] == "accepted_engineering_pilot"
    manifest = frozen_gate.load_manifest()
    production = binder._expected_production_authorization(manifest)
    boundary_evaluations = [
        {
            "index": cell["index"],
            "setting_id": cell["setting_id"],
            "arm_id": cell["arm_id"],
            "seed": cell["seed"],
            "boundaries": cell["boundaries"],
        }
        for cell in decision["cells"]
    ]
    events = []
    terminals = []
    for cell in bundle["cells"]:
        index = cell["index"]
        outcome = cell["boundaries"]["25000"]["outcome"]
        worker_sha = _sha(f"worker-complete/{index}")
        events.append(
            {
                "index": index,
                "event_files": [f"events.out.tfevents.synthetic-{index:02d}"],
                "event_file_sha256": {
                    f"events.out.tfevents.synthetic-{index:02d}": _sha(
                        f"event/{index}"
                    )
                },
                "hparams_event_files": [
                    f"hparams/events.out.tfevents.synthetic-{index:02d}"
                ],
                "hparams_event_file_sha256": {
                    f"hparams/events.out.tfevents.synthetic-{index:02d}": _sha(
                        f"hparams/{index}"
                    )
                },
                "excluded_eval_tags": ["eval/synthetic_excluded"],
                "fixed_validation_text_events": 1,
                "identical_scalar_duplicates": {},
            }
        )
        terminals.append(
            {
                "index": index,
                "worker_complete_sha256": worker_sha,
                "wave_index": 0,
                "array_job_id": "12345",
                "checkpoint_sha256": outcome["checkpoint_sha256"],
                "completion_sha256": outcome["completion_sha256"],
                "final_eval_progress_sha256": outcome[
                    "final_eval_progress_sha256"
                ],
                "completed_results_sha256": outcome[
                    "completed_results_sha256"
                ],
                "identity_sha256": _sha(f"identity/{index}"),
                "wave_lineage": {
                    "branch": "wave0_complete_wave1_noop",
                    "wave0_start_sha256": _sha(f"wave0-start/{index}"),
                    "wave0_predecessor_evidence_name": "WORKER_COMPLETE.json",
                    "wave0_predecessor_evidence_sha256": worker_sha,
                    "wave0_checkpoint_sha256": outcome["checkpoint_sha256"],
                    "wave1_start_sha256": _sha(f"wave1-start/{index}"),
                    "wave1_input_checkpoint_sha256": outcome[
                        "checkpoint_sha256"
                    ],
                    "wave1_predecessor_evidence_sha256": worker_sha,
                    "wave1_noop_sha256": _sha(f"wave1-noop/{index}"),
                },
            }
        )
    provenance = {
        "schema_version": 1,
        "campaign_id": runtime.EXPECTED_ACCEPTED_PILOT_CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "production_authorization_prerequisite": production,
        "production_authorization_prerequisite_sha256": runtime.stable_hash(
            production
        ),
        "outcome_blind_phase": {
            "status": "all_boundaries_parsed_and_calibration_computed_before_outcomes",
            "boundary_evaluations_sha256": runtime.stable_hash(
                boundary_evaluations
            ),
            "paired_calibration_sha256": runtime.stable_hash(
                decision["paired_25k_calibration"]
            ),
        },
        "event_artifacts": events,
        "terminal_artifacts": terminals,
        "report_bundle_sha256": runtime.stable_hash(bundle),
        "gate_sha256": decision["gate_sha256"],
    }
    return bundle, decision, provenance


def _publish_synthetic_quartet(
    submission_root: Path,
    bundle: dict,
    decision: dict,
    provenance: dict,
) -> Path:
    report_root = submission_root / "report"
    report_root.mkdir()
    bundle_sha = runtime.stable_hash(bundle)
    decision["report_bundle_sha256"] = bundle_sha
    decision_body = dict(decision)
    decision_body.pop("gate_sha256", None)
    gate_sha = runtime.stable_hash(decision_body)
    decision["gate_sha256"] = gate_sha
    provenance["report_bundle_sha256"] = bundle_sha
    provenance["gate_sha256"] = gate_sha
    provenance_sha = runtime.stable_hash(provenance)
    payloads = {
        f"REPORT_BUNDLE.{bundle_sha}.json": _pretty(bundle),
        f"GATE_DECISION.{gate_sha}.json": _pretty(decision),
        f"REPORT_PROVENANCE.{provenance_sha}.json": _pretty(provenance),
    }
    commit = {
        "schema_version": 1,
        "status": "accepted_engineering_pilot",
        "scientific_rejection": False,
        "campaign_id": runtime.EXPECTED_ACCEPTED_PILOT_CAMPAIGN_ID,
        "submission_sha256": provenance["submission_sha256"],
        "report_bundle": f"REPORT_BUNDLE.{bundle_sha}.json",
        "report_bundle_sha256": bundle_sha,
        "report_bundle_file_sha256": hashlib.sha256(
            payloads[f"REPORT_BUNDLE.{bundle_sha}.json"]
        ).hexdigest(),
        "gate_decision": f"GATE_DECISION.{gate_sha}.json",
        "gate_sha256": gate_sha,
        "gate_decision_file_sha256": hashlib.sha256(
            payloads[f"GATE_DECISION.{gate_sha}.json"]
        ).hexdigest(),
        "provenance": f"REPORT_PROVENANCE.{provenance_sha}.json",
        "provenance_sha256": provenance_sha,
        "provenance_file_sha256": hashlib.sha256(
            payloads[f"REPORT_PROVENANCE.{provenance_sha}.json"]
        ).hexdigest(),
    }
    payloads["REPORT_COMMIT.json"] = _pretty(commit)
    for name, payload in payloads.items():
        path = report_root / name
        path.write_bytes(payload)
        path.chmod(0o444)
    report_root.chmod(0o555)
    return report_root


def _quartet(
    tmp_path: Path,
    mutate: Callable[[dict, dict, dict], None] | None = None,
) -> tuple[Path, Path, str, dict]:
    submission_root = (tmp_path / "synthetic-launch8-submission").absolute()
    submission_root.mkdir()
    _copy_frozen_source(submission_root)
    submission_sha = _sha("synthetic-submission")
    bundle, decision, provenance = _synthetic_objects(submission_sha)
    if mutate is not None:
        mutate(bundle, decision, provenance)
    report_root = _publish_synthetic_quartet(
        submission_root, bundle, decision, provenance
    )
    return report_root, submission_root, submission_sha, bundle


def _verify(report_root: Path, submission_root: Path, submission_sha: str) -> dict:
    return binder.verify_engineering_pilot_report_quartet(
        report_root,
        expected_report_root=report_root,
        expected_submission_root=submission_root,
        expected_submission_sha256=submission_sha,
        expected_package_binding=runtime.FROZEN_LAUNCH8_PACKAGE_BINDING,
    )


def test_synthetic_accepted_quartet_recomputes_every_gate_but_never_binds(
    tmp_path: Path,
) -> None:
    report_root, submission_root, submission_sha, _bundle = _quartet(tmp_path)
    result = _verify(report_root, submission_root, submission_sha)
    assert result["status"] == "accepted_engineering_pilot_semantics_verified_unpublished"
    assert result["source_commit"] == runtime.FROZEN_LAUNCH8_SOURCE_COMMIT
    assert result["source_inventory_sha256"] == (
        runtime.FROZEN_LAUNCH8_VERIFIER_SOURCE_INVENTORY_SHA256
    )
    assert result["source_file_count"] == 147
    assert all(result["recomputed_top_gates"].values())
    assert all(result["recomputed_paired_calibration_gates"].values())
    assert all(result["recomputed_absolute_outcome_gates"].values())
    assert all(result["recomputed_paired_outcome_gates"].values())
    assert result["binding_state"] == "unbound"
    assert result["formal_submission_allowed"] is False
    assert result["persistent_writes_performed"] is False
    assert result["binding_published"] is False
    assert runtime.accepted_engineering_pilot_placeholder()["binding_sha256"] is None


def test_raw_scalar_boundary_cannot_hide_behind_forged_accepted_decision(
    tmp_path: Path,
) -> None:
    def mutate(bundle: dict, _decision: dict, _provenance: dict) -> None:
        cell = next(
            row
            for row in bundle["cells"]
            if row["setting_id"] == "scene"
            and row["arm_id"] == "GSEP"
            and row["seed"] == 110
        )
        cell["boundaries"]["25000"]["scalars"]["tree/support_recall"][0][1] = 0.0

    report_root, submission_root, submission_sha, _bundle = _quartet(
        tmp_path, mutate
    )
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match="gate did not accept the raw bundle",
    ):
        _verify(report_root, submission_root, submission_sha)


def test_rehashed_completed_rows_and_aggregates_still_require_exact_25_row_identity(
    tmp_path: Path,
) -> None:
    def mutate(bundle: dict, _decision: dict, _provenance: dict) -> None:
        outcome = bundle["cells"][0]["boundaries"]["25000"]["outcome"]
        outcome["completed_results"][7]["episode_seed"] += 1
        outcome["completed_results_sha256"] = runtime.stable_hash(
            outcome["completed_results"]
        )

    report_root, submission_root, submission_sha, _bundle = _quartet(
        tmp_path, mutate
    )
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match="gate did not accept the raw bundle",
    ):
        _verify(report_root, submission_root, submission_sha)


@pytest.mark.parametrize(
    "terminal_hash",
    [
        "completed_results_sha256",
        "completion_sha256",
        "final_eval_progress_sha256",
        "checkpoint_sha256",
    ],
)
def test_each_terminal_bundle_provenance_hash_join_is_mandatory(
    tmp_path: Path, terminal_hash: str
) -> None:
    def mutate(_bundle: dict, _decision: dict, provenance: dict) -> None:
        provenance["terminal_artifacts"][3][terminal_hash] = _sha(
            f"hostile/{terminal_hash}"
        )

    report_root, submission_root, submission_sha, _bundle = _quartet(
        tmp_path, mutate
    )
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match=f"{terminal_hash} join differs",
    ):
        _verify(report_root, submission_root, submission_sha)


def test_outcome_blind_boundary_hash_is_recomputed_not_trusted(
    tmp_path: Path,
) -> None:
    def mutate(_bundle: dict, _decision: dict, provenance: dict) -> None:
        provenance["outcome_blind_phase"]["boundary_evaluations_sha256"] = _sha(
            "forged-boundaries"
        )

    report_root, submission_root, submission_sha, _bundle = _quartet(
        tmp_path, mutate
    )
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match="telemetry-to-decision phase hashes differ",
    ):
        _verify(report_root, submission_root, submission_sha)


def test_exact_structural_method_candidate_gate_schemas_reject_extra_key(
    tmp_path: Path,
) -> None:
    def mutate(_bundle: dict, decision: dict, _provenance: dict) -> None:
        decision["cells"][0]["boundaries"]["5000"]["structural_gates"][
            "forged"
        ] = True

    report_root, submission_root, submission_sha, _bundle = _quartet(
        tmp_path, mutate
    )
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match="published gate decision differs",
    ):
        _verify(report_root, submission_root, submission_sha)


@pytest.mark.parametrize(
    "relative",
    [
        Path("provenance/__init__.pyc"),
        Path("treewm/__pycache__/__init__.cpython-311.pyc"),
        runtime.FROZEN_LAUNCH8_PACKAGE_RELATIVE
        / "hostile.cpython-311-x86_64-linux-gnu.so",
        Path("hostile.pth"),
        Path("sitecustomize.py"),
        Path("arbitrary-leaf"),
    ],
    ids=(
        "provenance-pyc",
        "unchecked-pycache",
        "extension-module",
        "pth",
        "sitecustomize",
        "arbitrary-leaf",
    ),
)
def test_every_unexpected_source_file_is_rejected_before_gate_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: Path
) -> None:
    report_root, submission_root, submission_sha, _bundle = _quartet(tmp_path)
    source_root = submission_root / "source-snapshot" / "repo"
    _add_sealed_file(source_root, relative)
    calls: list[object] = []

    def forbidden_replay(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        raise AssertionError("gate replay ran before exact source-tree authentication")

    monkeypatch.setattr(binder.subprocess, "run", forbidden_replay)
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match="unexpected (file|directory)",
    ):
        _verify(report_root, submission_root, submission_sha)
    assert calls == []


def test_empty_source_directory_is_rejected_before_gate_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_root, submission_root, submission_sha, _bundle = _quartet(tmp_path)
    source_root = submission_root / "source-snapshot" / "repo"
    _add_sealed_empty_directory(source_root, Path("empty-ignored-directory"))
    calls: list[object] = []

    def forbidden_replay(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        raise AssertionError("gate replay ran before exact source-tree authentication")

    monkeypatch.setattr(binder.subprocess, "run", forbidden_replay)
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match="unexpected directory",
    ):
        _verify(report_root, submission_root, submission_sha)
    assert calls == []


def test_symlinked_launch8_package_directory_is_rejected_before_gate_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_root, submission_root, submission_sha, _bundle = _quartet(tmp_path)
    source_root = submission_root / "source-snapshot" / "repo"
    package = source_root / runtime.FROZEN_LAUNCH8_PACKAGE_RELATIVE
    detached = package.parent / "zz-detached-launch8-package"
    package.parent.chmod(0o755)
    package.rename(detached)
    package.symlink_to(detached.name, target_is_directory=True)
    package.parent.chmod(0o555)
    calls: list[object] = []

    def forbidden_replay(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        raise AssertionError("gate replay ran before exact source-tree authentication")

    monkeypatch.setattr(binder.subprocess, "run", forbidden_replay)
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match="cannot open source|symlink or special entry",
    ):
        _verify(report_root, submission_root, submission_sha)
    assert calls == []


def test_add_remove_aba_during_replay_changes_bound_tree_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_root, submission_root, submission_sha, _bundle = _quartet(tmp_path)
    source_root = submission_root / "source-snapshot" / "repo"
    real_replay = binder._run_frozen_gate

    def mutating_replay(**kwargs: Any) -> dict:
        result = real_replay(**kwargs)
        package = source_root / runtime.FROZEN_LAUNCH8_PACKAGE_RELATIVE
        transient = package / "transient-ignored-leaf"
        package.chmod(0o755)
        transient.write_bytes(b"appeared and disappeared during replay\n")
        transient.unlink()
        package.chmod(0o555)
        return result

    monkeypatch.setattr(binder, "_run_frozen_gate", mutating_replay)
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match="source tree changed during gate replay",
    ):
        _verify(report_root, submission_root, submission_sha)


def test_source_snapshot_swap_restore_during_replay_cannot_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_root, submission_root, submission_sha, _bundle = _quartet(tmp_path)
    real_replay = binder._run_frozen_gate

    def swapping_replay(**kwargs: Any) -> dict:
        source_snapshot = submission_root / "source-snapshot"
        backup = submission_root / "source-snapshot.retained-aba"
        source_snapshot.rename(backup)
        source_snapshot.mkdir()
        try:
            result = real_replay(**kwargs)
        finally:
            source_snapshot.rmdir()
            backup.rename(source_snapshot)
        return result

    monkeypatch.setattr(binder, "_run_frozen_gate", swapping_replay)
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match="retained Launch8 trust object changed",
    ):
        _verify(report_root, submission_root, submission_sha)


def test_submission_ancestor_swap_restore_during_replay_cannot_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_root, submission_root, submission_sha, _bundle = _quartet(tmp_path)
    real_replay = binder._run_frozen_gate

    def swapping_replay(**kwargs: Any) -> dict:
        ancestor = submission_root.parent
        backup = ancestor.parent / f"{ancestor.name}.retained-aba"
        ancestor.rename(backup)
        ancestor.mkdir()
        try:
            result = real_replay(**kwargs)
        finally:
            ancestor.rmdir()
            backup.rename(ancestor)
        return result

    monkeypatch.setattr(binder, "_run_frozen_gate", swapping_replay)
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match="retained Launch8 trust object changed",
    ):
        _verify(report_root, submission_root, submission_sha)


def test_report_root_transient_add_delete_aba_cannot_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_root, submission_root, submission_sha, _bundle = _quartet(tmp_path)
    real_replay = binder._run_frozen_gate

    def mutating_replay(**kwargs: Any) -> dict:
        result = real_replay(**kwargs)
        transient = report_root / "TRANSIENT"
        report_root.chmod(0o755)
        transient.write_bytes(b"transient report-root authority\n")
        transient.unlink()
        report_root.chmod(0o555)
        return result

    monkeypatch.setattr(binder, "_run_frozen_gate", mutating_replay)
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match="report quartet changed during verification",
    ):
        _verify(report_root, submission_root, submission_sha)


def test_quartet_file_swap_restore_cannot_mix_report_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_root, submission_root, submission_sha, _bundle = _quartet(tmp_path)
    bundle_path = next(report_root.glob("REPORT_BUNDLE.*.json"))
    bundle_payload = bundle_path.read_bytes()
    real_replay = binder._run_frozen_gate

    def swapping_replay(**kwargs: Any) -> dict:
        backup = report_root / "quartet.retained-aba"
        report_root.chmod(0o755)
        bundle_path.rename(backup)
        bundle_path.write_bytes(b"{}\n")
        bundle_path.chmod(0o444)
        try:
            result = real_replay(**kwargs)
        finally:
            bundle_path.unlink()
            backup.rename(bundle_path)
            report_root.chmod(0o555)
        assert bundle_path.read_bytes() == bundle_payload
        return result

    monkeypatch.setattr(binder, "_run_frozen_gate", swapping_replay)
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match="report quartet changed during verification",
    ):
        _verify(report_root, submission_root, submission_sha)


def test_report_root_swap_restore_cannot_mix_retained_and_lexical_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_root, submission_root, submission_sha, _bundle = _quartet(tmp_path)
    real_replay = binder._run_frozen_gate

    def swapping_replay(**kwargs: Any) -> dict:
        backup = submission_root / "report.retained-aba"
        report_root.rename(backup)
        report_root.mkdir()
        report_root.chmod(0o555)
        try:
            result = real_replay(**kwargs)
        finally:
            report_root.rmdir()
            backup.rename(report_root)
        return result

    monkeypatch.setattr(binder, "_run_frozen_gate", swapping_replay)
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match="report quartet changed during verification|retained Launch8 trust object changed",
    ):
        _verify(report_root, submission_root, submission_sha)


def test_gate_replay_command_and_environment_are_exactly_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_root, submission_root, submission_sha, _bundle = _quartet(tmp_path)
    real_run = binder.subprocess.run
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def capturing_run(command: list[str], **kwargs: Any) -> Any:
        calls.append((list(command), dict(kwargs)))
        return real_run(command, **kwargs)

    monkeypatch.setenv("PYTHONPATH", "/hostile/pythonpath")
    monkeypatch.setenv("PYTHONHOME", "/hostile/pythonhome")
    monkeypatch.setenv("PYTHONSTARTUP", "/hostile/startup.py")
    monkeypatch.setenv("LD_PRELOAD", "/hostile/preload.so")
    monkeypatch.setattr(binder.subprocess, "run", capturing_run)
    result = _verify(report_root, submission_root, submission_sha)
    assert result["binding_published"] is False
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:4] == [binder.sys.executable, "-I", "-S", "-B"]
    assert command[4].startswith("/proc/self/fd/")
    assert command[4].endswith("/gate.py")
    assert command[5] == "--report"
    assert command[6].startswith("/proc/self/fd/")
    assert command[7] == "--manifest"
    assert command[8].startswith("/proc/self/fd/")
    assert kwargs["cwd"].startswith("/proc/self/fd/")
    assert kwargs["env"] == {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    assert kwargs["check"] is False
    assert kwargs["shell"] is False
    assert kwargs["close_fds"] is True
    passed = set(kwargs["pass_fds"])
    procfd_numbers = {
        int(Path(command[4]).parts[4]),
        int(Path(command[6]).name),
        int(Path(command[8]).name),
        int(Path(kwargs["cwd"]).name),
    }
    assert procfd_numbers <= passed
    assert len(passed) == 5
    assert kwargs["stdin"] is binder.subprocess.DEVNULL
    assert kwargs["stdout"] is binder.subprocess.PIPE
    assert kwargs["stderr"] is binder.subprocess.PIPE
    assert kwargs["timeout"] == 30


def test_frozen_verifier_dependency_drift_fails_before_report_semantics(
    tmp_path: Path,
) -> None:
    report_root, submission_root, submission_sha, _bundle = _quartet(tmp_path)
    gate_path = (
        submission_root
        / "source-snapshot"
        / "repo"
        / runtime.FROZEN_LAUNCH8_PACKAGE_RELATIVE
        / "gate.py"
    )
    gate_path.chmod(0o644)
    gate_path.write_bytes(gate_path.read_bytes() + b"\n# hostile drift\n")
    gate_path.chmod(0o444)
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match="source file set/hash differs",
    ):
        _verify(report_root, submission_root, submission_sha)


def test_durable_cancellation_latch_forbids_positive_verification(
    tmp_path: Path,
) -> None:
    report_root, submission_root, submission_sha, _bundle = _quartet(tmp_path)
    latch = submission_root / "CANCEL_REQUESTED.json"
    latch.write_text("{}\n", encoding="utf-8")
    latch.chmod(0o444)
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match="cancellation latch conflicts",
    ):
        _verify(report_root, submission_root, submission_sha)


def test_launch7_or_wrong_package_can_never_be_positive_authority(
    tmp_path: Path,
) -> None:
    report_root, submission_root, submission_sha, _bundle = _quartet(tmp_path)
    wrong = dict(runtime.FROZEN_LAUNCH8_PACKAGE_BINDING)
    wrong["package_protocol_sha256"] = _sha("launch7")
    with pytest.raises(
        binder.EngineeringPilotBindingError,
        match="expected package binding differs",
    ):
        binder.verify_engineering_pilot_report_quartet(
            report_root,
            expected_report_root=report_root,
            expected_submission_root=submission_root,
            expected_submission_sha256=submission_sha,
            expected_package_binding=wrong,
        )


def test_description_is_sealed_adapter_but_real_binding_remains_unbound() -> None:
    description = binder.adapter_description()
    assert description["adapter_state"] == "sealed_versioned_adapter"
    assert description["semantic_adapter_implemented"] is True
    assert description["frozen_source_commit"] == runtime.FROZEN_LAUNCH8_SOURCE_COMMIT
    assert description["frozen_protocol_sha256"] == (
        runtime.FROZEN_LAUNCH8_PROTOCOL_SHA256
    )
    assert description["frozen_source_inventory_sha256"] == (
        runtime.FROZEN_LAUNCH8_VERIFIER_SOURCE_INVENTORY_SHA256
    )
    assert description["binding_state"] == "unbound"
    assert description["persistent_writes_performed"] is False
    assert description["real_report_opened"] is False
    binding = json.loads(
        (runtime.PACKAGE_DIR / "accepted_engineering_pilot.binding.json").read_text()
    )
    assert binding == runtime.accepted_engineering_pilot_placeholder()
    assert binding["formal_submission_allowed"] is False
    assert binding["report_commit_file_sha256"] is None
    assert binding["binding_sha256"] is None
