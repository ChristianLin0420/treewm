from __future__ import annotations

import builtins
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from dataclasses import asdict
import stat
import subprocess
import threading

import pytest

import campaign
import cancel as cancel_cli
import engineering_pilot_binder
import runtime


def _completed(stdout: str, *, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["squeue"], returncode, stdout=stdout, stderr=stderr)


def test_default_scheduler_runner_has_exact_per_call_timeout(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured.update(kwargs)
        return _completed("")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    runtime.default_runner(["/usr/local/bin/squeue", "--version"], tmp_path, {"LANG": "C"})
    assert captured["timeout"] == 30


def test_m2a_implements_held_root_activation_but_keeps_execution_blocked() -> None:
    manifest = campaign.load_manifest()
    assert manifest["execution"]["requires_held_root_post_receipt_activation_handshake"] is True
    readiness = runtime.execution_readiness(manifest)
    assert readiness["ready"] is False
    assert readiness["held_root_activation_implemented"] is True
    assert readiness["interpreter_binary_pyvenv_path_provenance_implemented"] is True
    assert readiness["interpreter_environment_content_provenance_implemented"] is False
    assert readiness["same_stage_requeue_mutation_implemented"] is False
    assert readiness["launch7_terminal_negative_binding_state"] == (
        "sealed_authenticated_terminal_negative_no_reuse"
    )
    assert readiness["accepted_engineering_pilot_binding_state"] == "unbound"
    transaction = runtime.runtime_description(manifest)["transaction"]
    assert transaction["outer_transaction_lock_released_before_root_release"] is True
    assert transaction["queued_workers_bypass_outer_lock_after_authorization"] is True
    assert transaction["release_side_effect_serialized_with_cancellation"] is True
    assert transaction["activation_result_fsync_does_not_hold_cancel_lock"] is True


def test_exclusive_json_is_no_replace_idempotent_and_leaves_no_private_name(tmp_path: Path) -> None:
    path = tmp_path / "RECORD.json"
    value = {"schema_version": 1, "status": "sealed"}
    digest = runtime.exclusive_json(path, value)
    assert runtime.exclusive_json(path, value) == digest
    assert path.read_bytes() == (runtime.canonical_json(value) + "\n").encode()
    info = path.stat()
    assert stat.S_IMODE(info.st_mode) == 0o444
    assert info.st_nlink == 1
    assert sorted(child.name for child in tmp_path.iterdir()) == ["RECORD.json"]
    with pytest.raises(runtime.RuntimeContractError, match="differs"):
        runtime.exclusive_json(path, {"schema_version": 1, "status": "different"})


def test_exclusive_json_recovers_link_before_unlink_crash_ordinal(tmp_path: Path) -> None:
    path = tmp_path / "READY.json"
    temporary = tmp_path / ".READY.json.PUBLISHING"
    value = {"schema_version": 1, "status": "ready"}
    payload = (runtime.canonical_json(value) + "\n").encode()
    temporary.write_bytes(payload)
    temporary.chmod(0o444)
    os.link(temporary, path)
    assert path.stat().st_nlink == temporary.stat().st_nlink == 2
    runtime.exclusive_json(path, value)
    assert path.stat().st_nlink == 1
    assert not temporary.exists()
    assert json.loads(path.read_text()) == value


def test_exclusive_json_recovers_dead_partial_private_write_without_final(tmp_path: Path) -> None:
    path = tmp_path / "RECEIPT.json"
    temporary = tmp_path / ".RECEIPT.json.PUBLISHING"
    temporary.write_bytes(b'{"schema_version":')
    temporary.chmod(0o600)
    value = {"schema_version": 1, "status": "submitted"}
    runtime.exclusive_json(path, value)
    assert not temporary.exists()
    assert json.loads(path.read_text()) == value
    assert stat.S_IMODE(path.stat().st_mode) == 0o444


def test_parent_aware_reconciliation_collapses_split_array_rows() -> None:
    user = runtime.pwd.getpwuid(os.getuid()).pw_name
    name = "exp24-token-train-2000"
    comment = "treewm-exp24:" + "a" * 64
    stdout = "\n".join(
        [
            f"7000|7000|{name}|{user}|RUNNING|{comment}",
            f"7000|7001|{name}|{user}|RUNNING|{comment}",
            f"7000|7002|{name}|{user}|PENDING|{comment}",
        ]
    ) + "\n"
    assert runtime._squeue_rows(_completed(stdout), name, comment) == ["7000"]
    states = runtime._squeue_state_rows(_completed(stdout), name, comment)
    assert {row["parent_job_id"] for row in states} == {"7000"}
    assert {row["scheduler_row_id"] for row in states} == {"7000", "7001", "7002"}


def test_parent_aware_reconciliation_rejects_cross_parent_collision() -> None:
    user = runtime.pwd.getpwuid(os.getuid()).pw_name
    name = "exp24-token-train-2000"
    comment = "treewm-exp24:" + "b" * 64
    stdout = "\n".join(
        [
            f"7000|7001|{name}|{user}|RUNNING|{comment}",
            f"8000|8001|{name}|{user}|RUNNING|{comment}",
        ]
    ) + "\n"
    assert runtime._squeue_rows(_completed(stdout), name, comment) == ["7000", "8000"]


def test_initial_claim_is_privately_built_then_atomically_named(tmp_path: Path) -> None:
    submission_root = tmp_path / "submission"
    runtime.begin_transaction(submission_root, "c" * 64)
    assert submission_root.is_dir()
    assert not (tmp_path / ".submission.PRECLAIM").exists()
    assert sorted(child.name for child in submission_root.iterdir()) == ["journal", "logs"]
    journals = runtime.load_journals(submission_root)
    assert [row["event"] for row in journals] == ["CLAIMED"]
    assert journals[0]["payload"]["claim_token"] == "c" * 64
    assert stat.S_IMODE((submission_root / "journal" / "0000_CLAIMED.json").stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        runtime.begin_transaction(submission_root, "d" * 64)


def test_initial_claim_recovers_partial_private_preclaim_without_final(tmp_path: Path) -> None:
    private = tmp_path / ".submission.PRECLAIM"
    private.mkdir(mode=0o700)
    journal = private / "journal"
    journal.mkdir(mode=0o700)
    partial = journal / ".0000_CLAIMED.json.PUBLISHING"
    partial.write_bytes(b'{"schema_version":')
    partial.chmod(0o600)
    submission_root = tmp_path / "submission"
    runtime.begin_transaction(submission_root, "e" * 64)
    assert not private.exists()
    assert runtime.load_journals(submission_root)[0]["payload"]["claim_token"] == "e" * 64


def test_snapshot_manifest_race_is_rejected_before_any_scheduler_boundary(tmp_path: Path) -> None:
    manifest = campaign.load_manifest()
    inventory = runtime.m1_snapshot_inventory(campaign.REPOSITORY_ROOT)
    snapshot_root = tmp_path / "repo"
    runtime.create_source_snapshot(campaign.REPOSITORY_ROOT, snapshot_root, inventory)
    manifest_path = snapshot_root / runtime.PACKAGE_RELATIVE / "manifest.json"
    package_dir = manifest_path.parent
    raced = dict(manifest)
    raced["injected_race_probe"] = "different authenticated manifest bytes"
    package_dir.chmod(0o755)
    manifest_path.chmod(0o600)
    manifest_path.unlink()
    inventory[str(runtime.PACKAGE_RELATIVE / "manifest.json")] = runtime.exclusive_json(
        manifest_path,
        raced,
    )
    package_dir.chmod(0o555)
    with pytest.raises(runtime.RuntimeContractError, match="in-memory manifest"):
        runtime.verify_submission_snapshot_identity(manifest, snapshot_root, inventory)


def test_journal_residue_sweep_repairs_link_point_and_discards_unpublished(tmp_path: Path) -> None:
    journal = tmp_path / "journal"
    journal.mkdir(mode=0o700)
    final = journal / "0000_CLAIMED.json"
    linked = journal / ".0000_CLAIMED.json.PUBLISHING"
    linked.write_text("{}\n")
    linked.chmod(0o444)
    os.link(linked, final)
    unpublished = journal / ".0001_SNAPSHOT_SEALED.json.PUBLISHING"
    unpublished.write_text('{"partial":')
    unpublished.chmod(0o600)
    repaired = runtime.repair_publication_residues(
        journal,
        allowed_final=runtime.re.compile(r"[0-9]{4}_[A-Z0-9_]+\.json"),
    )
    assert repaired == [
        ".0000_CLAIMED.json.PUBLISHING",
        ".0001_SNAPSHOT_SEALED.json.PUBLISHING",
    ]
    assert final.stat().st_nlink == 1
    assert not unpublished.exists()


def test_queued_barrier_times_out_under_exclusive_lifetime_lock_then_shares(tmp_path: Path) -> None:
    submission_root = tmp_path / "submission"
    lock_context = runtime.transaction_recovery_lock(submission_root)
    lock_context.__enter__()
    try:
        with pytest.raises(runtime.RuntimeContractError, match="timed out"):
            with runtime.queued_transaction_barrier(
                submission_root,
                max_attempts=1,
                sleeper=lambda _seconds: None,
            ):
                raise AssertionError("barrier acquired while exclusive submit lock was held")
    finally:
        lock_context.__exit__(None, None, None)
    with runtime.queued_transaction_barrier(
        submission_root,
        max_attempts=1,
        sleeper=lambda _seconds: None,
    ) as evidence:
        assert evidence["attempt"] == 1


def _accepted_common(
    *,
    job_id: str,
    name: str,
    comment: str,
    state: str,
    dependency: str,
    partition: str,
    workdir: Path,
    stdout: str,
    command: str,
    requeue: str,
) -> list[str]:
    return [
        f"JobId={job_id}",
        f"JobName={name}",
        f"Comment={comment}",
        f"JobState={state}",
        f"Dependency={dependency}",
        "Account=edgeai_tao-ptm_image-foundation-model-clip",
        "QOS=normal",
        f"Partition={partition}",
        "NumCPUs=12",
        "CPUs/Task=12",
        "NumNodes=1-1" if state == "PENDING" else "NumNodes=1",
        "NumTasks=1",
        "MinMemoryNode=64G",
        "TimeLimit=04:00:00",
        f"WorkDir={workdir}",
        f"StdOut={stdout}",
        f"StdErr={stdout}",
        f"Command={command}",
        f"Requeue={requeue}",
        "Power=",
    ]


def test_site_normalized_scalar_cpu_accepted_record_omits_array_and_tres(tmp_path: Path) -> None:
    manifest = campaign.load_manifest()
    submission_root = tmp_path / "submission"
    snapshot_root = tmp_path / "snapshot"
    records = runtime.scheduler_commands(
        manifest,
        snapshot_root,
        submission_root,
        "a" * 64,
        {"train_2000": "7000"},
        through_index=1,
    )
    record = records[1]
    output = str(submission_root / "logs" / "gate_2000_8000.out")
    fields = _accepted_common(
        job_id="8000",
        name=record["job_name"],
        comment=record["comment"],
        state="PENDING",
        dependency="afterok:7000_*(unfulfilled)",
        partition="cpu",
        workdir=snapshot_root,
        stdout=output,
        command=next(value for value in record["command"] if value.endswith(".slurm")),
        requeue="0",
    )
    fields.append("KillOInInvalidDependent=Yes")
    normalized = runtime.validate_accepted_job_stdout(
        " ".join(fields) + "\n",
        job_id="8000",
        name=record["job_name"],
        comment=record["comment"],
        predecessor_job_id="7000",
        predecessor_elements=40,
        manifest=manifest,
        node=record["node"],
        submit_command=record["command"],
        cwd=snapshot_root,
    )
    assert normalized == {
        "record_count": 1,
        "array_task_ids": [],
        "states": ["PENDING"],
        "reasons": ["<absent>"],
        "root_lifecycle": "not_root",
        "dependency": "afterok:7000_*(unfulfilled)",
        "kill_on_invalid_dependency": "Yes",
    }


def test_site_normalized_compact_gpu_array_accepts_no_val_output(tmp_path: Path) -> None:
    manifest = campaign.load_manifest()
    submission_root = tmp_path / "submission"
    snapshot_root = tmp_path / "snapshot"
    record = runtime.scheduler_commands(
        manifest,
        snapshot_root,
        submission_root,
        "b" * 64,
        through_index=0,
    )[0]
    output = str(submission_root / "logs" / "train_2000_7000_4294967294.out")
    fields = _accepted_common(
        job_id="7000",
        name=record["job_name"],
        comment=record["comment"],
        state="PENDING",
        dependency="(null)",
        partition="polar4,polar3,polar,grizzly",
        workdir=snapshot_root,
        stdout=output,
        command=next(value for value in record["command"] if value.endswith(".slurm")),
        requeue="1",
    )
    fields.extend(
        [
            "Reason=JobHeldUser",
            "ArrayJobId=7000",
            "ArrayTaskId=0-39%40",
            "ArrayTaskThrottle=40",
            "TresPerNode=gres:gpu:1",
        ]
    )
    normalized = runtime.validate_accepted_job_stdout(
        " ".join(fields) + "\n",
        job_id="7000",
        name=record["job_name"],
        comment=record["comment"],
        predecessor_job_id=None,
        predecessor_elements=None,
        manifest=manifest,
        node=record["node"],
        submit_command=record["command"],
        cwd=snapshot_root,
    )
    assert normalized["array_task_ids"] == list(range(40))
    assert normalized["kill_on_invalid_dependency"] == "disabled_or_absent"


def test_site_normalized_split_gpu_array_binds_parent_and_exact_task_cover(tmp_path: Path) -> None:
    manifest = campaign.load_manifest()
    submission_root = tmp_path / "submission"
    snapshot_root = tmp_path / "snapshot"
    record = runtime.scheduler_commands(
        manifest,
        snapshot_root,
        submission_root,
        "c" * 64,
        through_index=0,
    )[0]
    script = next(value for value in record["command"] if value.endswith(".slurm"))
    lines: list[str] = []
    for task in range(40):
        scheduler_id = "7000" if task == 39 else str(7100 + task)
        output = str(submission_root / "logs" / f"train_2000_7000_{task}.out")
        fields = _accepted_common(
            job_id=scheduler_id,
            name=record["job_name"],
            comment=record["comment"],
            state="RUNNING",
            dependency="(null)",
            partition="polar4",
            workdir=snapshot_root,
            stdout=output,
            command=script,
            requeue="1",
        )
        fields.extend(
            [
                "ArrayJobId=7000",
                f"ArrayTaskId={task}",
                "ArrayTaskThrottle=40",
                "TresPerNode=gres:gpu:1",
            ]
        )
        lines.append(" ".join(fields))
    normalized = runtime.validate_accepted_job_stdout(
        "\n".join(lines) + "\n",
        job_id="7000",
        name=record["job_name"],
        comment=record["comment"],
        predecessor_job_id=None,
        predecessor_elements=None,
        manifest=manifest,
        node=record["node"],
        submit_command=record["command"],
        cwd=snapshot_root,
        root_lifecycle="released",
    )
    assert normalized["record_count"] == 40
    assert normalized["array_task_ids"] == list(range(40))


@pytest.mark.parametrize(
    ("role", "index", "elements"),
    [
        ("train_25000", 2, 40),
        ("heldout_eval", 8, 200),
        ("formal_report", 10, 1),
    ],
)
def test_site_normalized_remaining_dependency_and_geometry_forms(
    tmp_path: Path,
    role: str,
    index: int,
    elements: int,
) -> None:
    manifest = campaign.load_manifest()
    nodes = campaign.scheduler_dag(manifest)
    snapshot_root = tmp_path / "snapshot"
    submission_root = tmp_path / "submission"
    predecessor_ids = {node.name: str(7000 + offset) for offset, node in enumerate(nodes[:index])}
    record = runtime.scheduler_commands(
        manifest,
        snapshot_root,
        submission_root,
        "d" * 64,
        predecessor_ids,
        through_index=index,
    )[index]
    node = nodes[index]
    assert node.name == role and node.elements == elements and node.dependency is not None
    job_id = str(9000 + index)
    predecessor_id = predecessor_ids[node.dependency]
    predecessor = nodes[index - 1]
    dependency = runtime.expected_dependency_string(predecessor_id, predecessor.elements)
    gpu = role.startswith("train_") or role == "heldout_eval"
    output_template = next(
        value.split("=", 1)[1]
        for value in record["command"]
        if value.startswith("--output=")
    )
    if elements > 1:
        output = output_template.replace("%A", job_id).replace("%a", "4294967294")
    else:
        output = output_template.replace("%j", job_id)
    fields = _accepted_common(
        job_id=job_id,
        name=record["job_name"],
        comment=record["comment"],
        state="PENDING",
        dependency=dependency,
        partition=(manifest["execution"]["gpu_partitions"] if gpu else "cpu"),
        workdir=snapshot_root,
        stdout=output,
        command=next(value for value in record["command"] if value.endswith(".slurm")),
        requeue="1" if gpu else "0",
    )
    fields.append("KillOInInvalidDependent=Yes")
    if elements > 1:
        array = manifest["execution"]["training_array"] if elements == 40 else manifest["execution"]["heldout_array"]
        fields.extend(
            [
                f"ArrayJobId={job_id}",
                f"ArrayTaskId={array}",
                f"ArrayTaskThrottle={array.rsplit('%', 1)[1]}",
                "TresPerNode=gres:gpu:1",
            ]
        )
    normalized = runtime.validate_accepted_job_stdout(
        " ".join(fields) + "\n",
        job_id=job_id,
        name=record["job_name"],
        comment=record["comment"],
        predecessor_job_id=predecessor_id,
        predecessor_elements=predecessor.elements,
        manifest=manifest,
        node=record["node"],
        submit_command=record["command"],
        cwd=snapshot_root,
    )
    assert normalized["record_count"] == 1
    assert normalized["dependency"] == dependency
    assert normalized["array_task_ids"] == (list(range(elements)) if elements > 1 else [])


class FakeScheduler:
    def __init__(
        self,
        manifest: dict,
        submission_sha256: str,
        *,
        fail_role: str | None = None,
        fail_observe_role: str | None = None,
        fail_root_on_precommit_observation: bool = False,
        unparseable_stdout_role: str | None = None,
        unparseable_stdout: str = "",
    ) -> None:
        self.manifest = manifest
        self.submission_sha256 = submission_sha256
        self.nodes = {node.name: node for node in campaign.scheduler_dag(manifest)}
        self.names = runtime._node_job_names(list(self.nodes.values()), submission_sha256[:16])
        self.roles_by_name = {value: key for key, value in self.names.items()}
        self.jobs: dict[str, dict] = {}
        self.next_id = 9100
        self.calls: list[list[str]] = []
        self.fail_role = fail_role
        self.fail_observe_role = fail_observe_role
        self.failed = False
        self.fail_root_on_precommit_observation = fail_root_on_precommit_observation
        self.unparseable_stdout_role = unparseable_stdout_role
        self.unparseable_stdout = unparseable_stdout
        self.observation_counts: dict[str, int] = {}

    def _role(self, command: list[str]) -> str:
        raw = next(value for value in command if value.startswith("--job-name="))
        return self.roles_by_name[raw.split("=", 1)[1]]

    def _test_only(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        underlying = [command[0], *command[3:]]
        role = self._role(underlying)
        node = self.nodes[role]
        dependency = next(
            (value.split("=", 1)[1] for value in underlying if value.startswith("--dependency=")),
            None,
        )
        record = {"node": asdict(node), "command": underlying}
        options, partitions = runtime.expected_test_options(
            self.manifest,
            record,
            dependency=dependency,
        )
        options.update({"test-only": "set", "verbose": "3"})
        lines = ["sbatch: defined options"]
        lines.extend(f"sbatch: {key} : {value}" for key, value in sorted(options.items()))
        lines.append("sbatch: end of defined options")
        lines.append(
            "sbatch: Job 999999 to start at now using "
            f"{self.manifest['execution']['cpus_per_task']} processors on nodes fake in partition {partitions[0]}"
        )
        return _completed("", stderr="\n".join(lines) + "\n")

    def _sbatch(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        role = self._role(command)
        job_id = str(self.next_id)
        self.next_id += 1
        self.jobs[job_id] = {
            "role": role,
            "command": command,
            "state": "PENDING",
            "canceled": False,
            "held": "--hold" in command,
        }
        if role == self.fail_role and not self.failed:
            self.failed = True
            return _completed("", returncode=1, stderr="injected accepted-but-error response")
        if role == self.unparseable_stdout_role:
            return _completed(self.unparseable_stdout)
        return _completed(job_id + "\n")

    def _squeue(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        name = next(value.split("=", 1)[1] for value in command if value.startswith("--name="))
        rows: list[str] = []
        user = runtime.pwd.getpwuid(os.getuid()).pw_name
        for job_id, job in self.jobs.items():
            job_name = next(value.split("=", 1)[1] for value in job["command"] if value.startswith("--job-name="))
            if job_name != name or job["canceled"]:
                continue
            comment = next(value.split("=", 1)[1] for value in job["command"] if value.startswith("--comment="))
            rows.append(f"{job_id}|{job_id}|{job_name}|{user}|{job['state']}|{comment}")
        return _completed("\n".join(rows) + ("\n" if rows else ""))

    def _scontrol(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[1] == "release":
            job_id = command[2]
            job = self.jobs[job_id]
            if job["canceled"]:
                return _completed("", returncode=1, stderr="job is canceled")
            job["held"] = False
            return _completed("")
        assert command[1:3] == ["show", "job"]
        job_id = command[3]
        job = self.jobs[job_id]
        role = job["role"]
        self.observation_counts[role] = self.observation_counts.get(role, 0) + 1
        if (
            (
                role == self.fail_observe_role
                and self.observation_counts[role] == 1
            )
            or (
                role == "train_2000"
                and self.fail_root_on_precommit_observation
                and self.observation_counts[role] >= 2
            )
        ):
            job["state"] = "FAILED"
        node = self.nodes[role]
        submit = job["command"]
        name = next(value.split("=", 1)[1] for value in submit if value.startswith("--job-name="))
        comment = next(value.split("=", 1)[1] for value in submit if value.startswith("--comment="))
        output_template = next(value.split("=", 1)[1] for value in submit if value.startswith("--output="))
        dependency_raw = next(
            (value.split("=", 1)[1] for value in submit if value.startswith("--dependency=")),
            None,
        )
        if dependency_raw is None:
            dependency = "(null)"
        else:
            predecessor = self.nodes[str(node.dependency)]
            predecessor_id = dependency_raw.split(":", 1)[1]
            dependency = runtime.expected_dependency_string(predecessor_id, predecessor.elements)
        gpu = role.startswith("train_") or role == "heldout_eval"
        if node.elements > 1:
            output = output_template.replace("%A", job_id).replace("%a", "4294967294")
        else:
            output = output_template.replace("%j", job_id)
        fields = _accepted_common(
            job_id=job_id,
            name=name,
            comment=comment,
            state=job["state"],
            dependency=dependency,
            partition=(self.manifest["execution"]["gpu_partitions"] if gpu else "cpu"),
            workdir=Path(command and submit[-(5 if role.startswith(('train_', 'gate_')) else 4)]).parents[2] if False else Path.cwd(),
            stdout=output,
            command=next(value for value in submit if value.endswith(".slurm")),
            requeue="1" if gpu else "0",
        )
        if role == "train_2000":
            fields.append("Reason=JobHeldUser" if job["held"] else "Reason=Resources")
        # SchedulerBoundary supplies cwd separately; replace the synthetic WorkDir
        # in runner() after this helper builds the otherwise exact fixture.
        if dependency_raw is not None:
            fields.append("KillOInInvalidDependent=Yes")
        if node.elements > 1:
            array = self.manifest["execution"]["training_array"] if node.elements == 40 else self.manifest["execution"]["heldout_array"]
            fields.extend(
                [
                    f"ArrayJobId={job_id}",
                    f"ArrayTaskId={array}",
                    f"ArrayTaskThrottle={array.rsplit('%', 1)[1]}",
                ]
            )
        if gpu:
            fields.append("TresPerNode=gres:gpu:1")
        return _completed(" ".join(fields) + "\n")

    def __call__(
        self,
        raw_command,
        cwd: Path,
        _environment,
        _inherited_fds,
    ) -> subprocess.CompletedProcess[str]:
        command = list(raw_command)
        self.calls.append(command)
        executable = Path(command[0]).name
        if executable == "sbatch" and "--test-only" in command:
            return self._test_only(command)
        if executable == "sbatch":
            return self._sbatch(command)
        if executable == "squeue":
            return self._squeue(command)
        if executable == "scontrol":
            result = self._scontrol(command)
            result.stdout = result.stdout.replace(f"WorkDir={Path.cwd()}", f"WorkDir={cwd}")
            return result
        if executable == "scancel":
            assert command[1] == "--quiet"
            for job_id in command[2:]:
                if job_id in self.jobs:
                    self.jobs[job_id]["canceled"] = True
            return _completed("")
        raise AssertionError(f"unexpected fake scheduler command: {command}")


def _transaction_fixture(
    tmp_path: Path,
    *,
    fail_role: str | None = None,
    fail_observe_role: str | None = None,
    fail_root_on_precommit_observation: bool = False,
    unparseable_stdout_role: str | None = None,
    unparseable_stdout: str = "",
):
    manifest = campaign.load_manifest()
    submission_root = tmp_path / "submission"
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    runtime.begin_transaction(submission_root, "f" * 64)
    submission_sha = "1" * 64
    fake = FakeScheduler(
        manifest,
        submission_sha,
        fail_role=fail_role,
        fail_observe_role=fail_observe_role,
        fail_root_on_precommit_observation=fail_root_on_precommit_observation,
        unparseable_stdout_role=unparseable_stdout_role,
        unparseable_stdout=unparseable_stdout,
    )
    stable = {"schema_version": 1, "kind": "fake-stable-control-plane"}
    boundary = runtime.SchedulerBoundary(
        runner=fake,
        observer=lambda: stable,
        expected=stable,
    )
    return manifest, submission_root, snapshot_root, submission_sha, fake, boundary


def test_fake_scheduler_transaction_traverses_exact_eleven_nodes(tmp_path: Path) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _transaction_fixture(tmp_path)
    receipt = runtime.submit_dag_transaction(
        manifest,
        submission_root=submission_root,
        snapshot_root=snapshot_root,
        submission_sha256=submission_sha,
        boundary=boundary,
    )
    assert [row["role"] for row in receipt["jobs"]] == [node.name for node in campaign.scheduler_dag(manifest)]
    assert len({row["job_id"] for row in receipt["jobs"]}) == 11
    assert len([call for call in fake.calls if Path(call[0]).name == "sbatch" and "--test-only" in call]) == 11
    assert len([call for call in fake.calls if Path(call[0]).name == "sbatch" and "--test-only" not in call]) == 11
    assert len([call for call in fake.calls if Path(call[0]).name == "scontrol"]) == 22
    assert not [call for call in fake.calls if Path(call[0]).name == "scancel"]
    journals = runtime.load_journals(submission_root)
    assert len([row for row in journals if row["event"].endswith("_SUBMIT_TESTED")]) == 11
    assert journals[-1]["event"] == "READY_TO_COMMIT"


@pytest.mark.parametrize(
    ("role", "stdout"),
    [("train_2000", ""), ("formal_report", "unexpected scheduler banner\n")],
)
def test_unparseable_or_lost_sbatch_stdout_commits_only_exact_reconciled_id(
    tmp_path: Path,
    role: str,
    stdout: str,
) -> None:
    manifest, submission_root, snapshot_root, submission_sha, _fake, boundary = _transaction_fixture(
        tmp_path,
        unparseable_stdout_role=role,
        unparseable_stdout=stdout,
    )
    receipt = runtime.submit_dag_transaction(
        manifest,
        submission_root=submission_root,
        snapshot_root=snapshot_root,
        submission_sha256=submission_sha,
        boundary=boundary,
    )
    row = next(item for item in receipt["jobs"] if item["role"] == role)
    assert row["submit"]["response_mode"] == "exact_name_comment_reconciliation_after_unparseable_stdout"
    assert row["submit"]["reconciled_job_ids"] == [row["job_id"]]


@pytest.mark.parametrize(
    "failed_role",
    [
        "train_2000", "gate_2000", "train_25000", "gate_25000",
        "train_100000", "gate_100000", "train_1000000", "gate_1000000",
        "heldout_eval", "aggregate", "formal_report",
    ],
)
def test_fake_scheduler_partial_acceptance_rolls_back_every_parent(
    tmp_path: Path,
    failed_role: str,
) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _transaction_fixture(
        tmp_path,
        fail_role=failed_role,
    )
    with pytest.raises(runtime.SchedulerTransactionError):
        runtime.submit_dag_transaction(
            manifest,
            submission_root=submission_root,
            snapshot_root=snapshot_root,
            submission_sha256=submission_sha,
            boundary=boundary,
        )
    assert not (submission_root / "SUBMISSION_RECEIPT.json").exists()
    cancel_calls = [call for call in fake.calls if Path(call[0]).name == "scancel"]
    assert len(cancel_calls) == 1
    assert cancel_calls[0][1] == "--quiet"
    accepted = sorted(fake.jobs, key=int)
    assert sorted(cancel_calls[0][2:], key=int) == accepted
    assert all(fake.jobs[job_id]["canceled"] for job_id in accepted)
    events = [row["event"] for row in runtime.load_journals(submission_root)]
    assert "ROLLBACK_CANCELED" in events and "ABORTED" in events


@pytest.mark.parametrize(
    "failed_role",
    [
        "train_2000", "gate_2000", "train_25000", "gate_25000",
        "train_100000", "gate_100000", "train_1000000", "gate_1000000",
        "heldout_eval", "aggregate", "formal_report",
    ],
)
def test_fake_scheduler_observation_failure_at_every_role_rolls_back(
    tmp_path: Path,
    failed_role: str,
) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _transaction_fixture(
        tmp_path,
        fail_observe_role=failed_role,
    )
    with pytest.raises(runtime.SchedulerTransactionError, match="state differs"):
        runtime.submit_dag_transaction(
            manifest,
            submission_root=submission_root,
            snapshot_root=snapshot_root,
            submission_sha256=submission_sha,
            boundary=boundary,
        )
    assert not (submission_root / "SUBMISSION_RECEIPT.json").exists()
    assert all(row["canceled"] for row in fake.jobs.values())


def test_pre_ready_budget_expiry_rolls_back_without_receipt(monkeypatch, tmp_path: Path) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _transaction_fixture(tmp_path)
    original = runtime._require_pre_receipt_budget

    def expire_before_receipt(execution, started, phase):
        if phase == "before_ready_to_commit":
            raise runtime.RuntimeContractError(
                "pre-receipt transaction exceeded its 600s budget at before_ready_to_commit"
            )
        return original(execution, started, phase)

    monkeypatch.setattr(runtime, "_require_pre_receipt_budget", expire_before_receipt)
    with pytest.raises(runtime.SchedulerTransactionError, match="600s budget"):
        runtime.submit_dag_transaction(
            manifest,
            submission_root=submission_root,
            snapshot_root=snapshot_root,
            submission_sha256=submission_sha,
            boundary=boundary,
        )
    assert not (submission_root / "SUBMISSION_RECEIPT.json").exists()
    assert all(row["canceled"] for row in fake.jobs.values())
    events = [row["event"] for row in runtime.load_journals(submission_root)]
    assert "ROLLBACK_CANCELED" in events and events[-1] == "ABORTED"


def test_failed_root_before_ready_is_reobserved_and_full_dag_rolls_back(tmp_path: Path) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _transaction_fixture(
        tmp_path,
        fail_root_on_precommit_observation=True,
    )
    with pytest.raises(runtime.SchedulerTransactionError, match="JobState|state differs"):
        runtime.submit_dag_transaction(
            manifest,
            submission_root=submission_root,
            snapshot_root=snapshot_root,
            submission_sha256=submission_sha,
            boundary=boundary,
        )
    assert fake.observation_counts["train_2000"] == 2
    assert not (submission_root / "SUBMISSION_RECEIPT.json").exists()
    assert all(row["canceled"] for row in fake.jobs.values())
    assert "READY_TO_COMMIT" not in [
        row["event"] for row in runtime.load_journals(submission_root)
    ]


def test_ready_journal_write_ambiguity_never_rolls_back(monkeypatch, tmp_path: Path) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _transaction_fixture(tmp_path)
    original = runtime.append_journal

    def fail_after_ready(root, ordinal, event, payload):
        result = original(root, ordinal, event, payload)
        if event == "READY_TO_COMMIT":
            raise OSError("injected fsync return ambiguity after durable READY")
        return result

    monkeypatch.setattr(runtime, "append_journal", fail_after_ready)
    with pytest.raises(runtime.CommitRecoveryRequired, match="ambiguous"):
        runtime.submit_dag_transaction(
            manifest,
            submission_root=submission_root,
            snapshot_root=snapshot_root,
            submission_sha256=submission_sha,
            boundary=boundary,
        )
    assert "READY_TO_COMMIT" in [row["event"] for row in runtime.load_journals(submission_root)]
    assert not [call for call in fake.calls if Path(call[0]).name == "scancel"]
    assert not (submission_root / "SUBMISSION_RECEIPT.json").exists()


def test_receipt_write_failure_after_ready_never_rolls_back(monkeypatch, tmp_path: Path) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _transaction_fixture(tmp_path)
    original = runtime.exclusive_json

    def fail_receipt(path, value, *, mode=0o444):
        if path.name == "SUBMISSION_RECEIPT.json":
            raise OSError("injected receipt publication failure")
        return original(path, value, mode=mode)

    monkeypatch.setattr(runtime, "exclusive_json", fail_receipt)
    with pytest.raises(runtime.CommitRecoveryRequired, match="ambiguous"):
        runtime.submit_dag_transaction(
            manifest,
            submission_root=submission_root,
            snapshot_root=snapshot_root,
            submission_sha256=submission_sha,
            boundary=boundary,
        )
    assert "READY_TO_COMMIT" in [row["event"] for row in runtime.load_journals(submission_root)]
    assert not [call for call in fake.calls if Path(call[0]).name == "scancel"]


def _no_scheduler_boundary() -> runtime.SchedulerBoundary:
    def forbidden(*_args):
        raise AssertionError("pre-contract recovery contacted the scheduler")

    return runtime.SchedulerBoundary(runner=forbidden, observer=lambda: {}, expected={})


def test_precontract_recovery_result_is_exact_and_retry_is_authenticated(tmp_path: Path) -> None:
    manifest = campaign.load_manifest()
    submission_root = tmp_path / "submission"
    runtime.begin_transaction(submission_root, "2" * 64)
    first = runtime.recover_transaction(
        manifest,
        submission_root=submission_root,
        boundary=_no_scheduler_boundary(),
    )
    assert first["status"] == "aborted_before_scheduler_contract"
    assert first["scheduler_calls"] == 0
    assert first["new_jobs_created"] == 0
    assert first["journal_ledger_sha256"] == runtime.stable_hash(first["journal_ledger"])
    second = runtime.recover_transaction(
        manifest,
        submission_root=submission_root,
        boundary=_no_scheduler_boundary(),
    )
    assert second["retry"] is True
    assert second["recovery_result_sha256"] == first["recovery_result_sha256"]


def test_forged_loose_recovery_result_cannot_suppress_reconciliation(tmp_path: Path) -> None:
    manifest = campaign.load_manifest()
    submission_root = tmp_path / "submission"
    runtime.begin_transaction(submission_root, "3" * 64)
    runtime.exclusive_json(
        submission_root / "TRANSACTION_RECOVERY_RESULT.json",
        {
            "schema_version": 1,
            "campaign_id": runtime.CAMPAIGN_ID,
            "submission_root": str(submission_root),
            "status": "aborted_before_scheduler_contract",
        },
    )
    with pytest.raises(runtime.RuntimeContractError, match="schema differs"):
        runtime.recover_transaction(
            manifest,
            submission_root=submission_root,
            boundary=_no_scheduler_boundary(),
        )


def test_no_contract_recovery_rejects_any_scheduler_acceptance_evidence(tmp_path: Path) -> None:
    manifest = campaign.load_manifest()
    submission_root = tmp_path / "submission"
    runtime.begin_transaction(submission_root, "6" * 64)
    runtime.append_journal(
        submission_root,
        11,
        "TRAIN_2000_ACCEPTED",
        {"role": "train_2000", "job_id": "987654"},
    )
    with pytest.raises(runtime.RuntimeContractError, match="pre-contract journal inventory"):
        runtime.recover_transaction(
            manifest,
            submission_root=submission_root,
            boundary=_no_scheduler_boundary(),
        )
    assert not (submission_root / "TRANSACTION_RECOVERY_RESULT.json").exists()


_RECOVERY_ROLES = [
    "train_2000", "gate_2000", "train_25000", "gate_25000",
    "train_100000", "gate_100000", "train_1000000", "gate_1000000",
    "heldout_eval", "aggregate", "formal_report",
]
_RECOVERY_DURABLE_POINTS = ["CONTRACT_SEALED"] + [
    f"{role.upper()}_{suffix}"
    for role in _RECOVERY_ROLES
    for suffix in ("SUBMIT_TESTED", "ACCEPTED", "OBSERVED")
] + [f"{role.upper()}_PRECOMMIT_REOBSERVED" for role in _RECOVERY_ROLES]


@pytest.mark.parametrize("durable_point", _RECOVERY_DURABLE_POINTS)
def test_precommit_recovery_at_every_durable_transaction_ordinal_is_idempotent(
    tmp_path: Path,
    durable_point: str,
) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _prepared_submission_fixture(tmp_path)
    nodes = campaign.scheduler_dag(manifest)
    jobs_by_role: dict[str, str] = {}
    events: list[tuple[int, str, str]] = []
    for index, node in enumerate(nodes):
        events.extend(
            [
                (10 + 3 * index, f"{node.name.upper()}_SUBMIT_TESTED", node.name),
                (11 + 3 * index, f"{node.name.upper()}_ACCEPTED", node.name),
                (12 + 3 * index, f"{node.name.upper()}_OBSERVED", node.name),
            ]
        )
    events.extend(
        (50 + index, f"{node.name.upper()}_PRECOMMIT_REOBSERVED", node.name)
        for index, node in enumerate(nodes)
    )
    if durable_point != "CONTRACT_SEALED":
        for ordinal, event, role in events:
            node_index = _RECOVERY_ROLES.index(role)
            if event.endswith("_ACCEPTED"):
                record = runtime.scheduler_commands(
                    manifest,
                    snapshot_root,
                    submission_root,
                    submission_sha,
                    jobs_by_role,
                    through_index=node_index,
                )[node_index]
                job_id = fake._sbatch(record["command"]).stdout.strip()
                jobs_by_role[role] = job_id
                payload = {"role": role, "job_id": job_id}
            elif event.endswith("_OBSERVED") or event.endswith("_PRECOMMIT_REOBSERVED"):
                payload = {"role": role, "job_id": jobs_by_role[role]}
            else:
                payload = {"role": role}
            runtime.append_journal(submission_root, ordinal, event, payload)
            if event == durable_point:
                break
    result = runtime.recover_transaction(
        manifest,
        submission_root=submission_root,
        boundary=boundary,
    )
    assert result["status"] == "precommit_transaction_reconciled_and_aborted"
    assert result["new_jobs_created"] == 0
    assert not (submission_root / "SUBMISSION_RECEIPT.json").exists()
    calls_after_first = list(boundary.calls)
    retry = runtime.recover_transaction(
        manifest,
        submission_root=submission_root,
        boundary=boundary,
    )
    assert retry["retry"] is True
    assert boundary.calls == calls_after_first


def _prepared_submission_fixture(tmp_path: Path):
    manifest = campaign.load_manifest()
    submission_root = tmp_path / "submission"
    with runtime.transaction_recovery_lock(submission_root):
        runtime.begin_transaction(submission_root, "4" * 64)
    snapshot_parent = submission_root / "source-snapshot"
    runtime._mkdir_exact(snapshot_parent, 0o700, "test snapshot parent")
    snapshot_root = snapshot_parent / "repo"
    inventory = runtime.m1_snapshot_inventory(campaign.REPOSITORY_ROOT)
    runtime.create_source_snapshot(campaign.REPOSITORY_ROOT, snapshot_root, inventory)
    snapshot_parent.chmod(0o555)
    runtime._fsync_directory(snapshot_parent)
    runtime._fsync_directory(submission_root)
    runtime.append_journal(
        submission_root,
        1,
        "SNAPSHOT_SEALED",
        {"inventory_sha256": runtime.stable_hash(inventory), "file_count": len(inventory)},
    )
    stable = {"schema_version": 1, "kind": "fake-stable-control-plane"}
    fallback = runtime.scheduler_fallback_binding(b"fake sealed Slurm configuration\n", stable)
    contract = runtime.submission_contract(
        manifest,
        submission_root=submission_root,
        snapshot_root=snapshot_root,
        snapshot_inventory=inventory,
        control_plane=stable,
        scheduler_preclaim={
            "schema_version": 1,
            "status": "fake_zero_job_preclaim_for_unit_test",
            "scheduler_jobs_created": 0,
        },
        scheduler_fallback=fallback,
    )
    submission_sha = runtime.exclusive_json(submission_root / "SUBMISSION_CONTRACT.json", contract)
    runtime.append_journal(submission_root, 2, "CONTRACT_SEALED", {"submission_sha256": submission_sha})
    fake = FakeScheduler(manifest, submission_sha)
    boundary = runtime.SchedulerBoundary(runner=fake, observer=lambda: stable, expected=stable)
    return manifest, submission_root, snapshot_root, submission_sha, fake, boundary


def _sealed_submission_fixture(tmp_path: Path):
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _prepared_submission_fixture(tmp_path)
    receipt = runtime.submit_dag_transaction(
        manifest,
        submission_root=submission_root,
        snapshot_root=snapshot_root,
        submission_sha256=submission_sha,
        boundary=boundary,
    )
    runtime.activate_root_after_receipt(
        submission_root,
        boundary=boundary,
    )
    return manifest, submission_root, snapshot_root, submission_sha, receipt, fake, boundary


def test_submission_contract_joins_exact_negative_positive_adapter_and_interpreter_leaves(
    tmp_path: Path,
) -> None:
    manifest, submission_root, _snapshot_root, _sha, _fake, _boundary = (
        _prepared_submission_fixture(tmp_path)
    )
    contract, _digest = runtime.authenticated_immutable_json(
        submission_root / "SUBMISSION_CONTRACT.json",
        "test submission contract",
    )
    assert contract["interpreter_provenance"]["source_manifest_sha256"] == (
        campaign.manifest_sha256(manifest)
    )
    negative = contract["launch7_negative_binding"]
    assert negative["negative_binding_sha256"] == (
        "629610c2bb677f53ee3acb75a8bcd1e3089bee78a4c43600a944e4290f5148bd"
    )
    assert negative["evidence_file_sha256"] == (
        contract["snapshot_inventory"][str(campaign.LAUNCH7_NEGATIVE_EVIDENCE_RELATIVE)]
    )
    adapter = contract["engineering_pilot_adapter_interface"]
    assert adapter["adapter_file_sha256"] == contract["snapshot_inventory"][
        str(runtime.PACKAGE_RELATIVE / "engineering_pilot_binder.py")
    ]
    assert adapter["adapter_runtime_file_sha256"] == contract["snapshot_inventory"][
        str(runtime.PACKAGE_RELATIVE / "runtime.py")
    ]
    positive = contract["accepted_engineering_pilot_binding"]
    assert positive["adapter_file_sha256"] is None
    assert positive["adapter_runtime_file_sha256"] is None
    assert positive["report_commit_file_sha256"] is None
    assert positive["binding_sha256"] is None


def test_full_receipt_reauthenticates_snapshot_dag_tests_and_observations(tmp_path: Path) -> None:
    _manifest, submission_root, _snapshot_root, submission_sha, receipt, _fake, _boundary = _sealed_submission_fixture(tmp_path)
    authenticated, digest = runtime.load_receipt(submission_root)
    assert authenticated == receipt
    assert authenticated["submission_sha256"] == submission_sha
    assert len(authenticated["jobs"]) == 11
    receipt_path = submission_root / "SUBMISSION_RECEIPT.json"
    receipt_path.chmod(0o644)
    with pytest.raises(runtime.RuntimeContractError, match="ownership/mode"):
        runtime.load_receipt(submission_root)
    receipt_path.chmod(0o444)
    assert runtime.load_receipt(submission_root)[1] == digest


def test_full_receipt_rejects_extra_journal_and_malformed_foundation(tmp_path: Path) -> None:
    _manifest, submission_root, _snapshot_root, _sha, _receipt, _fake, _boundary = _sealed_submission_fixture(tmp_path)
    runtime.append_journal(submission_root, 91, "UNEXPECTED", {"extra": True})
    with pytest.raises(runtime.RuntimeContractError, match="journal inventory"):
        runtime.load_receipt(submission_root)

    extra = submission_root / "journal" / "0091_UNEXPECTED.json"
    extra.chmod(0o600)
    extra.unlink()
    snapshot_seal = submission_root / "journal" / "0001_SNAPSHOT_SEALED.json"
    snapshot_seal.chmod(0o600)
    snapshot_seal.unlink()
    runtime.append_journal(
        submission_root,
        1,
        "SNAPSHOT_SEALED",
        {"inventory_sha256": "0" * 64, "file_count": 1},
    )
    with pytest.raises(runtime.RuntimeContractError, match="snapshot-seal journal"):
        runtime.load_receipt(submission_root)


def test_live_cancel_launcher_execs_exact_snapshot_before_loading_drifted_live_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _manifest, submission_root, snapshot_root, _sha, _receipt, _fake, _boundary = _sealed_submission_fixture(tmp_path)
    raw = ["--cancel", "--submission-root", str(submission_root)]
    command = cancel_cli.snapshot_dispatch_command(submission_root, raw)
    assert command is not None
    assert command[:5] == [
        "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python",
        "-I",
        "-S",
        "-B",
        str(snapshot_root / runtime.PACKAGE_RELATIVE / "cancel.py"),
    ]
    assert command[5:] == ["--snapshot-resident", *raw]

    class ExecCaptured(Exception):
        pass

    captured: dict[str, object] = {}

    def capture_exec(path, argv, environment):
        captured.update(path=path, argv=list(argv), environment=dict(environment))
        raise ExecCaptured

    original_import = builtins.__import__

    def reject_live_package_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"campaign", "runtime"}:
            raise AssertionError(f"live package import used before dispatch: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_live_package_import)
    monkeypatch.setattr(cancel_cli, "_require_mutation_bootstrap", lambda: None)
    monkeypatch.setattr(cancel_cli.os, "execve", capture_exec)
    with pytest.raises(ExecCaptured):
        cancel_cli.main(raw)
    assert captured["argv"] == command
    assert captured["environment"] == {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


@pytest.mark.parametrize("mode", ["--cancel", "--recover"])
def test_absent_contract_publication_race_rechecks_under_lock_and_execs_snapshot(
    monkeypatch,
    tmp_path: Path,
    mode: str,
) -> None:
    submission_root = tmp_path / "submission"
    raw = [mode, "--submission-root", str(submission_root)]
    snapshot_command = [
        runtime.PINNED_PYTHON,
        "-I",
        "-S",
        "-B",
        str(tmp_path / "published-snapshot" / "cancel.py"),
        "--snapshot-resident",
        *raw,
    ]
    lock_state = {"held": False}
    decisions = 0

    def racing_decision(root: Path, argv):
        nonlocal decisions
        assert root == submission_root
        assert list(argv) == raw
        decisions += 1
        if decisions == 1:
            assert lock_state["held"] is False
            return "contract_absent", None
        assert decisions == 2
        assert lock_state["held"] is True
        return "exec_authenticated_snapshot", snapshot_command

    @contextmanager
    def fake_transaction_lock(root: Path):
        assert root == submission_root
        lock_state["held"] = True
        try:
            yield object()
        finally:
            lock_state["held"] = False

    class ExecCaptured(Exception):
        pass

    captured: dict[str, object] = {}

    def capture_exec(path, argv, environment):
        assert lock_state["held"] is False
        captured.update(path=path, argv=list(argv), environment=dict(environment))
        raise ExecCaptured

    def forbidden_live_action(*_args, **_kwargs):
        raise AssertionError("live recovery/cancellation used after contract publication")

    monkeypatch.setattr(cancel_cli, "_require_mutation_bootstrap", lambda: None)
    monkeypatch.setattr(cancel_cli, "_snapshot_dispatch_decision", racing_decision)
    monkeypatch.setattr(runtime, "transaction_recovery_lock", fake_transaction_lock)
    monkeypatch.setattr(runtime, "_recover_transaction_locked", forbidden_live_action)
    monkeypatch.setattr(runtime, "cancellation_plan", forbidden_live_action)
    monkeypatch.setattr(cancel_cli.os, "execve", capture_exec)
    with pytest.raises(ExecCaptured):
        cancel_cli.main(raw)
    assert decisions == 2
    assert captured["argv"] == snapshot_command


def test_absent_contract_cancel_fails_while_external_lock_is_held(
    monkeypatch,
    tmp_path: Path,
) -> None:
    submission_root = tmp_path / "submission"
    raw = ["--cancel", "--submission-root", str(submission_root)]
    lock_state = {"held": False}
    decisions = 0

    def absent_decision(_root: Path, _argv):
        nonlocal decisions
        decisions += 1
        if decisions == 2:
            assert lock_state["held"] is True
        return "contract_absent", None

    @contextmanager
    def fake_transaction_lock(_root: Path):
        lock_state["held"] = True
        try:
            yield object()
        finally:
            lock_state["held"] = False

    monkeypatch.setattr(cancel_cli, "_require_mutation_bootstrap", lambda: None)
    monkeypatch.setattr(cancel_cli, "_snapshot_dispatch_decision", absent_decision)
    monkeypatch.setattr(runtime, "transaction_recovery_lock", fake_transaction_lock)
    with pytest.raises(cancel_cli.DispatchError, match="contract and receipt"):
        cancel_cli.main(raw)
    assert decisions == 2
    assert lock_state["held"] is False


def test_precontract_recovery_remains_under_same_external_lock(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    submission_root = tmp_path / "submission"
    raw = ["--recover", "--submission-root", str(submission_root)]
    lock_state = {"held": False}
    lock_handle = object()

    @contextmanager
    def fake_transaction_lock(_root: Path):
        lock_state["held"] = True
        try:
            yield lock_handle
        finally:
            lock_state["held"] = False

    def fake_locked_recovery(_manifest, *, submission_root: Path, boundary, lock_handle: object):
        assert submission_root == tmp_path / "submission"
        assert lock_state["held"] is True
        assert lock_handle is globals_lock_handle
        assert boundary.calls == []
        return {"schema_version": 1, "status": "test_precontract_recovered"}

    globals_lock_handle = lock_handle
    monkeypatch.setattr(cancel_cli, "_require_mutation_bootstrap", lambda: None)
    monkeypatch.setattr(
        cancel_cli,
        "_snapshot_dispatch_decision",
        lambda _root, _argv: ("contract_absent", None),
    )
    monkeypatch.setattr(runtime, "transaction_recovery_lock", fake_transaction_lock)
    monkeypatch.setattr(runtime, "_recover_transaction_locked", fake_locked_recovery)
    assert cancel_cli.main(raw) == 0
    assert lock_state["held"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "test_precontract_recovered"


def test_cancel_mutation_rejects_unisolated_bootstrap_before_dispatch(tmp_path: Path) -> None:
    with pytest.raises(cancel_cli.DispatchError, match="-I -S -B"):
        cancel_cli.main(["--cancel", "--submission-root", str(tmp_path / "absent")])


@pytest.mark.parametrize("mode", ["--cancel", "--recover"])
def test_cancel_mutation_requires_actual_bytecode_flag_not_mutable_runtime_variable(
    tmp_path: Path,
    mode: str,
) -> None:
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONDONTWRITEBYTECODE"
    }
    completed = subprocess.run(
        [
            runtime.PINNED_PYTHON,
            "-I",
            "-S",
            str(runtime.PACKAGE_DIR / "cancel.py"),
            mode,
            "--submission-root",
            str(tmp_path / "absent"),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 2
    assert "exact pinned Python 3.11 with -I -S -B" in completed.stderr


def test_queued_worker_requires_actual_bytecode_flag_before_snapshot_access(tmp_path: Path) -> None:
    _manifest, submission_root, snapshot_root, submission_sha, receipt, _fake, _boundary = _sealed_submission_fixture(tmp_path)
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONDONTWRITEBYTECODE"
    }
    completed = subprocess.run(
        [
            runtime.PINNED_PYTHON,
            "-I",
            "-S",
            str(snapshot_root / runtime.PACKAGE_RELATIVE / "worker.py"),
            "run",
            "--submission-root",
            str(submission_root),
            "--snapshot-root",
            str(snapshot_root),
            "--submission-sha256",
            submission_sha,
            "--node",
            "train_2000",
            "--cell-index",
            "0",
            "--restart-count",
            "0",
            "--array-job-id",
            str(receipt["jobs"][0]["job_id"]),
            "--array-task-id",
            "0",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 2
    assert "queued entry requires Python -B" in completed.stderr


def test_emergency_dispatch_and_cancel_plan_survive_unrelated_snapshot_loss(tmp_path: Path) -> None:
    _manifest, submission_root, snapshot_root, _sha, _receipt, _fake, _boundary = _sealed_submission_fixture(tmp_path)
    unrelated = snapshot_root / runtime.PACKAGE_RELATIVE / "train_entry.py"
    package_dir = unrelated.parent
    package_dir.chmod(0o755)
    unrelated.unlink()
    package_dir.chmod(0o555)
    raw = ["--cancel", "--submission-root", str(submission_root)]
    assert cancel_cli.snapshot_dispatch_command(submission_root, raw) is not None
    with pytest.raises(runtime.RuntimeContractError, match="snapshot tree differs"):
        runtime.load_receipt(submission_root)
    plan = runtime.cancellation_plan(submission_root)
    assert len(plan["job_ids_reverse_dag"]) == 11


@pytest.mark.parametrize(
    "updates",
    [
        {},
        {"array_task_id": 40, "cell_index": 40},
        {"promotion_authority": "gate_2000"},
        {"wandb_id": "forged-run", "run_identity_sha256": "9" * 64},
    ],
)
def test_every_requeue_ready_is_non_authoritative_and_never_reaches_scheduler(
    tmp_path: Path,
    updates: dict[str, object],
) -> None:
    _manifest, submission_root, snapshot_root, _sha, receipt, _fake, _boundary = _sealed_submission_fixture(tmp_path)
    _authenticated, receipt_sha = runtime.load_receipt(submission_root)
    job = receipt["jobs"][0]
    generation_root = submission_root / "runtime" / "train_2000" / "cell-0" / "generation-1"
    generation_root.mkdir(parents=True, mode=0o700)
    ready: dict[str, object] = {
        "schema_version": 1,
        "status": "ready_for_same_stage_same_cell_requeue",
        "campaign_id": runtime.CAMPAIGN_ID,
        "submission_sha256": receipt["submission_sha256"],
        "submission_receipt_sha256": receipt_sha,
        "role": "train_2000",
        "stage_target": 2000,
        "cell_index": 0,
        "restart_count": 1,
        "array_job_id": job["job_id"],
        "array_task_id": 0,
        "completed_updates": 100,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_identity_sha256": "b" * 64,
        "run_identity_sha256": "c" * 64,
        "wandb_id": "fixed-run-identity",
        "promotion_authority": "none_within_stage_requeue_only",
    }
    ready.update(updates)
    runtime.exclusive_json(generation_root / "REQUEUE_READY.json", ready)
    calls: list[list[str]] = []

    def forbidden_runner(command, **_kwargs):
        calls.append(list(command))
        return _completed("")

    boundary = runtime.SchedulerBoundary(
        runner=forbidden_runner,
        observer=lambda: {},
        expected={},
    )
    with pytest.raises(runtime.RuntimeContractError, match="disabled in M2A"):
        runtime.call_same_run_requeue(
            submission_root,
            generation_root,
            ready,
            boundary=boundary,
            execution=campaign.load_manifest()["execution"],
            cwd=snapshot_root,
        )
    assert calls == []
    assert not (generation_root / "REQUEUE_CALLING.json").exists()


def test_train_wrapper_resets_child_signal_dispositions_before_exec() -> None:
    wrapper = (runtime.PACKAGE_DIR / "train.slurm").read_text()
    assert "trap - USR1 TERM INT" in wrapper
    assert "trap '' USR1 TERM INT" not in wrapper


def test_explicit_cancel_is_exact_quiet_and_terminal_retry_is_nonmutating(tmp_path: Path) -> None:
    manifest, submission_root, _snapshot_root, _sha, receipt, fake, boundary = _sealed_submission_fixture(tmp_path)
    first = runtime.explicit_cancel(
        submission_root,
        boundary=boundary,
        execution=manifest["execution"],
    )
    assert first["status"] == "cancellation_converged_terminal_or_absent"
    scancel_calls = [call for call in fake.calls if Path(call[0]).name == "scancel"]
    assert scancel_calls == [
        [manifest["execution"]["scancel"], "--quiet", *[row["job_id"] for row in reversed(receipt["jobs"])]],
    ]
    second = runtime.explicit_cancel(
        submission_root,
        boundary=boundary,
        execution=manifest["execution"],
    )
    assert second["retry"] is True
    assert [call for call in fake.calls if Path(call[0]).name == "scancel"] == scancel_calls


def test_explicit_cancel_reuses_latch_after_failed_attempt(monkeypatch, tmp_path: Path) -> None:
    manifest, submission_root, _snapshot_root, _sha, _receipt, _fake, boundary = _sealed_submission_fixture(tmp_path)
    original = runtime.cancel_exact

    def fail_after_latch(*_args, **_kwargs):
        raise runtime.RuntimeContractError("injected post-latch scheduler failure")

    monkeypatch.setattr(runtime, "cancel_exact", fail_after_latch)
    with pytest.raises(runtime.RuntimeContractError, match="retry is safe"):
        runtime.explicit_cancel(
            submission_root,
            boundary=boundary,
            execution=manifest["execution"],
        )
    latch, latch_sha = runtime.authenticated_immutable_json(
        submission_root / "CANCEL_REQUESTED.json",
        "test cancellation latch",
    )
    monkeypatch.setattr(runtime, "cancel_exact", original)
    result = runtime.explicit_cancel(
        submission_root,
        boundary=boundary,
        execution=manifest["execution"],
    )
    assert result["status"] == "cancellation_converged_terminal_or_absent"
    assert result["cancel_latch_sha256"] == latch_sha
    assert latch["status"] == "cancellation_latched_before_scheduler_call"


def test_committed_receipt_barrier_success_and_cancel_rejection(tmp_path: Path) -> None:
    _manifest, submission_root, _snapshot_root, _sha, _receipt, _fake, _boundary = _sealed_submission_fixture(tmp_path)
    # Even a later recovery holding the outer lock cannot block an already
    # authorized/released worker; the immutable authorization is sufficient.
    with runtime.transaction_recovery_lock(submission_root):
        with runtime.queued_transaction_barrier(
            submission_root,
            max_attempts=1,
            sleeper=lambda _seconds: None,
            require_committed_receipt=True,
        ) as evidence:
            assert evidence["status"] == "durable_release_authorization_bypassed_outer_transaction_lock"
            assert evidence["attempt"] == 0
    runtime.exclusive_json(
        submission_root / "CANCEL_REQUESTED.json",
        {"schema_version": 1, "status": "test_cancel_latched"},
    )
    with pytest.raises(runtime.RuntimeContractError, match="terminal state"):
        with runtime.queued_transaction_barrier(
            submission_root,
            max_attempts=1,
            sleeper=lambda _seconds: None,
            require_committed_receipt=True,
        ):
            raise AssertionError("canceled transaction passed queued barrier")


def test_receipt_barrier_times_out_on_recoverable_link_residue_and_rejects_malformed_final(tmp_path: Path) -> None:
    submission_root = tmp_path / "submission"
    with runtime.transaction_recovery_lock(submission_root):
        runtime.begin_transaction(submission_root, "5" * 64)
    temporary = submission_root / ".SUBMISSION_RECEIPT.json.PUBLISHING"
    final = submission_root / "SUBMISSION_RECEIPT.json"
    temporary.write_text("{}\n")
    temporary.chmod(0o444)
    os.link(temporary, final)
    with pytest.raises(runtime.RuntimeContractError, match="timed out"):
        with runtime.queued_transaction_barrier(
            submission_root,
            max_attempts=1,
            sleeper=lambda _seconds: None,
            require_committed_receipt=True,
        ):
            raise AssertionError("linked receipt residue passed queued barrier")
    temporary.unlink()
    final.unlink()
    final.write_text('{"schema_version":')
    final.chmod(0o600)
    with pytest.raises(runtime.RuntimeContractError, match="identity/mode"):
        runtime._queued_commit_state(submission_root)


def test_isolated_snapshot_worker_crosses_committed_barrier_then_hits_authority_block(tmp_path: Path) -> None:
    _manifest, submission_root, snapshot_root, submission_sha, receipt, _fake, _boundary = _sealed_submission_fixture(tmp_path)
    worker = snapshot_root / runtime.PACKAGE_RELATIVE / "worker.py"
    command = [
        "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python",
        "-I",
        "-S",
        "-B",
        str(worker),
        "run",
        "--snapshot-root",
        str(snapshot_root),
        "--submission-root",
        str(submission_root),
        "--submission-sha256",
        submission_sha,
        "--node",
        "train_2000",
        "--cell-index",
        "0",
        "--restart-count",
        "0",
        "--array-job-id",
        str(receipt["jobs"][0]["job_id"]),
        "--array-task-id",
        "0",
    ]
    completed = subprocess.run(
        command,
        cwd=snapshot_root,
        env={
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 2
    assert "EXP24_WORKER_BLOCKED" in completed.stderr
    assert "Launch8 semantic adapter" in completed.stderr
    assert "receipt" not in completed.stderr.lower()


def test_m2a_authority_graph_is_exact_acyclic_and_execution_blocked() -> None:
    schema, file_sha = campaign.load_m2a_schema()
    assert len(schema["topological_order"]) == 14
    assert file_sha == hashlib.sha256(campaign.M2A_SCHEMA_PATH.read_bytes()).hexdigest()
    assert schema["invariants"]["formal_submission_allowed"] is False
    assert schema["invariants"]["scientific_protocol_sealed"] is False
    assert schema["invariants"]["execution_readiness_ready"] is False
    assert schema["invariants"]["same_stage_requeue_mutation_is_disabled"] is True
    artifacts = {row["id"]: row for row in schema["artifacts"]}
    assert artifacts["interpreter_provenance"]["depends_on"] == ["source_manifest_file"]
    assert artifacts["launch7_terminal_negative_binding"]["depends_on"] == [
        "launch7_terminal_negative_evidence"
    ]
    assert artifacts["accepted_engineering_pilot_binding"]["depends_on"] == [
        "engineering_pilot_adapter_interface",
        "future_engineering_pilot_report_commit",
    ]
    assert set(artifacts["source_snapshot_inventory"]["depends_on"]) == {
        "m2a_schema_file",
        "source_manifest_file",
        "launch7_terminal_negative_evidence",
        "launch7_terminal_negative_binding",
        "engineering_pilot_adapter_interface",
        "accepted_engineering_pilot_binding",
    }
    malformed = json.loads(json.dumps(schema))
    malformed["artifacts"][0]["depends_on"] = ["submission_contract"]
    with pytest.raises(campaign.ContractError, match="artifact contract|cyclic"):
        campaign.validate_m2a_schema(malformed)
    forged_hash_field = json.loads(json.dumps(schema))
    forged_hash_field["artifacts"][2]["hash_fields"] = ["arbitrary_sha256"]
    with pytest.raises(campaign.ContractError, match="artifact contract"):
        campaign.validate_m2a_schema(forged_hash_field)
    forged_authority = json.loads(json.dumps(schema))
    forged_authority["artifacts"][12]["authority"] = "release_was_probably_observed"
    with pytest.raises(campaign.ContractError, match="artifact contract"):
        campaign.validate_m2a_schema(forged_authority)


def test_interpreter_provenance_is_self_hashed_and_exactly_recaptured() -> None:
    manifest = campaign.load_manifest()
    provenance = runtime.capture_interpreter_provenance(manifest)
    assert provenance["python_version"] == "3.11.15"
    assert provenance["source_manifest_sha256"] == campaign.manifest_sha256(manifest)
    assert provenance["lexical_kind"] == "symlink"
    assert provenance["resolved_executable_sha256"] == hashlib.sha256(
        Path(provenance["resolved_executable"]).read_bytes()
    ).hexdigest()
    assert runtime.validate_interpreter_provenance(manifest, provenance) == provenance
    forged = dict(provenance)
    forged["resolved_executable_sha256"] = "0" * 64
    forged["provenance_sha256"] = runtime.stable_hash(
        {key: value for key, value in forged.items() if key != "provenance_sha256"}
    )
    with pytest.raises(runtime.RuntimeContractError, match="drifted"):
        runtime.validate_interpreter_provenance(manifest, forged)
    wrong_manifest = dict(provenance)
    wrong_manifest["source_manifest_sha256"] = "1" * 64
    wrong_manifest["provenance_sha256"] = runtime.stable_hash(
        {key: value for key, value in wrong_manifest.items() if key != "provenance_sha256"}
    )
    with pytest.raises(runtime.RuntimeContractError, match="drifted"):
        runtime.validate_interpreter_provenance(manifest, wrong_manifest)


def test_train_2000_is_exactly_held_in_command_and_test_contract(tmp_path: Path) -> None:
    manifest = campaign.load_manifest()
    record = runtime.scheduler_commands(
        manifest,
        tmp_path / "snapshot",
        tmp_path / "submission",
        "a" * 64,
        through_index=0,
    )[0]
    assert record["node"]["name"] == "train_2000"
    assert record["command"].count("--hold") == 1
    options, _partitions = runtime.expected_test_options(manifest, record, dependency=None)
    assert options["hold"] == "set"


def test_held_root_authorization_is_durable_without_release_side_effect(tmp_path: Path) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _prepared_submission_fixture(tmp_path)
    receipt = runtime.submit_dag_transaction(
        manifest,
        submission_root=submission_root,
        snapshot_root=snapshot_root,
        submission_sha256=submission_sha,
        boundary=boundary,
    )
    root_id = str(receipt["jobs"][0]["job_id"])
    authorization = runtime.authorize_root_release_after_receipt(
        submission_root,
        boundary=boundary,
    )
    assert authorization["status"] == "receipt_committed_root_release_authorized"
    assert fake.jobs[root_id]["held"] is True
    assert [call for call in fake.calls if call[1:2] == ["release"]] == []
    assert (submission_root / "ROOT_RELEASE_AUTHORIZATION.json").is_file()


def test_submit_wrapper_releases_outer_lock_before_root_release(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = json.loads(json.dumps(campaign.load_manifest()))
    manifest["paths"]["run_root"] = str(tmp_path / "run")
    lock_state = {"held": False}
    boundary = object()
    receipt = {"status": "submitted"}
    authorization = {"status": "receipt_committed_root_release_authorized"}

    @contextmanager
    def fake_lock(_submission_root: Path):
        assert lock_state["held"] is False
        lock_state["held"] = True
        try:
            yield object()
        finally:
            lock_state["held"] = False

    def fake_locked_submit(*_args, **_kwargs):
        assert lock_state["held"] is True
        return {
            "schema_version": 1,
            "status": "submission_receipt_and_held_root_release_authorization_committed",
            "receipt": receipt,
            "authorization": authorization,
        }

    def fake_activate(_submission_root: Path, *, boundary: object):
        assert lock_state["held"] is False
        return {"status": "train_2000_released_and_observed"}

    monkeypatch.setattr(campaign, "assert_launch_authorized", lambda _manifest: None)
    monkeypatch.setattr(runtime, "execution_readiness", lambda _manifest: {"ready": True, "blockers": []})
    monkeypatch.setattr(runtime, "capture_interpreter_provenance", lambda _manifest: {"test": "provenance"})
    monkeypatch.setattr(runtime, "capture_scheduler_control_plane_bundle", lambda _execution: (b"config", {"control": "plane"}))
    monkeypatch.setattr(runtime, "scheduler_fallback_binding", lambda _payload, _observation: {"fallback": True})
    monkeypatch.setattr(runtime, "scheduler_preclaim_test", lambda *_args, **_kwargs: {"preclaim": True})
    monkeypatch.setattr(runtime, "transaction_recovery_lock", fake_lock)
    monkeypatch.setattr(runtime, "_authorized_submit_locked", fake_locked_submit)
    monkeypatch.setattr(runtime, "activate_root_after_receipt", fake_activate)
    result = runtime.authorized_submit(manifest, boundary_factory=lambda _observation: boundary)
    assert result["status"] == "submission_receipt_committed_and_root_activated"
    assert result["receipt"] is receipt
    assert result["authorization"] is authorization
    assert lock_state["held"] is False


def test_recovery_wrapper_releases_outer_lock_before_root_release(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = campaign.load_manifest()
    submission_root = tmp_path / "submission"
    lock_state = {"held": False}
    boundary = type("FakeRecoveryBoundary", (), {"calls": []})()

    @contextmanager
    def fake_lock(root: Path):
        assert root == submission_root and lock_state["held"] is False
        lock_state["held"] = True
        try:
            yield object()
        finally:
            lock_state["held"] = False

    def fake_locked_recovery(*_args, **_kwargs):
        assert lock_state["held"] is True
        return {
            "schema_version": 1,
            "status": "receipt_already_committed_root_release_authorized",
            "authorization": {"status": "receipt_committed_root_release_authorized"},
            "_post_transaction_activation_status": (
                "receipt_already_committed_root_activation_recovered"
            ),
        }

    def fake_activate(root: Path, *, boundary: object):
        assert root == submission_root and lock_state["held"] is False
        return {"status": "train_2000_released_and_observed"}

    monkeypatch.setattr(runtime, "transaction_recovery_lock", fake_lock)
    monkeypatch.setattr(runtime, "_recover_transaction_locked", fake_locked_recovery)
    monkeypatch.setattr(runtime, "activate_root_after_receipt", fake_activate)
    result = runtime.recover_transaction(
        manifest,
        submission_root=submission_root,
        boundary=boundary,
    )
    assert result["status"] == "receipt_already_committed_root_activation_recovered"
    assert "_post_transaction_activation_status" not in result
    assert result["activation"]["status"] == "train_2000_released_and_observed"
    assert lock_state["held"] is False


def test_committed_receipt_recovery_authorizes_then_activates_idempotently(
    tmp_path: Path,
) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _prepared_submission_fixture(tmp_path)
    receipt = runtime.submit_dag_transaction(
        manifest,
        submission_root=submission_root,
        snapshot_root=snapshot_root,
        submission_sha256=submission_sha,
        boundary=boundary,
    )
    root_id = str(receipt["jobs"][0]["job_id"])
    first = runtime.recover_transaction(
        manifest,
        submission_root=submission_root,
        boundary=boundary,
    )
    assert first["status"] == "receipt_already_committed_root_activation_recovered"
    assert first["authorization"]["status"] == "receipt_committed_root_release_authorized"
    assert first["activation"]["status"] == "train_2000_released_and_observed"
    assert fake.jobs[root_id]["held"] is False
    assert len([call for call in fake.calls if call[1:2] == ["release"]]) == 1
    second = runtime.recover_transaction(
        manifest,
        submission_root=submission_root,
        boundary=boundary,
    )
    assert second["activation"]["retry"] is True
    assert len([call for call in fake.calls if call[1:2] == ["release"]]) == 1


def test_root_activation_orders_receipt_authorization_release_observation_result(tmp_path: Path) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _prepared_submission_fixture(tmp_path)
    receipt = runtime.submit_dag_transaction(
        manifest,
        submission_root=submission_root,
        snapshot_root=snapshot_root,
        submission_sha256=submission_sha,
        boundary=boundary,
    )
    root_id = str(receipt["jobs"][0]["job_id"])
    ordinals: list[str] = []

    def inspect(ordinal: str) -> None:
        ordinals.append(ordinal)
        assert (submission_root / "SUBMISSION_RECEIPT.json").is_file()
        if ordinal in {
            "authorization_published", "before_release_call", "after_release_call",
            "released_observed", "activation_result_published",
        }:
            assert (submission_root / "ROOT_RELEASE_AUTHORIZATION.json").is_file()
        release_calls = [call for call in fake.calls if call[1:2] == ["release"]]
        if ordinal in {"authorization_published", "before_release_call"}:
            assert release_calls == []

    result = runtime.activate_root_after_receipt(
        submission_root,
        boundary=boundary,
        fault_hook=inspect,
    )
    assert ordinals == [
        "authorization_published",
        "before_release_call",
        "after_release_call",
        "released_observed",
        "activation_result_published",
    ]
    assert result["status"] == "train_2000_released_and_observed"
    assert result["release_command"] == [manifest["execution"]["scontrol"], "release", root_id]
    assert fake.jobs[root_id]["held"] is False
    assert len([call for call in fake.calls if call[1:2] == ["release"]]) == 1
    assert runtime.activate_root_after_receipt(submission_root, boundary=boundary)["retry"] is True
    assert len([call for call in fake.calls if call[1:2] == ["release"]]) == 1


@pytest.mark.parametrize(
    "crash_ordinal",
    [
        "authorization_published",
        "before_release_call",
        "after_release_call",
        "released_observed",
        "activation_result_published",
    ],
)
def test_root_activation_recovers_every_durable_release_ordinal(
    tmp_path: Path,
    crash_ordinal: str,
) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _prepared_submission_fixture(tmp_path)
    receipt = runtime.submit_dag_transaction(
        manifest,
        submission_root=submission_root,
        snapshot_root=snapshot_root,
        submission_sha256=submission_sha,
        boundary=boundary,
    )
    root_id = str(receipt["jobs"][0]["job_id"])

    class InjectedCrash(RuntimeError):
        pass

    def crash(ordinal: str) -> None:
        if ordinal == crash_ordinal:
            raise InjectedCrash(ordinal)

    with pytest.raises(InjectedCrash):
        runtime.activate_root_after_receipt(
            submission_root,
            boundary=boundary,
            fault_hook=crash,
        )
    recovered = runtime.recover_transaction(
        manifest,
        submission_root=submission_root,
        boundary=boundary,
    )
    assert recovered["status"] == "receipt_already_committed_root_activation_recovered"
    assert recovered["activation"]["status"] == "train_2000_released_and_observed"
    assert fake.jobs[root_id]["held"] is False
    assert (submission_root / "ROOT_RELEASE_AUTHORIZATION.json").is_file()
    assert (submission_root / "ROOT_ACTIVATION_RESULT.json").is_file()
    assert len([call for call in fake.calls if call[1:2] == ["release"]]) == 1


def test_cancellation_latch_blocks_root_activation_without_release(tmp_path: Path) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _prepared_submission_fixture(tmp_path)
    receipt = runtime.submit_dag_transaction(
        manifest,
        submission_root=submission_root,
        snapshot_root=snapshot_root,
        submission_sha256=submission_sha,
        boundary=boundary,
    )
    canceled = runtime.explicit_cancel(
        submission_root,
        boundary=boundary,
        execution=manifest["execution"],
    )
    assert canceled["status"] == "cancellation_converged_terminal_or_absent"
    with pytest.raises(runtime.RuntimeContractError, match="conflicts with CANCEL"):
        runtime.activate_root_after_receipt(submission_root, boundary=boundary)
    assert [call for call in fake.calls if call[1:2] == ["release"]] == []
    assert fake.jobs[str(receipt["jobs"][0]["job_id"])]["canceled"] is True


def test_forged_release_authorization_fails_before_scheduler_release(tmp_path: Path) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _prepared_submission_fixture(tmp_path)
    runtime.submit_dag_transaction(
        manifest,
        submission_root=submission_root,
        snapshot_root=snapshot_root,
        submission_sha256=submission_sha,
        boundary=boundary,
    )
    context = runtime._root_activation_context(submission_root)
    held, lifecycle = runtime._observe_root_lifecycle(boundary, context)
    assert lifecycle == "held"
    forged = runtime._authorization_seed(context, held)
    forged["root_job_id"] = "999999"
    forged["authorization_body_sha256"] = runtime.stable_hash(
        {key: value for key, value in forged.items() if key != "authorization_body_sha256"}
    )
    runtime.exclusive_json(submission_root / "ROOT_RELEASE_AUTHORIZATION.json", forged)
    calls_before = list(fake.calls)
    with pytest.raises(runtime.RuntimeContractError, match="authorization differs"):
        runtime.activate_root_after_receipt(submission_root, boundary=boundary)
    assert [call for call in fake.calls[len(calls_before):] if call[1:2] == ["release"]] == []


def test_activation_repairs_link_before_unlink_authorization_publication(tmp_path: Path) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _prepared_submission_fixture(tmp_path)
    runtime.submit_dag_transaction(
        manifest,
        submission_root=submission_root,
        snapshot_root=snapshot_root,
        submission_sha256=submission_sha,
        boundary=boundary,
    )
    context = runtime._root_activation_context(submission_root)
    held, lifecycle = runtime._observe_root_lifecycle(boundary, context)
    assert lifecycle == "held"
    authorization = runtime._authorization_seed(context, held)
    payload = (runtime.canonical_json(authorization) + "\n").encode()
    temporary = submission_root / ".ROOT_RELEASE_AUTHORIZATION.json.PUBLISHING"
    final = submission_root / "ROOT_RELEASE_AUTHORIZATION.json"
    temporary.write_bytes(payload)
    temporary.chmod(0o444)
    os.link(temporary, final)
    assert final.stat().st_nlink == 2
    result = runtime.activate_root_after_receipt(submission_root, boundary=boundary)
    assert result["status"] == "train_2000_released_and_observed"
    assert final.stat().st_nlink == 1
    assert not temporary.exists()


def test_activation_recovers_release_effect_with_lost_client_response(tmp_path: Path) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, _boundary = _prepared_submission_fixture(tmp_path)
    stable = {"schema_version": 1, "kind": "fake-stable-control-plane"}
    lost = False

    def ambiguous_runner(raw_command, cwd, environment, inherited_fds):
        nonlocal lost
        command = list(raw_command)
        completed = fake(command, cwd, environment, inherited_fds)
        if command[1:2] == ["release"] and not lost:
            lost = True
            raise subprocess.TimeoutExpired(command, 30)
        return completed

    boundary = runtime.SchedulerBoundary(
        runner=ambiguous_runner,
        observer=lambda: stable,
        expected=stable,
    )
    runtime.submit_dag_transaction(
        manifest,
        submission_root=submission_root,
        snapshot_root=snapshot_root,
        submission_sha256=submission_sha,
        boundary=boundary,
    )
    with pytest.raises(runtime.ActivationRecoveryRequired, match="ambiguous"):
        runtime.activate_root_after_receipt(submission_root, boundary=boundary)
    assert (submission_root / "ROOT_RELEASE_AUTHORIZATION.json").is_file()
    assert not (submission_root / "ROOT_ACTIVATION_RESULT.json").exists()
    recovered = runtime.recover_transaction(
        manifest,
        submission_root=submission_root,
        boundary=boundary,
    )
    assert recovered["activation"]["release_response_mode"] == (
        "reconciled_already_released_after_durable_authorization"
    )
    assert (submission_root / "ROOT_ACTIVATION_RESULT.json").is_file()
    assert len([call for call in fake.calls if call[1:2] == ["release"]]) == 1


def test_release_and_cancellation_are_linearly_ordered_by_one_lock(tmp_path: Path) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _prepared_submission_fixture(tmp_path)
    runtime.submit_dag_transaction(
        manifest,
        submission_root=submission_root,
        snapshot_root=snapshot_root,
        submission_sha256=submission_sha,
        boundary=boundary,
    )
    activation_at_release = threading.Event()
    permit_release = threading.Event()
    activation_done = threading.Event()
    cancellation_done = threading.Event()
    failures: list[BaseException] = []

    def activation_hook(ordinal: str) -> None:
        if ordinal == "before_release_call":
            activation_at_release.set()
            assert permit_release.wait(5)

    def activate() -> None:
        try:
            runtime.activate_root_after_receipt(
                submission_root,
                boundary=boundary,
                fault_hook=activation_hook,
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            activation_done.set()

    def cancel() -> None:
        try:
            runtime.explicit_cancel(
                submission_root,
                boundary=boundary,
                execution=manifest["execution"],
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            cancellation_done.set()

    activation_thread = threading.Thread(target=activate)
    activation_thread.start()
    assert activation_at_release.wait(5)
    cancel_thread = threading.Thread(target=cancel)
    cancel_thread.start()
    assert not cancellation_done.wait(0.1)
    assert not any(Path(call[0]).name == "scancel" for call in fake.calls)
    permit_release.set()
    activation_thread.join(5)
    cancel_thread.join(5)
    assert activation_done.is_set() and cancellation_done.is_set()
    assert len(failures) <= 1
    if failures:
        assert isinstance(failures[0], runtime.RuntimeContractError)
        assert "activation return conflicts" in str(failures[0])
    release_index = next(index for index, call in enumerate(fake.calls) if call[1:2] == ["release"])
    cancel_index = next(index for index, call in enumerate(fake.calls) if Path(call[0]).name == "scancel")
    assert release_index < cancel_index
    assert (submission_root / "ROOT_ACTIVATION_RESULT.json").is_file()
    assert (submission_root / "CANCEL_RESULT.json").is_file()


def test_post_release_result_delay_does_not_block_emergency_cancellation(
    tmp_path: Path,
) -> None:
    manifest, submission_root, snapshot_root, submission_sha, fake, boundary = _prepared_submission_fixture(tmp_path)
    runtime.submit_dag_transaction(
        manifest,
        submission_root=submission_root,
        snapshot_root=snapshot_root,
        submission_sha256=submission_sha,
        boundary=boundary,
    )
    released_observed = threading.Event()
    permit_result = threading.Event()
    activation_done = threading.Event()
    cancellation_done = threading.Event()
    failures: list[BaseException] = []

    def activation_hook(ordinal: str) -> None:
        if ordinal == "released_observed":
            released_observed.set()
            assert permit_result.wait(5)

    def activate() -> None:
        try:
            runtime.activate_root_after_receipt(
                submission_root,
                boundary=boundary,
                fault_hook=activation_hook,
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            activation_done.set()

    def cancel() -> None:
        try:
            runtime.explicit_cancel(
                submission_root,
                boundary=boundary,
                execution=manifest["execution"],
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            cancellation_done.set()

    activation_thread = threading.Thread(target=activate)
    activation_thread.start()
    assert released_observed.wait(5)
    cancellation_thread = threading.Thread(target=cancel)
    cancellation_thread.start()
    assert cancellation_done.wait(5)
    assert not activation_done.is_set()
    assert (submission_root / "CANCEL_RESULT.json").is_file()
    permit_result.set()
    activation_thread.join(5)
    cancellation_thread.join(5)
    assert len(failures) == 1
    assert isinstance(failures[0], runtime.RuntimeContractError)
    assert "activation return conflicts" in str(failures[0])
    assert (submission_root / "ROOT_ACTIVATION_RESULT.json").is_file()


def test_engineering_pilot_adapter_rejects_invalid_identity_before_path_access(tmp_path: Path) -> None:
    missing = tmp_path / "must-not-be-opened"
    with pytest.raises(
        engineering_pilot_binder.EngineeringPilotBindingError,
        match="report/submission root identity differs",
    ):
        engineering_pilot_binder.verify_engineering_pilot_report_quartet(
            missing,
            expected_report_root=missing,
            expected_submission_root=missing,
            expected_submission_sha256="not-even-parsed",
            expected_package_binding={"must": "not-be-read"},
        )
    assert not missing.exists()


def test_engineering_pilot_adapter_description_forbids_launch7_positive_authority() -> None:
    description = engineering_pilot_binder.adapter_description()
    assert description["expected_campaign_id"] == (
        "treewm-executable-prefix-repair-pilot-v1-launch8"
    )
    assert description["forbidden_positive_campaign_id"] == (
        "treewm-executable-prefix-repair-pilot-v1-launch7"
    )
    assert description["adapter_state"] == "sealed_versioned_adapter"
    assert description["binding_state"] == "unbound"
    assert description["implementation_dependency_files"] == [
        str(runtime.PACKAGE_RELATIVE / "engineering_pilot_binder.py"),
        str(runtime.PACKAGE_RELATIVE / "runtime.py"),
    ]
    assert description["frozen_source_commit"] == runtime.FROZEN_LAUNCH8_SOURCE_COMMIT
    assert description["frozen_protocol_sha256"] == runtime.FROZEN_LAUNCH8_PROTOCOL_SHA256
    assert description["frozen_source_inventory_sha256"] == (
        runtime.FROZEN_LAUNCH8_VERIFIER_SOURCE_INVENTORY_SHA256
    )
    assert description["frozen_source_file_count"] == 147
    assert description["frozen_package_binding"] == runtime.FROZEN_LAUNCH8_PACKAGE_BINDING
    assert description["semantic_adapter_implemented"] is True
    assert description["persistent_writes_performed"] is False
    assert description["real_report_opened"] is False
    assert len(description["requirements"]) == 11


def test_positive_placeholder_and_authenticated_launch7_negative_are_distinct() -> None:
    positive = json.loads(
        (runtime.PACKAGE_DIR / "accepted_engineering_pilot.binding.json").read_text()
    )
    negative = json.loads(
        (runtime.PACKAGE_DIR / "launch7_negative.binding.json").read_text()
    )
    assert positive == {
        "schema_version": 1,
        "status": "awaiting_launch8_accepted_engineering_pilot",
        "campaign_id": "treewm-executable-prefix-repair-pilot-v1-launch8",
        "formal_submission_allowed": False,
        "adapter_file_sha256": None,
        "adapter_runtime_file_sha256": None,
        "adapter_description_sha256": None,
        "report_commit_file_sha256": None,
        "binding_sha256": None,
    }
    assert campaign.validate_launch7_negative_binding(negative) == (
        "629610c2bb677f53ee3acb75a8bcd1e3089bee78a4c43600a944e4290f5148bd"
    )
    assert negative["status"] == "authenticated_terminal_negative_no_reuse"
    assert negative["accepted"] is negative["reusable"] is False
    assert not (runtime.PACKAGE_DIR / "launch7_acceptance.binding.json").exists()
    assert not (runtime.PACKAGE_DIR / "launch7_binder.py").exists()
