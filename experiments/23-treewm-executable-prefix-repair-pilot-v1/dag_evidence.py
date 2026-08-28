#!/usr/bin/env python3
"""Pure semantic validator for Exp23 Launch8 scheduler evidence.

This module never calls a scheduler client.  Login-side recovery, compute workers,
and the terminal reporter all use the same validator so a coherently rehashed set
of JSON records cannot authorize a different graph.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping


CAMPAIGN_ID = "treewm-executable-prefix-repair-pilot-v1-launch8"
ROLES = ("wave0", "wave1", "report")
SBATCH_RESPONSE = re.compile(r"^(?P<job_id>[0-9]+)(?:;(?P<cluster>[A-Za-z0-9_.-]+))?$")


class EvidenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


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


def sealed_json_sha256(value: object) -> str:
    payload = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _field(stdout: object, name: str, label: str) -> str:
    require(isinstance(stdout, str), f"{label} stdout differs")
    lines = [line for line in stdout.splitlines() if line.strip()]
    require(len(lines) == 1, f"{label} stdout is ambiguous")
    prefix = f"{name}="
    values = [
        token[len(prefix) :]
        for token in lines[0].split()
        if token.startswith(prefix)
    ]
    require(len(values) == 1 and bool(values[0]), f"{label} {name} differs")
    return values[0]


def _paths(
    manifest: Mapping[str, Any],
    snapshot_root: Path,
    submission_root: Path,
    submission_sha256: str,
    job_ids: Mapping[str, str],
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, str], str]:
    execution = manifest["execution"]
    submit_client = str(execution["sbatch"])
    token = submission_sha256[:16]
    names = {
        role: f"exp23-launch8-{token}-{role}" for role in ROLES
    }
    comment = f"treewm-exp23:{submission_sha256}"
    logs = submission_root / "logs"
    package = snapshot_root / "experiments/23-treewm-executable-prefix-repair-pilot-v1"
    train = package / "train.slurm"
    report_script = package / "report.slurm"
    commands = {
        "wave0": [
            submit_client,
            "--parsable",
            "--export=NONE",
            "--hold",
            "--array=0-19%20",
            f"--job-name={names['wave0']}",
            f"--comment={comment}",
            f"--output={logs / 'wave0_%A_%a.out'}",
            str(train),
            str(snapshot_root),
            str(submission_root),
            submission_sha256,
            "0",
            "none",
        ],
        "wave1": [
            submit_client,
            "--parsable",
            "--export=NONE",
            "--array=0-19%20",
            f"--dependency=afterok:{job_ids['wave0']}",
            "--kill-on-invalid-dep=yes",
            f"--job-name={names['wave1']}",
            f"--comment={comment}",
            f"--output={logs / 'wave1_%A_%a.out'}",
            str(train),
            str(snapshot_root),
            str(submission_root),
            submission_sha256,
            "1",
            job_ids["wave0"],
        ],
        "report": [
            submit_client,
            "--parsable",
            "--export=NONE",
            f"--dependency=afterok:{job_ids['wave1']}",
            "--kill-on-invalid-dep=yes",
            f"--job-name={names['report']}",
            f"--comment={comment}",
            f"--output={logs / 'report_%j.out'}",
            str(report_script),
            str(snapshot_root),
            str(submission_root),
            submission_sha256,
        ],
    }
    tests = {
        role: [command[0], "-vvv", "--test-only", *command[1:]]
        for role, command in commands.items()
        if role != "wave0"
    }
    return commands, tests, names, comment


def _validate_lock(value: object, label: str) -> None:
    require(
        isinstance(value, Mapping)
        and set(value) == {"path", "device", "inode", "uid", "mode"}
        and isinstance(value.get("path"), str)
        and Path(str(value["path"])).is_absolute()
        and all(
            type(value.get(key)) is int and value[key] >= 0
            for key in ("device", "inode", "uid", "mode")
        )
        and value.get("mode") == 0o600,
        f"{label} transaction-lock binding differs",
    )


def _validate_calling(
    value: object,
    *,
    role: str,
    command: list[str],
    job_name: str,
    comment: str,
    submission_root: Path,
    submission_sha256: str,
    claim_token: str,
    expected_lock: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{role} calling record is absent")
    require(
        set(value)
        == {
            "schema_version",
            "status",
            "campaign_id",
            "submission_sha256",
            "claim_token",
            "role",
            "job_name",
            "scheduler_comment",
            "command",
            "transaction_lock",
        }
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "scheduler_calling"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and value.get("claim_token") == claim_token
        and value.get("role") == role
        and value.get("job_name") == job_name
        and value.get("scheduler_comment") == comment
        and value.get("command") == command,
        f"{role} calling record differs",
    )
    _validate_lock(value.get("transaction_lock"), role)
    expected_path = submission_root.absolute().parents[2] / (
        ".exp23-"
        + hashlib.sha256(str(submission_root.absolute()).encode("utf-8")).hexdigest()[:16]
        + ".transaction.lock"
    )
    binding = value["transaction_lock"]
    require(binding.get("path") == str(expected_path), f"{role} calling lock path differs")
    try:
        lock_info = expected_path.lstat()
    except OSError as exc:
        raise EvidenceError(f"{role} calling lock is unavailable: {exc}") from exc
    require(
        stat.S_ISREG(lock_info.st_mode)
        and (lock_info.st_dev, lock_info.st_ino)
        == (binding["device"], binding["inode"])
        and lock_info.st_uid == binding["uid"] == os.getuid()
        and stat.S_IMODE(lock_info.st_mode) == binding["mode"] == 0o600,
        f"{role} calling lock inode differs",
    )
    if expected_lock is not None:
        require(
            exact_json_equal(value.get("transaction_lock"), expected_lock),
            f"{role} calling lock lineage differs",
        )
    return value


def _validate_control(value: object, expected: Mapping[str, Any], label: str) -> None:
    require(
        exact_json_equal(value, expected),
        f"{label} scheduler control-plane evidence differs",
    )


def _validate_base_record(
    value: object,
    *,
    role: str,
    job_id: str,
    command: list[str],
    calling: Mapping[str, Any],
    expected_control: Mapping[str, Any],
) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{role} accepted record is absent")
    extra = {"accepted_hold"} if role == "wave0" else {
        "exact_dependency_test_only",
        "accepted_dependency",
    }
    require(
        set(value)
        == {
            "command",
            "returncode",
            "stdout",
            "stderr",
            "reconciled_job_ids",
            "scheduler_control_plane",
            "calling_sha256",
            *extra,
        },
        f"{role} accepted record fields differ",
    )
    stdout = value.get("stdout")
    require(
        value.get("command") == command
        and type(value.get("returncode")) is int
        and value.get("returncode") == 0
        and isinstance(stdout, str)
        and isinstance(value.get("stderr"), str)
        and len(stdout) <= 1024 * 1024
        and len(value["stderr"]) <= 1024 * 1024
        and value.get("reconciled_job_ids") == [job_id]
        and value.get("calling_sha256") == sealed_json_sha256(calling),
        f"{role} accepted submission identity differs",
    )
    response = SBATCH_RESPONSE.fullmatch(stdout.strip())
    require(
        response is None or response.group("job_id") == job_id,
        f"{role} parseable accepted response differs",
    )
    _validate_control(value.get("scheduler_control_plane"), expected_control, role)
    return value


def _validate_hold(
    value: object,
    *,
    control_client: str,
    job_id: str,
    job_name: str,
    comment: str,
    expected_control: Mapping[str, Any],
) -> None:
    require(
        isinstance(value, Mapping)
        and set(value)
        == {
            "command",
            "returncode",
            "stdout",
            "stderr",
            "state",
            "reason",
            "scheduler_control_plane",
        },
        "wave0 held evidence fields differ",
    )
    require(
        value.get("command")
        == [control_client, "show", "job", job_id, "--oneliner"]
        and type(value.get("returncode")) is int
        and value.get("returncode") == 0
        and isinstance(value.get("stdout"), str)
        and isinstance(value.get("stderr"), str)
        and len(value["stdout"]) <= 1024 * 1024
        and len(value["stderr"]) <= 1024 * 1024
        and value.get("state") == "PENDING"
        and value.get("reason") == "JobHeldUser"
        and _field(value["stdout"], "JobId", "wave0 hold") == job_id
        and _field(value["stdout"], "JobName", "wave0 hold") == job_name
        and _field(value["stdout"], "Comment", "wave0 hold") == comment
        and _field(value["stdout"], "JobState", "wave0 hold") == "PENDING"
        and _field(value["stdout"], "Reason", "wave0 hold") == "JobHeldUser",
        "wave0 was not accepted as the exact held array",
    )
    _validate_control(value.get("scheduler_control_plane"), expected_control, "wave0 hold")


def _parsed_test_only(
    stderr: str,
    *,
    role: str,
    manifest: Mapping[str, Any],
    dependency: str,
) -> dict[str, Any]:
    lines = stderr.splitlines()
    try:
        start = lines.index("sbatch: defined options")
        end = lines.index("sbatch: end of defined options", start + 1)
    except ValueError as exc:
        raise EvidenceError(f"{role} dependency test omitted defined options") from exc
    options: dict[str, str] = {}
    for line in lines[start + 1 : end]:
        match = re.fullmatch(r"sbatch: ([a-z0-9-]+)\s+:\s+(.*)", line)
        if match is not None:
            key, raw = match.groups()
            require(key not in options, f"{role} dependency test duplicated {key}")
            options[key] = raw
    decisions = []
    for line in lines:
        match = re.fullmatch(
            r"sbatch: Job ([0-9]+) to start at (\S+) using ([0-9]+) processors "
            r"on nodes (\S+) in partition (\S+)",
            line,
        )
        if match is not None:
            decisions.append(match.groups())
    require(len(decisions) == 1, f"{role} dependency test decision differs")
    _job, _start, processors, _nodes, partition = decisions[0]
    execution = manifest["execution"]
    expected_output = (
        str(Path(str(manifest["paths"]["run_root"])) / "state/submission/logs/wave1_%A_%a.out")
        if role == "wave1"
        else str(Path(str(manifest["paths"]["run_root"])) / "state/submission/logs/report_%j.out")
    )
    expected = {
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "cpus-per-task": str(execution["cpus_per_task"]),
        "comment": f"treewm-exp23:{dependency.split(':', 1)[1] if False else ''}",
        "export": "NONE",
        "job-name": "",
        "mem": str(execution["memory_per_task"]),
        "nodes": "1",
        "ntasks-per-node": "1",
        "open-mode": "a",
        "output": expected_output,
        "parsable": "set",
        "partition": str(
            execution["gpu_partitions"] if role == "wave1" else execution["cpu_partition"]
        ),
        "qos": "normal",
        "test-only": "set",
        "time": str(execution["walltime"]),
        "verbose": "3",
        "dependency": dependency,
        "kill-on-invalid-dep": "yes",
    }
    if role == "wave1":
        expected.update(
            {
                "array": "0-19%20",
                "gpus-per-node": str(execution["gpus_per_task"]),
                "no-requeue": "no-requeue",
                "signal": f"B:USR1@{execution['signal_seconds_before_end']}",
            }
        )
    # Job name/comment are validated separately from the context because they are
    # derived from the post-contract submission hash, not the manifest.
    expected["comment"] = options.get("comment", "")
    expected["job-name"] = options.get("job-name", "")
    require(
        exact_json_equal(options, expected),
        f"{role} dependency test options differ",
    )
    require(
        int(processors) == int(execution["cpus_per_task"]),
        f"{role} dependency test processor decision differs",
    )
    allowed = (
        str(execution["gpu_partitions"]).split(",")
        if role == "wave1"
        else [str(execution["cpu_partition"])]
    )
    require(partition in allowed, f"{role} dependency test partition differs")
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


def _validate_dependency(
    value: object,
    *,
    role: str,
    predecessor_id: str,
    control_client: str,
    job_id: str,
    job_name: str,
    comment: str,
    expected_control: Mapping[str, Any],
) -> None:
    require(
        isinstance(value, Mapping)
        and set(value)
        == {
            "command",
            "returncode",
            "stdout",
            "stderr",
            "dependency",
            "role",
            "kill_on_invalid_dependency",
            "scheduler_control_plane",
        },
        f"{role} accepted dependency fields differ",
    )
    dependency = f"afterok:{predecessor_id}_*(unfulfilled)"
    require(
        value.get("command")
        == [control_client, "show", "job", job_id, "--oneliner"]
        and type(value.get("returncode")) is int
        and value.get("returncode") == 0
        and isinstance(value.get("stdout"), str)
        and isinstance(value.get("stderr"), str)
        and len(value["stdout"]) <= 1024 * 1024
        and len(value["stderr"]) <= 1024 * 1024
        and value.get("dependency") == dependency
        and value.get("role") == role
        and value.get("kill_on_invalid_dependency") == "Yes"
        and _field(value["stdout"], "JobId", role) == job_id
        and _field(value["stdout"], "JobName", role) == job_name
        and _field(value["stdout"], "Comment", role) == comment
        and _field(value["stdout"], "JobState", role) == "PENDING"
        and _field(value["stdout"], "Dependency", role) == dependency
        and _field(value["stdout"], "KillOInInvalidDependent", role) == "Yes",
        f"{role} accepted dependency differs",
    )
    _validate_control(value.get("scheduler_control_plane"), expected_control, role)


def validate_dag_records(
    records: Mapping[str, Any],
    calling_records: Mapping[str, Any],
    calling_sha256_by_role: Mapping[str, str],
    *,
    manifest: Mapping[str, Any],
    snapshot_root: Path,
    submission_root: Path,
    submission_sha256: str,
    claim_token: str,
    job_ids: Mapping[str, str],
    expected_control_plane: Mapping[str, Any],
    expected_lock: Mapping[str, Any] | None = None,
    allow_prefix: bool = False,
) -> str:
    roles = tuple(role for role in ROLES if role in records)
    allowed_roles = tuple(ROLES[: len(roles)])
    require(
        roles == allowed_roles
        and set(records) == set(roles)
        and set(calling_records) == set(roles)
        and set(calling_sha256_by_role) == set(roles)
        and all(
            isinstance(calling_sha256_by_role[role], str)
            and re.fullmatch(r"[0-9a-f]{64}", calling_sha256_by_role[role])
            is not None
            for role in roles
        )
        and (allow_prefix or roles == ROLES)
        and bool(roles),
        "accepted record roles differ",
    )
    require(
        set(job_ids) == set(roles)
        and all(
            isinstance(job_ids[role], str)
            and re.fullmatch(r"[1-9][0-9]*", job_ids[role]) is not None
            for role in roles
        )
        and len(set(job_ids.values())) == len(roles),
        "DAG job IDs differ",
    )
    normalized_job_ids = {
        **{"wave0": "999999990", "wave1": "999999991", "report": "999999992"},
        **dict(job_ids),
    }
    commands, test_commands, names, comment = _paths(
        manifest,
        snapshot_root,
        submission_root,
        submission_sha256,
        normalized_job_ids,
    )
    validated_calling: dict[str, Mapping[str, Any]] = {}
    for role in roles:
        require(
            sealed_json_sha256(calling_records[role])
            == calling_sha256_by_role[role]
            == records[role].get("calling_sha256"),
            f"{role} calling artifact bytes differ",
        )
        validated_calling[role] = _validate_calling(
            calling_records[role],
            role=role,
            command=commands[role],
            job_name=names[role],
            comment=comment,
            submission_root=submission_root,
            submission_sha256=submission_sha256,
            claim_token=claim_token,
            expected_lock=expected_lock,
        )
        record = _validate_base_record(
            records[role],
            role=role,
            job_id=normalized_job_ids[role],
            command=commands[role],
            calling=validated_calling[role],
            expected_control=expected_control_plane,
        )
        if role == "wave0":
            _validate_hold(
                record["accepted_hold"],
                control_client=str(manifest["execution"]["scontrol"]),
                job_id=normalized_job_ids[role],
                job_name=names[role],
                comment=comment,
                expected_control=expected_control_plane,
            )
            continue
        test = record["exact_dependency_test_only"]
        require(
            isinstance(test, Mapping)
            and set(test)
            == {
                "command",
                "returncode",
                "stdout",
                "stderr",
                "parsed",
                "scheduler_control_plane",
                "zero_job_after_test",
            }
            and test.get("command") == test_commands[role]
            and type(test.get("returncode")) is int
            and test.get("returncode") == 0
            and test.get("stdout") == ""
            and isinstance(test.get("stderr"), str)
            and test.get("zero_job_after_test") is True,
            f"{role} dependency-test evidence differs",
        )
        _validate_control(
            test.get("scheduler_control_plane"), expected_control_plane, f"{role} test"
        )
        predecessor = normalized_job_ids["wave0" if role == "wave1" else "wave1"]
        parsed = _parsed_test_only(
            test["stderr"],
            role=role,
            manifest=manifest,
            dependency=f"afterok:{predecessor}",
        )
        require(
            exact_json_equal(test.get("parsed"), parsed),
            f"{role} parsed dependency test differs",
        )
        require(
            parsed["defined_options"]["job-name"] == names[role]
            and parsed["defined_options"]["comment"] == comment,
            f"{role} dependency-test scheduler identity differs",
        )
        _validate_dependency(
            record["accepted_dependency"],
            role=role,
            predecessor_id=predecessor,
            control_client=str(manifest["execution"]["scontrol"]),
            job_id=normalized_job_ids[role],
            job_name=names[role],
            comment=comment,
            expected_control=expected_control_plane,
        )
    require(
        len(
            {
                canonical_json(calling_records[role]["transaction_lock"])
                for role in roles
            }
        )
        == 1,
        "DAG calling records use different transaction locks",
    )
    return stable_hash(records)


def validate_calling_intent(
    value: Mapping[str, Any],
    *,
    role: str,
    manifest: Mapping[str, Any],
    snapshot_root: Path,
    submission_root: Path,
    submission_sha256: str,
    claim_token: str,
    predecessor_job_ids: Mapping[str, str],
    expected_lock: Mapping[str, Any] | None = None,
) -> str:
    require(role in ROLES, "calling intent role differs")
    normalized = {
        "wave0": "999999990",
        "wave1": "999999991",
        "report": "999999992",
        **dict(predecessor_job_ids),
    }
    commands, _tests, names, comment = _paths(
        manifest, snapshot_root, submission_root, submission_sha256, normalized
    )
    _validate_calling(
        value,
        role=role,
        command=commands[role],
        job_name=names[role],
        comment=comment,
        submission_root=submission_root,
        submission_sha256=submission_sha256,
        claim_token=claim_token,
        expected_lock=expected_lock,
    )
    return sealed_json_sha256(value)
