#!/usr/bin/env python3
"""Seal validator and externally gated publisher for Exp24 calibration checkpoints.

Synthetic validation only emits a non-production candidate to stdout.  Durable lock
publication additionally requires a manifest-pinned outer writer-closure authority;
an advisory flock or repeated pathname check alone is never treated as immutability.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import calibration_contract as contract
import calibration_controller as controller
import calibration_worker as worker


FORBIDDEN_LIVE_NAMES = {
    "COMPLETED.json",
    "final_eval_progress.json",
    "FAILED.json",
    "FAILURE.json",
    "CANCELLED.json",
    "ADVERSE.json",
    "FORMAL_REUSE.json",
}
LOCK_FIELDS = {
    "schema_version", "status", "campaign_id", "authority_sha256",
    "validation_profile", "production_ready",
    "runtime_lock_sha256", "model_state_authority_sha256",
    "result_creation_receipt_sha256", "scheduler_terminal_census_sha256", "roots",
    "seed_collision_census_sha256", "result_root", "result_root_identity",
    "completed_updates", "expected_checkpoints", "inventory", "inventory_sha256",
    "model_parameter_schema_by_setting", "fixed_weight_audit",
    "formal_training_or_resume_reuse_allowed", "launch_authorization_sha256",
    "publication_boundary_authority_sha256",
    "seal_sha256",
}
LOCK_PRODUCTION_STATUS = "sealed_exp24_all_ten_zero_prefix_checkpoint_source"
LOCK_PRODUCTION_PROFILE = "externally_pinned_production_authorities_v1"
LOCK_SYNTHETIC_STATUS = (
    "validated_synthetic_exp24_checkpoint_source_candidate_never_publication_ready"
)
LOCK_SYNTHETIC_PROFILE = "production_shaped_validator_fixture_only_v1"
PUBLICATION_BOUNDARY_STATUS = (
    "sealed_exp24_checkpoint_lock_publication_boundary_v1"
)
PUBLICATION_BOUNDARY_FIELDS = {
    "schema_version", "status", "campaign_id", "manifest_file_sha256",
    "scratch_root", "scratch_root_identity", "publication_parent_relative_path",
    "publication_parent_identity", "terminal_census_sha256",
    "writer_closure_status", "external_writer_closure_sha256",
    "external_anchor_sha256", "production_ready",
    "publication_boundary_authority_sha256",
}
INVENTORY_FIELDS = {
    "cell_index", "setting_id", "env_config", "seed", "run_name",
    "completion_wave", "receipt_relative_path", "receipt_sha256",
    "checkpoint_relative_path", "checkpoint_raw_sha256", "checkpoint_size",
    "run_identity_sha256", "model_parameter_schema_sha256",
    "model_parameter_tensor_count", "model_parameter_total_numel",
    "resolved_config_contract_sha256", "launch_relative_path",
    "launch_source_sha256", "launch_sha256",
    "runtime_lock_sha256", "model_state_authority_sha256",
    "result_creation_receipt_sha256", "scheduler_terminal_census_sha256",
    "expected_model_parameter_schema_sha256",
}
CHECKPOINT_LOCK_RELATIVE_PATH = (
    f"locks/{contract.CAMPAIGN_ID}/checkpoint_source.lock.json"
)


def _production_prerequisites_match(
    manifest: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
    terminal_census: Mapping[str, Any],
    publication_boundary_authority: Mapping[str, Any] | None,
) -> bool:
    external = manifest["external_prerequisites"]
    pins = (
        external["runtime_content_lock"]["production_lock_sha256"],
        external["model_state_authority"]["production_authority_sha256"],
        external["scheduler_terminal_census"]["production_lock_sha256"],
        external["lock_publication_boundary"]["production_lock_sha256"],
    )
    if not all(type(value) is str and contract.SHA256.fullmatch(value) is not None
               for value in pins):
        return False
    if publication_boundary_authority is None:
        return False
    validate_publication_boundary_authority(
        publication_boundary_authority, manifest, terminal_census,
        external["lock_publication_boundary"]["production_lock_sha256"],
    )
    return (
        runtime_lock.get("production_ready") is True
        and runtime_lock.get("runtime_lock_sha256") == pins[0]
        and model_authority.get("production_ready") is True
        and model_authority.get("model_state_authority_sha256") == pins[1]
        and terminal_census.get("production_ready") is True
        and terminal_census.get("terminal_census_sha256") == pins[2]
        and publication_boundary_authority.get(
            "publication_boundary_authority_sha256"
        ) == pins[3]
    )


def validate_publication_boundary_authority(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    terminal_census: Mapping[str, Any],
    expected_sha256: object,
) -> Mapping[str, Any]:
    """Consume an outer writer-closure authority; this leaf cannot infer it."""
    contract.require_exact_keys(
        value, PUBLICATION_BOUNDARY_FIELDS, "lock publication boundary authority"
    )
    contract.require_int(value["schema_version"], "publication boundary schema")
    contract.require(
        value["schema_version"] == 1
        and value["status"] == PUBLICATION_BOUNDARY_STATUS
        and value["campaign_id"] == contract.CAMPAIGN_ID
        and value["manifest_file_sha256"]
        == contract.file_sha256(contract.MANIFEST_PATH),
        "lock publication boundary header differs",
    )
    contract.require(value["production_ready"] is True
                     and type(value["production_ready"]) is bool,
                     "lock publication boundary is not production-ready")
    contract.require(
        value["writer_closure_status"]
        == "outer_transaction_terminal_no_owner_writer_capabilities_v1",
        "lock publication writer closure is unavailable",
    )
    for name in (
        "external_writer_closure_sha256", "external_anchor_sha256",
        "publication_boundary_authority_sha256",
    ):
        contract.require_sha256(value[name], f"lock publication boundary {name}")
    contract.require_sha256(expected_sha256,
                            "manifest publication boundary authority SHA256")
    contract.require(
        value["publication_boundary_authority_sha256"] == expected_sha256,
        "lock publication boundary differs from the external manifest pin",
    )
    contract.require(
        value["terminal_census_sha256"]
        == terminal_census["terminal_census_sha256"],
        "lock publication boundary terminal census differs",
    )
    scratch = Path(contract.require_string(value["scratch_root"], "boundary scratch"))
    contract.require(scratch.is_absolute() and ".." not in scratch.parts,
                     "lock publication boundary scratch path differs")
    relative = contract.safe_relative(
        value["publication_parent_relative_path"],
        "lock publication boundary parent path",
    )
    contract.require(
        str(relative) == str(PurePosixPath(CHECKPOINT_LOCK_RELATIVE_PATH).parent),
        "lock publication boundary parent relative path differs",
    )
    with contract.DirectoryCapability(
        scratch, "lock publication boundary scratch"
    ) as scratch_authority:
        contract.require(
            contract.canonical_json(value["scratch_root_identity"])
            == contract.canonical_json(
                contract.directory_identity(scratch_authority.before)
            ),
            "lock publication boundary scratch identity differs",
        )
        parent = scratch_authority.open_directory(relative)
        try:
            contract.require(
                contract.canonical_json(value["publication_parent_identity"])
                == contract.canonical_json(
                    contract.directory_identity(os.fstat(parent))
                ),
                "lock publication boundary parent identity differs",
            )
            scratch_authority.require_directory_identity(
                relative, parent, "lock publication boundary parent"
            )
        finally:
            os.close(parent)
    body = dict(value)
    claimed = body.pop("publication_boundary_authority_sha256")
    contract.require(
        claimed == contract.stable_hash(body),
        "lock publication boundary self-hash differs",
    )
    return value


class SecureRoot:
    """Frozen result authority retaining every capability through consumption."""

    def __init__(self, path: Path) -> None:
        contract.require(path.is_absolute(), "result root is not absolute")
        normalized = Path(os.path.normpath(str(path)))
        self.path = normalized
        self.tree = contract.RetainedTree(
            normalized,
            "governed result tree",
            directory_mode=0o555,
            file_mode=0o444,
            lock_exclusive=True,
        )
        self.fd = self.tree.root.fd
        self.before = self.tree.root.before
        self._closed = False
        self._initial_snapshot = self.tree.inventory

    def close(self) -> None:
        if not self._closed:
            try:
                first = self._scan_recursive()
                second = self._scan_recursive()
                contract.require(
                    contract.canonical_json(first)
                    == contract.canonical_json(self._initial_snapshot),
                    "governed result tree changed in first final scan",
                )
                contract.require(
                    contract.canonical_json(second)
                    == contract.canonical_json(self._initial_snapshot)
                    == contract.canonical_json(first),
                    "governed result tree changed in second final scan",
                )
                self.tree._verify_retained()
            finally:
                self.tree._release(verify=False)
                self._closed = True

    def __enter__(self) -> "SecureRoot":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        # Successful validation retains authority through lock construction and
        # schema validation; validate_result_root closes it in its finalizer.
        if exc_type is not None:
            try:
                self.close()
            except BaseException:
                pass

    def _directory_fd(self, relative: PurePosixPath | str) -> int:
        return os.dup(self.tree.descriptor_for_directory(relative))

    def _scan_recursive(self) -> dict[str, Any]:
        snapshot = self.tree._fresh_scan()
        self.tree._verify_retained()
        return snapshot

    def list_directory(self, relative: PurePosixPath | str, *, mode: int | None = None) -> list[str]:
        descriptor = self.tree.descriptor_for_directory(relative)
        before = os.fstat(descriptor)
        if mode is not None:
            contract.require(stat.S_IMODE(before.st_mode) == mode,
                             f"{relative} directory mode differs")
        self.tree._verify_retained()
        return self.tree.list_directory(relative)

    def read_regular(
        self,
        relative: PurePosixPath | str,
        *,
        mode: int = 0o444,
        max_bytes: int = 16 * 1024 * 1024,
    ) -> tuple[bytes, os.stat_result]:
        contract.require(mode == 0o444, "sealed read mode differs")
        return self.tree.read_regular(relative, max_bytes=max_bytes)

    def hash_regular(self, relative: PurePosixPath | str) -> tuple[str, int]:
        path = contract.safe_relative(str(relative), "sealed checkpoint path")
        descriptor, before, _digest = self.tree._files[str(path)]
        contract.require(0 < before.st_size <= 16 * 1024**3,
                         f"sealed checkpoint size differs: {path}")
        digest, size = contract.hash_descriptor(
            descriptor, before, f"sealed checkpoint {path}"
        )
        self.tree._verify_retained()
        return digest, size

    def checkpoint_view(
        self, relative: PurePosixPath | str, decoder: Any
    ) -> "SealedCheckpointView":
        return SealedCheckpointView(self, contract.safe_relative(
            str(relative), "sealed checkpoint path"
        ), decoder)


class SealedCheckpointView:
    """Torch inspection view over the one FD retained by SecureRoot."""

    def __init__(
        self, root: SecureRoot, relative: PurePosixPath, decoder: Any
    ) -> None:
        self.root = root
        self.relative = relative
        self.fd, self.before, self.initial_sha256 = root.tree._files[str(relative)]
        contract.require(0 < self.before.st_size <= 16 * 1024**3,
                         "sealed checkpoint size is outside bound")
        self.safe_load_evidence: dict[str, Any] | None = None
        self.decoder = decoder

    def verify(self) -> None:
        digest, _size = contract.hash_descriptor(
            self.fd, self.before, f"sealed checkpoint {self.relative}"
        )
        contract.require(digest == self.initial_sha256,
                         f"sealed checkpoint bytes changed: {self.relative}")
        path = self.relative
        parent = self.root.tree.descriptor_for_directory(path.parent)
        contract.require(
            contract.stat_identity(os.stat(path.name, dir_fd=parent, follow_symlinks=False))
            == contract.stat_identity(self.before),
            f"sealed checkpoint pathname changed: {path}",
        )

    def load(self) -> Mapping[str, Any]:
        payload, evidence = contract.safe_load_checkpoint_fd(
            self.fd,
            self.before,
            f"sealed calibration checkpoint {self.relative}",
            verify=self.verify,
            decoder=self.decoder,
        )
        self.safe_load_evidence = evidence
        return payload

    def sha256(self) -> tuple[str, int]:
        self.verify()
        return self.initial_sha256, int(self.before.st_size)


def _read_json(root: SecureRoot, relative: str, label: str) -> dict[str, Any]:
    payload, _ = root.read_regular(relative, mode=0o444)
    return contract.parse_json_bytes(payload, label)


def _validate_marker_hash(value: Mapping[str, Any], label: str) -> None:
    body = dict(value)
    claimed = body.pop("marker_sha256", None)
    contract.require_sha256(claimed, f"{label} SHA256")
    contract.require(claimed == contract.stable_hash(body), f"{label} self-hash differs")


def _scan_live_names(root: SecureRoot) -> None:
    def walk(relative: PurePosixPath) -> None:
        descriptor = root._directory_fd(relative)
        try:
            names = sorted(os.listdir(descriptor))
            for name in names:
                contract.require(name not in FORBIDDEN_LIVE_NAMES,
                                 f"forbidden terminal/adverse marker exists: {relative / name}")
                contract.require(not name.endswith(".pt"),
                                 f"unsealed checkpoint source exists: {relative / name}")
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                contract.require(not stat.S_ISLNK(info.st_mode),
                                 f"symlink exists in calibration live tree: {relative / name}")
                if stat.S_ISDIR(info.st_mode):
                    walk(relative / name)
                else:
                    contract.require(stat.S_ISREG(info.st_mode),
                                     f"special file exists in calibration live tree: {relative / name}")
        finally:
            os.close(descriptor)

    walk(PurePosixPath("live-runs"))


def _validate_receipt_semantics(
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    cell: contract.CalibrationCell,
    result_root: str | Path,
    expected_launch: Mapping[str, Any],
) -> None:
    worker.validate_receipt(
        receipt,
        manifest,
        authority,
        cell,
        result_root,
        expected_launch,
    )


def _publish_lock_exclusive(
    path: Path,
    value: Mapping[str, Any],
    authority: Mapping[str, Any],
    result_authority: SecureRoot,
) -> None:
    """Publish under the outer authority, detecting and cleaning persistent rebinds.

    This helper does not claim that advisory flock prevents a same-owner transient
    rename.  The public production path calls it only after consuming the separately
    pinned writer-capability closure.  A detected rebind removes the unpublished inode.
    """
    scratch = Path(authority["environment"]["scratch_root"])
    expected = scratch / CHECKPOINT_LOCK_RELATIVE_PATH
    normalized = Path(os.path.normpath(str(path.absolute())))
    contract.require(normalized == expected,
                     "checkpoint lock publication path differs")
    contract.lexical_descendant(normalized, scratch, "checkpoint lock publication")
    formal = Path(authority["environment"]["formal_output_root"])
    contract.require(normalized != formal and not normalized.is_relative_to(formal),
                     "checkpoint lock publication enters formal output")
    result_authority.tree._verify_retained()
    scratch_authority = contract.DirectoryCapability(
        scratch, "checkpoint lock publication scratch root"
    )
    parent = contract.DirectoryCapability(
        normalized.parent, "checkpoint lock publication parent"
    )
    descriptor: int | None = None
    created = False
    try:
        relative = contract.safe_relative(
            CHECKPOINT_LOCK_RELATIVE_PATH, "checkpoint lock publication relative path"
        )
        contract.require(
            contract.directory_identity(scratch_authority.before)
            == authority["environment"]["scratch_root_identity"],
            "checkpoint lock publication scratch identity differs",
        )
        scratch_authority.require_directory_identity(
            relative.parent, parent.fd, "checkpoint lock publication parent"
        )
        try:
            fcntl.flock(parent.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise contract.CalibrationContractError(
                "checkpoint lock publication parent is not exclusively quiescent"
            ) from exc
        parent_info = os.fstat(parent.fd)
        contract.require(parent_info.st_uid == os.getuid()
                         and stat.S_IMODE(parent_info.st_mode) == 0o700,
                         "checkpoint lock publication parent is not owner-only")
        descriptor = os.open(
            normalized.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent.fd,
        )
        created = True
        payload = json.dumps(
            dict(value), sort_keys=True, indent=2, allow_nan=False
        ).encode("utf-8") + b"\n"
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            contract.require(count > 0, "checkpoint lock publication write stopped")
            offset += count
        os.fsync(descriptor)
        scratch_authority.require_directory_identity(
            relative.parent, parent.fd, "checkpoint lock publication parent"
        )
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        contract.require(before.st_nlink == 1
                         and stat.S_IMODE(before.st_mode) == 0o444,
                         "published checkpoint lock mode/link differs")
        digest, size = contract.hash_descriptor(
            descriptor, before, "published checkpoint source lock"
        )
        contract.require(size == len(payload)
                         and digest == hashlib.sha256(payload).hexdigest()
                         and os.pread(descriptor, len(payload) + 1, 0) == payload,
                         "published checkpoint lock bytes differ")
        named = os.stat(normalized.name, dir_fd=parent.fd, follow_symlinks=False)
        contract.require(contract.stat_identity(named) == contract.stat_identity(before),
                         "published checkpoint lock pathname changed")
        scratch_authority.require_directory_identity(
            relative.parent, parent.fd, "checkpoint lock publication parent"
        )
        os.fsync(parent.fd)
        scratch_authority.require_directory_identity(
            relative.parent, parent.fd, "checkpoint lock publication parent"
        )
        result_authority.tree.verify_two_scans()
        scratch_authority.require_directory_identity(
            relative.parent, parent.fd, "checkpoint lock publication parent"
        )
    except BaseException:
        if created and descriptor is not None:
            try:
                named = os.stat(normalized.name, dir_fd=parent.fd, follow_symlinks=False)
                if contract.stat_identity(named) == contract.stat_identity(os.fstat(descriptor)):
                    os.unlink(normalized.name, dir_fd=parent.fd)
                    os.fsync(parent.fd)
            except OSError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            fcntl.flock(parent.fd, fcntl.LOCK_UN)
        except OSError:
            pass
        parent.close()
        scratch_authority.close()


def validate_result_root(
    result_root: str | Path,
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
    terminal_census: Mapping[str, Any],
    launch_authorization: Mapping[str, Any],
    *,
    publication_boundary_authority: Mapping[str, Any] | None = None,
    publication_path: str | Path | None = None,
    _regeneration_only: bool = False,
) -> dict[str, Any]:
    contract.validate_manifest(manifest)
    contract.require(publication_path is not None or _regeneration_only,
                     "exclusive fsynced lock publication path is required")
    environment = authority["environment"]
    expected_result = (
        Path(environment["scratch_root"]) / environment["result_relative_path"]
    )
    observed_result = Path(os.path.normpath(str(Path(result_root))))
    contract.require(observed_result == expected_result,
                     "result root differs from authorized scratch-relative root")
    contract.lexical_descendant(observed_result, environment["scratch_root"], "result root")
    formal_root = Path(environment["formal_output_root"])
    contract.require(observed_result != formal_root
                     and not observed_result.is_relative_to(formal_root),
                     "result root enters formal output namespace")
    contract.require(launch_authorization["result_root"] == str(observed_result),
                     "launch authorization result root differs")
    with SecureRoot(observed_result) as root, contract.RuntimeCheckpointDecoder(
        runtime_lock,
        manifest,
        allow_synthetic=True,
    ) as decoder:
        # The retained result-root EX|NB lock is acquired before any authority,
        # launch, terminal, receipt, or checkpoint claim is accepted.
        contract.validate_authority(authority, manifest, runtime_lock)
        creation_receipt = controller.validate_result_creation_receipt(
            root.tree.root, manifest, authority, runtime_lock, model_authority
        )
        launch_rows = controller.validate_launch_authorization(
            launch_authorization, manifest, authority, runtime_lock, model_authority
        )
        contract.validate_terminal_census(
            terminal_census,
            manifest,
            authority,
            runtime_lock,
            model_authority,
            creation_receipt["result_creation_receipt_sha256"],
        )
        production_ready = _production_prerequisites_match(
            manifest, runtime_lock, model_authority, terminal_census,
            publication_boundary_authority,
        )
        if publication_path is not None:
            contract.require_production_runtime(
                runtime_lock, "checkpoint-source lock publication"
            )
            contract.require_production_model_authority(
                model_authority, "checkpoint-source lock publication"
            )
            contract.require(
                terminal_census.get("production_ready") is True
                and terminal_census.get("external_anchor_sha256") is not None,
                "checkpoint-source lock publication requires externally anchored terminal evidence",
            )
            contract.require(production_ready,
                             "production prerequisite hashes are absent or differ")
        result_root_identity = contract.directory_identity(root.before)
        contract.require(stat.S_IMODE(root.before.st_mode) == 0o555,
                         "result root is not frozen mode 0555")
        contract.require(root.list_directory(".") == [
            controller.RESULT_CREATION_RECEIPT_PATH,
            "control", "live-runs", "sealed-cells", "sealed-checkpoints"
        ], "result root inventory differs")
        expected_files = [f"cell-{index:03d}.json" for index in range(20)]
        expected_checkpoints = [f"cell-{index:03d}.pt" for index in range(20)]
        expected_control = [f"cell-{index:03d}" for index in range(20)]
        contract.require(root.list_directory("sealed-cells", mode=0o555) == expected_files,
                         "sealed cell inventory is missing or has extras")
        contract.require(
            root.list_directory("sealed-checkpoints", mode=0o555) == expected_checkpoints,
            "sealed checkpoint inventory is missing or has extras",
        )
        contract.require(root.list_directory("control", mode=0o555) == expected_control,
                         "calibration control inventory is missing or has extras")
        root.list_directory("live-runs", mode=0o555)
        _scan_live_names(root)

        inventory: list[dict[str, Any]] = []
        checkpoint_hashes: set[str] = set()
        run_identity_hashes: set[str] = set()
        schema_by_setting: dict[str, str] = {}
        for cell in contract.expand_cells(manifest):
            launch_row = launch_rows[cell.index]
            launch = launch_row["launch"]
            receipt_relative = f"sealed-cells/cell-{cell.index:03d}.json"
            receipt = _read_json(root, receipt_relative, f"cell {cell.index} receipt")
            _validate_receipt_semantics(
                receipt,
                manifest,
                authority,
                cell,
                result_root,
                launch,
            )
            checkpoint_relative = receipt["checkpoint_relative_path"]
            contract.require(
                checkpoint_relative == f"sealed-checkpoints/cell-{cell.index:03d}.pt",
                f"cell {cell.index} checkpoint path differs",
            )
            checkpoint = root.checkpoint_view(checkpoint_relative, decoder)
            inspected = worker._inspect_retained_checkpoint(
                checkpoint, launch, authority
            )
            raw_sha = inspected["checkpoint_raw_sha256"]
            size = inspected["checkpoint_size"]
            contract.require(
                raw_sha == receipt["checkpoint_raw_sha256"]
                and size == receipt["checkpoint_size"],
                f"cell {cell.index} checkpoint bytes/size differ",
            )
            for name in (
                "run_identity_sha256", "model_parameter_schema",
                "model_parameter_schema_sha256",
                "model_parameter_tensor_count", "model_parameter_total_numel",
                "resolved_config_contract_sha256",
            ):
                contract.require(
                    contract.canonical_json(receipt[name])
                    == contract.canonical_json(inspected[name]),
                    f"cell {cell.index} checkpoint-derived {name} differs",
                )
            contract.require(
                contract.canonical_json(receipt["run_identity"])
                == contract.canonical_json(inspected["run_identity"]),
                f"cell {cell.index} checkpoint-derived run identity differs",
            )
            contract.require(
                contract.canonical_json(receipt["resolved_config"])
                == contract.canonical_json(inspected["resolved_config"]),
                f"cell {cell.index} checkpoint-derived resolved config differs",
            )
            contract.require(raw_sha not in checkpoint_hashes,
                             f"cell {cell.index} checkpoint bytes are duplicated")
            contract.require(receipt["run_identity_sha256"] not in run_identity_hashes,
                             f"cell {cell.index} run identity is duplicated")
            checkpoint_hashes.add(raw_sha)
            run_identity_hashes.add(receipt["run_identity_sha256"])
            previous_schema = schema_by_setting.setdefault(
                cell.setting_id, receipt["model_parameter_schema_sha256"]
            )
            contract.require(previous_schema == receipt["model_parameter_schema_sha256"],
                             f"{cell.setting_id} parameter schema differs across seeds")

            control_relative = PurePosixPath("control") / f"cell-{cell.index:03d}"
            contract.require(root.list_directory(control_relative, mode=0o555)
                             == ["wave-0.json", "wave-1.json"],
                             f"cell {cell.index} lifecycle inventory differs")
            wave0 = _read_json(root, str(control_relative / "wave-0.json"),
                               f"cell {cell.index} wave-zero marker")
            wave1 = _read_json(root, str(control_relative / "wave-1.json"),
                               f"cell {cell.index} wave-one marker")
            marker_launch = launch
            if wave0.get("status") == worker.COMPLETE_STATUS:
                worker.validate_marker(
                    wave0, marker_launch, worker.COMPLETE_STATUS, 0,
                    receipt_sha256=receipt["receipt_sha256"],
                )
                worker.validate_marker(
                    wave1, marker_launch, worker.NOOP_STATUS, 1,
                    receipt_sha256=receipt["receipt_sha256"],
                )
                completion_wave = 0
            else:
                worker.validate_marker(
                    wave0, marker_launch, worker.CONTINUATION_STATUS, 0,
                    inspected=receipt,
                )
                worker.validate_marker(
                    wave1, marker_launch, worker.COMPLETE_STATUS, 1,
                    receipt_sha256=receipt["receipt_sha256"],
                )
                completion_wave = 1
            contract.require(receipt["wave_index"] == completion_wave,
                             f"cell {cell.index} receipt completion wave differs")
            inventory.append({
                "cell_index": cell.index,
                "setting_id": cell.setting_id,
                "env_config": cell.env_config,
                "seed": cell.seed,
                "run_name": cell.run_name,
                "completion_wave": completion_wave,
                "receipt_relative_path": receipt_relative,
                "receipt_sha256": receipt["receipt_sha256"],
                "checkpoint_relative_path": checkpoint_relative,
                "checkpoint_raw_sha256": raw_sha,
                "checkpoint_size": size,
                "run_identity_sha256": receipt["run_identity_sha256"],
                "model_parameter_schema_sha256": receipt[
                    "model_parameter_schema_sha256"
                ],
                "model_parameter_tensor_count": receipt[
                    "model_parameter_tensor_count"
                ],
                "model_parameter_total_numel": receipt[
                    "model_parameter_total_numel"
                ],
                "resolved_config_contract_sha256": receipt[
                    "resolved_config_contract_sha256"
                ],
                "launch_relative_path": launch_row["launch_relative_path"],
                "launch_source_sha256": launch_row["launch_source_sha256"],
                "launch_sha256": launch_row["launch_sha256"],
                "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
                "model_state_authority_sha256": model_authority[
                    "model_state_authority_sha256"
                ],
                "result_creation_receipt_sha256": creation_receipt[
                    "result_creation_receipt_sha256"
                ],
                "scheduler_terminal_census_sha256": terminal_census[
                    "terminal_census_sha256"
                ],
                "expected_model_parameter_schema_sha256": launch_row[
                    "expected_model_parameter_schema_sha256"
                ],
            })
    try:
        contract.require(
            len(inventory) == len(checkpoint_hashes) == len(run_identity_hashes) == 20,
            "calibration does not prove 20 unique complete checkpoints",
        )
        contract.require(
            set(schema_by_setting) == {setting for setting, _ in contract.SETTINGS},
            "calibration parameter schemas do not cover all ten settings",
        )
        lock: dict[str, Any] = {
            "schema_version": 1,
            "status": (
                LOCK_PRODUCTION_STATUS if production_ready else LOCK_SYNTHETIC_STATUS
            ),
            "validation_profile": (
                LOCK_PRODUCTION_PROFILE if production_ready else LOCK_SYNTHETIC_PROFILE
            ),
            "production_ready": production_ready,
            "campaign_id": contract.CAMPAIGN_ID,
            "authority_sha256": authority["authority_sha256"],
            "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
            "model_state_authority_sha256": model_authority[
                "model_state_authority_sha256"
            ],
            "result_creation_receipt_sha256": creation_receipt[
                "result_creation_receipt_sha256"
            ],
            "scheduler_terminal_census_sha256": terminal_census[
                "terminal_census_sha256"
            ],
            "roots": dict(authority["roots"]),
            "seed_collision_census_sha256": authority["seed_collision_census"][
                "census_sha256"
            ],
            "result_root": str(observed_result),
            "result_root_identity": result_root_identity,
            "completed_updates": 5_000,
            "expected_checkpoints": 20,
            "inventory": inventory,
            "inventory_sha256": contract.stable_hash(inventory),
            "model_parameter_schema_by_setting": schema_by_setting,
            "fixed_weight_audit": {
                "regimes": ["exp24_zero_prefix_exact_5000", "scratch_initialization"],
                "checkpoint_seeds": [244, 245],
                "scratch_seeds": [230, 231],
                "settings": 10,
                "batches_per_setting_regime": 2,
                "expected_rows": 80,
                "device": "cpu_fp32",
                "optimizer_steps": 0,
                "weight_tuple_source": "frozen_exp24_formal_manifest",
                "retuning_allowed": False,
            },
            "formal_training_or_resume_reuse_allowed": False,
            "launch_authorization_sha256": launch_authorization[
                "authorization_sha256"
            ],
            "publication_boundary_authority_sha256": (
                publication_boundary_authority[
                    "publication_boundary_authority_sha256"
                ] if production_ready else None
            ),
        }
        lock["seal_sha256"] = contract.stable_hash(lock)
        _validate_lock_schema(
            lock, manifest, authority, runtime_lock, model_authority,
            terminal_census, launch_authorization, publication_boundary_authority
        )
        if publication_path is not None:
            _publish_lock_exclusive(
                Path(publication_path), lock, authority, root
            )
        return lock
    finally:
        root.close()


def _validate_lock_schema(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
    terminal_census: Mapping[str, Any],
    launch_authorization: Mapping[str, Any],
    publication_boundary_authority: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    contract.validate_manifest(manifest)
    contract.validate_authority(authority, manifest, runtime_lock)
    launch_rows = controller.validate_launch_authorization(
        launch_authorization, manifest, authority, runtime_lock, model_authority
    )
    contract.require_exact_keys(value, LOCK_FIELDS, "checkpoint source lock")
    contract.require(type(value.get("schema_version")) is int
                     and value.get("schema_version") == 1,
                     "checkpoint lock schema differs")
    production_ready = contract.require_bool(
        value.get("production_ready"), "checkpoint lock production readiness"
    )
    expected_ready = _production_prerequisites_match(
        manifest, runtime_lock, model_authority, terminal_census,
        publication_boundary_authority,
    )
    contract.require(production_ready is expected_ready,
                     "checkpoint lock production readiness differs")
    contract.require(
        (value.get("status"), value.get("validation_profile")) == (
            (LOCK_PRODUCTION_STATUS, LOCK_PRODUCTION_PROFILE)
            if expected_ready else (LOCK_SYNTHETIC_STATUS, LOCK_SYNTHETIC_PROFILE)
        ),
        "checkpoint source lock status/profile differs",
    )
    contract.require(
        value.get("publication_boundary_authority_sha256")
        == (
            publication_boundary_authority[
                "publication_boundary_authority_sha256"
            ] if expected_ready else None
        ),
        "checkpoint lock publication boundary binding differs",
    )
    contract.require(value.get("campaign_id") == contract.CAMPAIGN_ID,
                     "checkpoint source lock campaign differs")
    contract.require_sha256(value.get("authority_sha256"), "checkpoint lock authority SHA256")
    contract.require(value["authority_sha256"] == authority["authority_sha256"],
                     "checkpoint lock authority differs")
    creation_sha = launch_authorization["result_creation_receipt_sha256"]
    contract.validate_terminal_census(
        terminal_census, manifest, authority, runtime_lock, model_authority,
        creation_sha,
    )
    prerequisite_hashes = {
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "model_state_authority_sha256": model_authority[
            "model_state_authority_sha256"
        ],
        "result_creation_receipt_sha256": creation_sha,
        "scheduler_terminal_census_sha256": terminal_census[
            "terminal_census_sha256"
        ],
    }
    for name, expected_hash in prerequisite_hashes.items():
        contract.require_sha256(value[name], f"checkpoint lock {name}")
        contract.require(value[name] == expected_hash,
                         f"checkpoint lock {name} differs")
    roots = contract.require_exact_keys(value.get("roots"), set(contract.ROOT_NAMES),
                                        "checkpoint lock roots")
    for name in contract.ROOT_NAMES:
        contract.require_sha256(roots[name], f"checkpoint lock root {name}")
    contract.require(contract.canonical_json(roots)
                     == contract.canonical_json(authority["roots"]),
                     "checkpoint lock authority roots differ")
    contract.require_sha256(value["seed_collision_census_sha256"],
                            "checkpoint lock seed census SHA256")
    contract.require(value["seed_collision_census_sha256"]
                     == authority["seed_collision_census"]["census_sha256"],
                     "checkpoint lock seed census differs")
    contract.require_sha256(value["launch_authorization_sha256"],
                            "checkpoint lock launch authorization SHA256")
    contract.require(
        value["launch_authorization_sha256"]
        == launch_authorization["authorization_sha256"],
        "checkpoint lock launch authorization differs",
    )
    expected_result = str(
        Path(authority["environment"]["scratch_root"])
        / authority["environment"]["result_relative_path"]
    )
    contract.require(type(value["result_root"]) is str
                     and value["result_root"] == expected_result,
                     "checkpoint lock result root differs")
    result_identity = contract.require_exact_keys(
        value["result_root_identity"], {"device", "inode", "mode", "uid", "gid"},
        "checkpoint lock result root identity",
    )
    for name in ("device", "inode", "mode", "uid", "gid"):
        contract.require_int(result_identity[name],
                             f"checkpoint lock result identity {name}", minimum=0)
    contract.require(result_identity["mode"] == 0o555,
                     "checkpoint lock result root is not frozen mode 0555")
    contract.require(
        dict(result_identity)
        == contract.nofollow_directory_identity(value["result_root"], "checkpoint result root"),
        "checkpoint lock result root identity differs",
    )
    contract.require(type(value.get("completed_updates")) is int
                     and value["completed_updates"] == 5_000,
                     "checkpoint lock update differs")
    contract.require(type(value.get("expected_checkpoints")) is int
                     and value["expected_checkpoints"] == 20,
                     "checkpoint lock count differs")
    inventory = value.get("inventory")
    contract.require(isinstance(inventory, list) and len(inventory) == 20,
                     "checkpoint lock inventory does not contain 20 rows")
    cells = contract.expand_cells(manifest)
    checkpoint_hashes: set[str] = set()
    run_identity_hashes: set[str] = set()
    derived_schemas: dict[str, str] = {}
    for row, cell in zip(inventory, cells, strict=True):
        contract.require_exact_keys(row, INVENTORY_FIELDS,
                                    f"checkpoint lock cell {cell.index}")
        expected_scalars = {
            "cell_index": cell.index, "setting_id": cell.setting_id,
            "env_config": cell.env_config, "seed": cell.seed, "run_name": cell.run_name,
            "receipt_relative_path": f"sealed-cells/cell-{cell.index:03d}.json",
            "checkpoint_relative_path": f"sealed-checkpoints/cell-{cell.index:03d}.pt",
            "launch_relative_path": launch_rows[cell.index]["launch_relative_path"],
            **prerequisite_hashes,
            "expected_model_parameter_schema_sha256": contract.setting_model_schema(
                model_authority, cell.setting_id
            )["schema_sha256"],
        }
        for name, expected in expected_scalars.items():
            contract.require(type(row[name]) is type(expected) and row[name] == expected,
                             f"checkpoint lock cell {cell.index} {name} differs")
        for name in (
            "cell_index", "seed", "completion_wave", "checkpoint_size",
            "model_parameter_tensor_count", "model_parameter_total_numel",
        ):
            contract.require_int(row[name], f"checkpoint lock cell {cell.index} {name}",
                                 minimum=0)
        contract.require(row["completion_wave"] in (0, 1),
                         f"checkpoint lock cell {cell.index} completion wave differs")
        contract.require(row["checkpoint_size"] > 0
                         and row["model_parameter_tensor_count"] > 0
                         and row["model_parameter_total_numel"] > 0,
                         f"checkpoint lock cell {cell.index} has empty checkpoint/schema")
        for name in (
            "receipt_sha256", "checkpoint_raw_sha256", "run_identity_sha256",
            "model_parameter_schema_sha256", "resolved_config_contract_sha256",
            "launch_source_sha256", "launch_sha256",
            "runtime_lock_sha256", "model_state_authority_sha256",
            "result_creation_receipt_sha256", "scheduler_terminal_census_sha256",
            "expected_model_parameter_schema_sha256",
        ):
            contract.require_sha256(row[name], f"checkpoint lock cell {cell.index} {name}")
        contract.safe_relative(row["receipt_relative_path"],
                               f"checkpoint lock cell {cell.index} receipt path")
        contract.safe_relative(row["checkpoint_relative_path"],
                               f"checkpoint lock cell {cell.index} checkpoint path")
        contract.safe_relative(row["launch_relative_path"],
                               f"checkpoint lock cell {cell.index} launch path")
        contract.require(
            row["launch_source_sha256"]
            == launch_rows[cell.index]["launch_source_sha256"]
            and row["launch_sha256"] == launch_rows[cell.index]["launch_sha256"],
            f"checkpoint lock cell {cell.index} launch authority differs",
        )
        contract.require(
            row["model_parameter_schema_sha256"]
            == row["expected_model_parameter_schema_sha256"],
            f"checkpoint lock cell {cell.index} schema differs from pre-output authority",
        )
        contract.require(row["checkpoint_raw_sha256"] not in checkpoint_hashes,
                         f"checkpoint lock cell {cell.index} checkpoint hash duplicated")
        contract.require(row["run_identity_sha256"] not in run_identity_hashes,
                         f"checkpoint lock cell {cell.index} run identity duplicated")
        checkpoint_hashes.add(row["checkpoint_raw_sha256"])
        run_identity_hashes.add(row["run_identity_sha256"])
        previous = derived_schemas.setdefault(
            cell.setting_id, row["model_parameter_schema_sha256"]
        )
        contract.require(previous == row["model_parameter_schema_sha256"],
                         f"checkpoint lock {cell.setting_id} schema differs across seeds")
        expected_config_contract = contract.setting_authority(
            authority, cell.setting_id
        )["resolved_config_contract_sha256_by_seed"][str(cell.seed)]
        contract.require(
            row["resolved_config_contract_sha256"] == expected_config_contract,
            f"checkpoint lock cell {cell.index} resolved config contract differs",
        )
    contract.require_sha256(value.get("inventory_sha256"),
                            "checkpoint lock inventory SHA256")
    contract.require(value["inventory_sha256"] == contract.stable_hash(inventory),
                     "checkpoint lock inventory hash differs")
    schemas = value.get("model_parameter_schema_by_setting")
    contract.require(isinstance(schemas, Mapping)
                     and set(schemas) == {setting for setting, _ in contract.SETTINGS},
                     "checkpoint lock model schemas differ")
    for setting, digest in schemas.items():
        contract.require_sha256(digest, f"checkpoint lock {setting} parameter schema")
        contract.require(
            digest == contract.setting_model_schema(
                model_authority, setting
            )["schema_sha256"],
            f"checkpoint lock {setting} schema differs from model authority",
        )
    contract.require(contract.canonical_json(dict(schemas))
                     == contract.canonical_json(derived_schemas),
                     "checkpoint lock schema map is not derived from inventory")
    fixed_expected = {
        "regimes": ["exp24_zero_prefix_exact_5000", "scratch_initialization"],
        "checkpoint_seeds": [244, 245],
        "scratch_seeds": [230, 231],
        "settings": 10,
        "batches_per_setting_regime": 2,
        "expected_rows": 80,
        "device": "cpu_fp32",
        "optimizer_steps": 0,
        "weight_tuple_source": "frozen_exp24_formal_manifest",
        "retuning_allowed": False,
    }
    contract.require(contract.canonical_json(value.get("fixed_weight_audit"))
                     == contract.canonical_json(fixed_expected),
                     "checkpoint lock fixed-weight audit contract differs")
    fixed = value["fixed_weight_audit"]
    for name in ("settings", "batches_per_setting_regime", "expected_rows", "optimizer_steps"):
        contract.require_int(fixed[name], f"checkpoint lock fixed audit {name}", minimum=0)
    contract.require(type(fixed["retuning_allowed"]) is bool
                     and fixed["retuning_allowed"] is False,
                     "checkpoint lock permits retuning")
    contract.require(type(value.get("formal_training_or_resume_reuse_allowed")) is bool
                     and value.get("formal_training_or_resume_reuse_allowed") is False,
                     "checkpoint lock permits formal reuse")
    body = dict(value)
    claimed = body.pop("seal_sha256", None)
    contract.require_sha256(claimed, "checkpoint lock seal SHA256")
    contract.require(claimed == contract.stable_hash(body),
                     "checkpoint lock self-hash differs")
    return value


def validate_lock(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
    terminal_census: Mapping[str, Any],
    launch_authorization: Mapping[str, Any],
    publication_boundary_authority: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate both lock structure and every claim against frozen result bytes."""
    contract.require(isinstance(value, Mapping),
                     "checkpoint source lock is not a mapping")
    contract.require(
        value.get("status") in (LOCK_PRODUCTION_STATUS, LOCK_SYNTHETIC_STATUS),
        "checkpoint source lock status is invalid",
    )
    contract.require(type(value.get("result_root")) is str,
                     "checkpoint lock result root is not an exact string")
    regenerated = validate_result_root(
        value["result_root"], manifest, authority, runtime_lock, model_authority,
        terminal_census, launch_authorization,
        publication_boundary_authority=publication_boundary_authority,
        _regeneration_only=True,
    )
    contract.require(
        contract.canonical_json(value) == contract.canonical_json(regenerated),
        "checkpoint lock differs from regenerated frozen result evidence",
    )
    return value


def validate_unsealed_placeholder(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Authenticate the exact fail-closed dependency ledger shipped with this leaf."""
    expected = {
        "schema_version": 1,
        "status": "unsealed_exp24_all_ten_zero_prefix_checkpoint_source",
        "campaign_id": contract.CAMPAIGN_ID,
        "authority_sha256": None,
        "runtime_lock_sha256": None,
        "model_state_authority_sha256": None,
        "result_creation_receipt_sha256": None,
        "scheduler_terminal_census_sha256": None,
        "seed_collision_census_sha256": None,
        "roots": {name: None for name in contract.ROOT_NAMES},
        "result_root": None,
        "result_root_identity": None,
        "completed_updates": 5_000,
        "expected_checkpoints": 20,
        "inventory": [],
        "inventory_sha256": None,
        "model_parameter_schema_by_setting": {},
        "fixed_weight_audit": {
            "regimes": ["exp24_zero_prefix_exact_5000", "scratch_initialization"],
            "checkpoint_seeds": [244, 245],
            "scratch_seeds": [230, 231],
            "settings": 10,
            "batches_per_setting_regime": 2,
            "expected_rows": 80,
            "device": "cpu_fp32",
            "optimizer_steps": 0,
            "weight_tuple_source": "frozen_exp24_formal_manifest",
            "retuning_allowed": False,
        },
        "formal_training_or_resume_reuse_allowed": False,
        "launch_authorization_sha256": None,
        "publication_boundary_authority_sha256": None,
        "unsealed_prerequisites": {
            "runtime_content_lock": {
                "status": "unsealed_exp24_calibration_runtime_content_v1",
                "required_status": contract.RUNTIME_LOCK_STATUS,
                "lock_path": None,
                "runtime_root": None,
                "runtime_root_identity": None,
                "runtime_inventory_sha256": None,
                "interpreter_relative_path": None,
                "interpreter_sha256": None,
                "pyvenv_cfg_sha256": None,
                "stdlib_roots_sha256": None,
                "site_package_roots_sha256": None,
                "native_extensions_sha256": None,
                "shared_libraries_sha256": None,
                "sys_path_sha256": None,
                "loader_paths_sha256": None,
                "symlink_policy": "forbid_all_components_and_entries",
                "execution_profile": (
                    "required_complete_relocatable_authenticated_runtime_v1"
                ),
                "production_ready": False,
                "decoder_source_relative_path": "libexec/calibration_contract.py",
                "decoder_source_sha256": None,
                "closure_attestation_sha256": None,
                "decoder_isolation": (
                    "clean_exec_close_fds_hard_rlimit_canonical_json_only"
                ),
                "max_central_directory_bytes": (
                    contract.SAFE_CHECKPOINT_MAX_CENTRAL_DIRECTORY_BYTES
                ),
                "runtime_content_sha256": None,
                "runtime_lock_sha256": None,
            },
            "model_state_authority": {
                "status": "unsealed_exp24_calibration_model_state_authority_v1",
                "required_status": contract.MODEL_AUTHORITY_STATUS,
                "lock_path": None,
                "snapshot_root": None,
                "snapshot_root_identity": None,
                "snapshot_inventory_sha256": None,
                "hook_relative_path": contract.MODEL_AUTHORITY_HOOK_PATH,
                "hook_raw_sha256": None,
                "hook_interface": contract.MODEL_AUTHORITY_HOOK_INTERFACE,
                "execution_profile": contract.MODEL_UNSEALED_PROFILE,
                "production_ready": False,
                "external_hook_source_sha256": None,
                "construction": (
                    "authenticated_source_resolved_config_deterministic_zero_initialization"
                ),
                "model_parameter_schema_by_setting": {},
                "model_state_authority_sha256": None,
            },
            "result_creation_receipt": {
                "status": "unsealed_exclusive_calibration_result_creation_v1",
                "required_status": "sealed_exclusive_calibration_result_creation_v1",
                "receipt_relative_path": controller.RESULT_CREATION_RECEIPT_PATH,
                "result_root": None,
                "result_root_initial_identity": None,
                "receipt_file_identity": None,
                "creation_protocol_sha256": controller.RESULT_CREATION_PROTOCOL_SHA256,
                "result_creation_receipt_sha256": None,
            },
            "scheduler_terminal_census": {
                "status": "unsealed_exp24_calibration_scheduler_terminal_census_v1",
                "required_status": contract.TERMINAL_CENSUS_STATUS,
                "lock_path": None,
                "evidence_relative_path": None,
                "evidence_file_identity": None,
                "evidence_sha256": None,
                "required_task_rows": 40,
                "waves": [0, 1],
                "required_state": "COMPLETED",
                "required_exit_code": "0:0",
                "submission_receipt_sha256": None,
                "external_anchor_sha256": None,
                "production_ready": False,
                "required_whole_array_topology": {
                    "wave0": "0-19%20",
                    "wave1": "0-19%20",
                    "wave1_dependency": "afterok_exact_wave0_array_job_id",
                    "report_dependency": "afterok_exact_wave1_array_job_id",
                    "distinct_array_job_ids": 2,
                },
                "terminal_census_sha256": None,
            },
            "lock_publication_boundary": {
                "status": "unsealed_exp24_checkpoint_lock_publication_boundary_v1",
                "required_status": PUBLICATION_BOUNDARY_STATUS,
                "lock_path": None,
                "production_lock_sha256": None,
                "scratch_root": None,
                "scratch_root_identity": None,
                "publication_parent_relative_path": str(
                    PurePosixPath(CHECKPOINT_LOCK_RELATIVE_PATH).parent
                ),
                "publication_parent_identity": None,
                "terminal_census_sha256": None,
                "writer_closure_status": (
                    "outer_transaction_terminal_no_owner_writer_capabilities_v1"
                ),
                "external_writer_closure_sha256": None,
                "external_anchor_sha256": None,
                "production_ready": False,
            },
        },
        "production_readiness": {
            "executable": False,
            "sealable": False,
            "blocked_on": [
                "sealed_runtime_content_lock",
                "reviewed_production_model_authority_hook_and_sealed_model_state_authority",
                "sealed_result_creation_receipt",
                "sealed_external_scheduler_terminal_census",
                "sealed_external_lock_publication_boundary",
            ],
            "synthetic_fixture_is_production_authority": False,
        },
        "seal_sha256": None,
    }
    contract.require(
        contract.canonical_json(value) == contract.canonical_json(expected),
        "unsealed checkpoint dependency ledger differs",
    )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--describe", action="store_true")
    modes.add_argument("--test-only", action="store_true")
    modes.add_argument("--emit-lock", action="store_true")
    modes.add_argument("--validate-lock", action="store_true")
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--runtime-lock", type=Path)
    parser.add_argument("--model-state-authority", type=Path)
    parser.add_argument("--terminal-census", type=Path)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--launch-authorization", type=Path)
    parser.add_argument("--publication-boundary-authority", type=Path)
    parser.add_argument("--publication-path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = contract.load_manifest()
    if args.describe:
        print(json.dumps({
            "schema_version": 1,
            "status": "read_only_calibration_seal_validator_ready_production_blocked",
            "production_sealable": False,
            "blocked_on": [
                "sealed_runtime_content_lock",
                "reviewed_production_model_authority_hook_and_sealed_model_state_authority",
                "sealed_result_creation_receipt",
                "sealed_external_scheduler_terminal_census",
                "outer_launch_authorization",
                "sealed_external_lock_publication_boundary",
            ],
            "synthetic_fixture_is_production_authority": False,
            "expected_checkpoints": 20,
            "completed_updates": 5_000,
            "binds_roots": list(contract.ROOT_NAMES),
            "writes_performed": False,
            "source_or_output_scan_performed": False,
        }, sort_keys=True, indent=2, allow_nan=False))
        return 0
    if args.test_only:
        placeholder = contract.read_json(contract.PLACEHOLDER_PATH,
                                         "checkpoint source placeholder")
        validate_unsealed_placeholder(placeholder)
        print(json.dumps({
            "schema_version": 1,
            "status": "test_only_passed_placeholder_remains_unsealed",
            "production_sealable": False,
            "result_scan_performed": False,
            "writes_performed": False,
        }, sort_keys=True, indent=2, allow_nan=False))
        return 0
    if args.validate_lock:
        contract.require(args.lock is not None and args.authority is not None
                         and args.runtime_lock is not None
                         and args.model_state_authority is not None
                         and args.terminal_census is not None
                         and args.launch_authorization is not None,
                         "all sealed prerequisite locks and --lock are required")
        authority = contract.read_json(args.authority, "calibration authority")
        runtime_lock = contract.read_json(args.runtime_lock, "calibration runtime lock")
        contract.validate_authority(authority, manifest, runtime_lock)
        model_authority = contract.read_json(
            args.model_state_authority, "calibration model-state authority"
        )
        terminal_census = contract.read_json(
            args.terminal_census, "calibration scheduler terminal census"
        )
        value = contract.read_json(args.lock, "checkpoint source lock")
        launch_authorization = contract.read_json(
            args.launch_authorization, "calibration launch authorization"
        )
        publication_boundary = (
            contract.read_json(
                args.publication_boundary_authority,
                "checkpoint lock publication boundary authority",
            ) if args.publication_boundary_authority is not None else None
        )
        validate_lock(
            value, manifest, authority, runtime_lock, model_authority,
            terminal_census, launch_authorization, publication_boundary
        )
        print(json.dumps({
            "schema_version": 1,
            "status": "checkpoint_source_lock_valid",
            "seal_sha256": value["seal_sha256"],
            "writes_performed": False,
        }, sort_keys=True, indent=2, allow_nan=False))
        return 0
    contract.require(args.authority is not None and args.runtime_lock is not None
                     and args.model_state_authority is not None
                     and args.terminal_census is not None
                     and args.result_root is not None
                     and args.launch_authorization is not None
                     and args.publication_boundary_authority is not None
                     and args.publication_path is not None,
                     "all sealed prerequisites, --result-root, --launch-authorization, --publication-boundary-authority, and --publication-path are required")
    authority = contract.read_json(args.authority, "calibration authority")
    runtime_lock = contract.read_json(args.runtime_lock, "calibration runtime lock")
    contract.validate_authority(authority, manifest, runtime_lock)
    model_authority = contract.read_json(
        args.model_state_authority, "calibration model-state authority"
    )
    terminal_census = contract.read_json(
        args.terminal_census, "calibration scheduler terminal census"
    )
    launch_authorization = contract.read_json(
        args.launch_authorization, "calibration launch authorization"
    )
    publication_boundary = contract.read_json(
        args.publication_boundary_authority,
        "checkpoint lock publication boundary authority",
    )
    value = validate_result_root(
        args.result_root,
        manifest,
        authority,
        runtime_lock,
        model_authority,
        terminal_census,
        launch_authorization,
        publication_boundary_authority=publication_boundary,
        publication_path=args.publication_path,
    )
    print(json.dumps({
        "schema_version": 1,
        "status": "checkpoint_source_lock_published",
        "seal_sha256": value["seal_sha256"],
        "publication_path": str(args.publication_path),
    }, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (contract.CalibrationContractError, OSError) as exc:
        print(f"EXP24_CALIBRATION_SEAL_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
