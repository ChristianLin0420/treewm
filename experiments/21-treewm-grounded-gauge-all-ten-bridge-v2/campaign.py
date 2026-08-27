#!/usr/bin/env python3
"""Immutable contracts for the fresh Exp20-selected Exp21 all-ten bridge."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


CAMPAIGN_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CAMPAIGN_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
MANIFEST_PATH = CAMPAIGN_DIR / "manifest.json"
BINDING_PATH = CAMPAIGN_DIR / "exp20_binding.json"
PROTOCOL_LOCK_PATH = CAMPAIGN_DIR / "protocol.sha256"
CONFIG_PATH = REPOSITORY_ROOT / "configs/experiment/treewm_v2_grounded_gauge_all_ten_bridge_v2.yaml"
PINNED_PYTHON = "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RUNS = 20
STAGE_TARGET = 25_000
SEEDS = (106, 107)
TASK_IDS = (1, 2, 3, 4, 5)
SEPARATE_CLIP_TAGS = (
    "train/grad_norm_world_rest",
    "train/grad_norm_branch_transformer",
    "train/grad_clip_coefficient_world_rest",
    "train/grad_clip_coefficient_branch_transformer",
)
SETTING_IDS = (
    "scene",
    "puzzle-3x3",
    "puzzle-4x4-100m",
    "cube-double",
    "cube-triple",
    "cube-quadruple-100m",
    "antmaze-large",
    "antmaze-giant",
    "humanoidmaze-medium",
    "humanoidmaze-large",
)
EXPECTED_UNION_COUNTS = {
    "scene": (758_084, 75_816),
    "puzzle-3x3": (758_084, 75_816),
    "puzzle-4x4-100m": (1_194_586, 119_473),
    "cube-double": (758_084, 75_816),
    "cube-triple": (1_030_685, 102_824),
    "cube-quadruple-100m": (1_194_586, 119_473),
    "antmaze-large": (758_084, 75_816),
    "antmaze-giant": (759_154, 76_196),
    "humanoidmaze-medium": (955_698, 95_746),
    "humanoidmaze-large": (955_698, 95_746),
}
EXPECTED_MANIFEST_SECTION_SHA256 = {
    "prerequisite": "10eedc38468e5bef85c1be671f43b6dbc7c87d45f9847230a856ff5347a8f413",
    "method": "f173cfba0947dbb4bf41839e29fc55ebe8e576074e53e2d374af8272f8729194",
    "design": "6ee8f7db32083b9ba9514e909371c702e15e93f1018391a49b0e073a0e2f85dd",
    "selected_recipe_contract": "824f0ca90d5d3dc19e35a5b85aec795957e61d1dbd38ddcb62232c29feec9f99",
    "scientific_contract": "a6c4b96b3dac8bea5368a67473395691bba3816cabb08ec680b998802eb6ca85",
    "stage_acceptance": "46a7417948822d9fd3a0fbfc98dcbedd7c00099db62f2602e7ec218cbd78a8a7",
    "execution": "c00abf67bdd673b916e0cf0dacbe6e4f2ee16c9a0d595a04bff67e6c8d676490",
    "paths": "3e91c2ef496dc669826f59d00c74e13d66c5d8927081bcade786823755b70de1",
    "logging": "c8c7fbe6caffee57876506b6a9bb6aca21e978f33935f2e6f70426ebc76ef9a3",
    "settings": "33f4c6b9f348f5abeb8ec96d31ad2b059c47417d9f274bcd9f5d9d3f9091eaa5",
    "compatible_v2_recipe_input": "fde8338ae6f6c60daaf5813f6f5702e107c7eabdac8c8d03c955d588dd334a38",
    "claim_policy": "853c89f42a34a1a83879b40017b839a9a67825e9d1b15f893a7019fb42cee4e4",
    "lifecycle": "c6966f16e07851e915465b68e52883dfd71fd5d2b1cec3de935b20429148f7b8",
    "source_snapshot": "df327c774067f97374a5221386886b7eaa09c0ece9e15ea3b042f0fd3ff8a9d6",
}
PROTOCOL_FILES = (
    "manifest.json",
    "exp20_binding.json",
    "campaign.py",
    "bind_exp20.py",
    "train_entry.py",
    "metric_boundary.py",
    "worker.py",
    "stage_gate.py",
    "submit.py",
    "train.slurm",
    "gate.slurm",
    "README.md",
    "tests/conftest.py",
    "tests/test_exp20_binding.py",
    "tests/test_gauge_bridge.py",
)
TRAINER_SOURCE_FILES = (
    "scripts/train.py",
    "treewm/models/tree_world_model.py",
    "treewm/losses/total.py",
    "treewm/losses/multistep.py",
    "treewm/losses/expansion_losses.py",
    "treewm/losses/support_losses.py",
    "treewm/logging/metrics.py",
)


class ContractError(RuntimeError):
    """A scientific, provenance, or lifecycle contract is not exact."""


@dataclass(frozen=True)
class RunSpec:
    index: int
    setting_index: int
    seed_index: int
    setting_id: str
    env_config: str
    seed: int
    run_name: str
    wandb_id: str


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read exact JSON object {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return stable_hash(manifest)


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = read_json(path)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    require(set(manifest) == {
        "schema_version", "campaign_id", "classification", "formal_validation",
        "expected_runs", "claim_policy", "prerequisite", "method", "design",
        "selected_recipe_contract", "scientific_contract", "stage_acceptance",
        "compatible_v2_recipe_input", "source_snapshot", "lifecycle", "execution",
        "paths", "logging", "settings",
    }, "manifest top-level schema differs")
    for section, expected_hash in EXPECTED_MANIFEST_SECTION_SHA256.items():
        require(stable_hash(manifest.get(section)) == expected_hash, f"manifest {section} bytes differ")
    require(manifest.get("schema_version") == 1, "manifest schema differs")
    require(manifest.get("campaign_id") == "treewm-grounded-gauge-all-ten-bridge-v2", "campaign ID differs")
    require(manifest.get("classification") == "bounded_all_ten_gauge_bridge_v2", "classification differs")
    require(manifest.get("formal_validation") is False, "bridge claims formal validation")
    require(manifest.get("expected_runs") == RUNS, "run count differs")
    prerequisite = manifest.get("prerequisite") or {}
    require(prerequisite.get("campaign_id") == "treewm-grounded-gauge-pilot-v2-launch2", "prerequisite is not Exp20 launch2")
    require(prerequisite.get("required_status") == "accepted_for_fresh_formal_campaign_design", "Exp20 status differs")
    require(prerequisite.get("allowed_selected_arms") == ["G", "GS"], "Exp20 selected-arm set differs")
    require(prerequisite.get("selection_precedence") == ["G", "GS"], "Exp20 precedence differs")
    require(prerequisite.get("forbidden_ancestry_tokens") == [
        "exp15",
        "exp16",
        "exp18",
        "15-treewm-grounded-repair-pilot-v1",
        "treewm-grounded-repair-pilot-v1",
        "16-treewm-grounded-repair-all-ten-bridge-v1",
        "treewm-grounded-repair-all-ten-bridge-v1",
        "18-treewm-grounded-gauge-pilot-v1",
        "treewm-grounded-gauge-pilot-v1",
    ], "ancestry rejection differs")
    for key in (
        "manifest_sha256",
        "package_protocol_sha256",
        "source_sha256",
        "runtime_sha256",
        "actual_evaluation_bank_sha256",
    ):
        require(SHA256.fullmatch(str(prerequisite.get(key, ""))) is not None, f"bad Exp20 {key}")
    require(Path(str(prerequisite.get("stage_5000_gate_path", ""))).is_absolute(), "Exp20 5k path is not absolute")
    require(Path(str(prerequisite.get("acceptance_path", ""))).is_absolute(), "Exp20 acceptance path is not absolute")
    raw = prerequisite.get("raw_recomputation") or {}
    require(raw.get("settings") == ["antmaze-large", "scene", "puzzle-3x3", "puzzle-4x4-100m", "cube-quadruple-100m"], "Exp20 settings differ")
    require(raw.get("arms") == ["N", "G", "GS"] and raw.get("seeds") == [108, 109], "Exp20 raw design differs")
    require(raw.get("stage_5000_runs") == 30, "Exp20 5k count differs")
    require(raw.get("stage_25000_selected_runs") == 10 and raw.get("stage_25000_skipped_runs") == 10, "Exp20 terminal count differs")
    require(raw.get("min_scale_ratio") == 0.8 and raw.get("min_paired_mean_ratio_delta_vs_n") == 0.0, "Exp20 gauge rails differ")
    require(raw.get("min_clip_coefficient") == 0.05 and raw.get("max_clip_fraction_below_threshold") == 0.25, "Exp20 clipping rails differ")
    require(raw.get("required_nonpromotable_structural_gates") == [
        "required_finite_telemetry",
        "target_appropriate_telemetry",
        "fixed_common_validation_sample",
        "complete_recent_gradient_axis",
        "nonzero_world_gain_and_required_split_gradients",
        "valid_gradient_clip_coefficients",
        "gauge_reference_sealed_at_update_zero",
        "gauge_ratio_consistent",
        "complete_recent_gauge_axis",
    ], "Exp20 nonpromotable structural gate differs")
    require(raw.get("min_settings_with_both_seed_positive_progress") == 3, "Exp20 outcome quorum differs")

    method = manifest.get("method") or {}
    require(method == {
        "arm": "treewm",
        "model_class": "TreeWM",
        "experiment_config": "treewm_v2_grounded_gauge_all_ten_bridge_v2",
        "objective_version": "treewm_v2_grounded_gauge_all_ten_bridge_v2",
        "node_budget": 64,
        "branch_factor": 4,
        "inference_profile": "learned_guard_on",
        "scorer": "learned",
        "require_first_edge_improvement": True,
    }, "bridge method identity differs")
    design = manifest.get("design") or {}
    require(tuple(design.get("seeds") or ()) == SEEDS, "fresh bridge seeds differ")
    require(design.get("fresh_start") is True and design.get("old_checkpoints_allowed") is False, "fresh-start contract differs")
    require(design.get("no_recipe_selection_within_bridge") is True, "adaptive recipe selection enabled")
    common = manifest.get("selected_recipe_contract", {}).get("common") or {}
    require(common.get("latent_gauge_enabled") is True and common.get("latent_gauge_weight") == 1.0, "gauge recipe differs")
    require(manifest["selected_recipe_contract"].get("G") == {"separate_branch_transformer_grad_clip": False}, "G clip mode differs")
    require(manifest["selected_recipe_contract"].get("GS") == {"separate_branch_transformer_grad_clip": True}, "GS clip mode differs")
    scientific = manifest.get("scientific_contract") or {}
    require(scientific.get("optimizer_updates") == STAGE_TARGET and scientific.get("scheduler_total_steps") == 1_000_000, "bounded horizon differs")
    require(scientific.get("training_log_every_updates") == 50, "training telemetry cadence differs")
    require(scientific.get("validation_every_updates") == 1_000 and scientific.get("diagnostics_every_updates") == 1_000, "validation cadence differs")
    require(scientific.get("scheduled_sampling_granularity") == "sequence", "sequence sampling differs")
    require(scientific.get("planner_decoded_metric") == "domain_raw" and scientific.get("planner_execute_steps") == 4, "domain_raw/e4 differs")
    stage = manifest.get("stage_acceptance") or {}
    require(stage.get("required_method_runs") == RUNS, "20/20 method gate differs")
    require(stage.get("min_settings_with_both_seed_positive_progress") == 6, "all-ten progress quorum is not 6/10")
    require(stage.get("min_total_successes_per_seed") == 1 and stage.get("min_mean_distance_reduction_per_seed_exclusive") == 0.0, "per-seed outcome rails differ")
    require(stage.get("min_settings_with_both_seed_success") == 1, "replicated success rail differs")
    require(stage.get("min_scale_ratio") == 0.8, "gauge ratio rail differs")
    require(stage.get("min_clip_coefficient") == 0.05 and stage.get("max_clip_fraction_below_threshold") == 0.25, "clip rails differ")
    lifecycle = manifest.get("lifecycle") or {}
    metric_boundary = str(lifecycle.get("metric_boundary_policy", ""))
    require(
        "exact 50-update graceful requeue boundary" in metric_boundary
        and "reset only that tracker window" in metric_boundary,
        "MetricTracker boundary-recovery policy differs",
    )
    require(manifest.get("execution", {}).get("array") == "0-19%20", "array differs")
    require(manifest.get("paths", {}).get("python") == PINNED_PYTHON, "Python is not pinned")
    require(manifest.get("logging", {}).get("wandb_project") == manifest["campaign_id"], "W&B project differs")
    settings = manifest.get("settings") or []
    require(tuple(value.get("id") for value in settings) == SETTING_IDS, "setting order differs")
    require(len(settings) * len(SEEDS) == RUNS, "design does not expand to 20")
    for setting in settings:
        setting_id = str(setting["id"])
        require((setting.get("published_union_train_anchors"), setting.get("published_union_validation_anchors")) == EXPECTED_UNION_COUNTS[setting_id], f"{setting_id}: union counts differ")
        for key in ("input_contract_sha256", "calibration_sha256", "future_recipe_sha256"):
            require(SHA256.fullmatch(str(setting.get(key, ""))) is not None, f"{setting_id}: bad {key}")


def protocol_sha256(campaign_dir: str | Path = CAMPAIGN_DIR) -> str:
    root = Path(campaign_dir).resolve()
    repository = root.parents[1]
    require(len(PROTOCOL_FILES) == len(set(PROTOCOL_FILES)), "duplicate protocol inventory")
    files: dict[str, str] = {}
    for relative in PROTOCOL_FILES:
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"protocol file missing/symlinked: {path}")
        files[f"experiments/21-treewm-grounded-gauge-all-ten-bridge-v2/{relative}"] = file_sha256(path)
    config = repository / "configs/experiment/treewm_v2_grounded_gauge_all_ten_bridge_v2.yaml"
    require(config.is_file() and not config.is_symlink(), "formal bridge config missing/symlinked")
    files["configs/experiment/treewm_v2_grounded_gauge_all_ten_bridge_v2.yaml"] = file_sha256(config)
    return stable_hash({"schema_version": 1, "files": files})


def verify_protocol_lock(campaign_dir: str | Path = CAMPAIGN_DIR) -> str:
    root = Path(campaign_dir).resolve()
    try:
        locked = (root / "protocol.sha256").read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ContractError(f"protocol lock unavailable: {exc}") from exc
    live = protocol_sha256(root)
    require(SHA256.fullmatch(locked) is not None and locked == live, "protocol.sha256 is stale")
    return live


def _reject_forbidden_ancestry(value: object, tokens: Sequence[str], label: str) -> None:
    def visit(node: object) -> bool:
        if isinstance(node, Mapping):
            return any(visit(key) or visit(item) for key, item in node.items())
        if isinstance(node, (list, tuple)):
            return any(visit(item) for item in node)
        if isinstance(node, str):
            lowered = node.lower()
            return any(token.lower() in lowered for token in tokens)
        return False
    require(not visit(value), f"{label} contains forbidden Exp15/Exp16/Exp18 ancestry")


def selected_recipe(manifest: Mapping[str, Any], arm: str) -> dict[str, Any]:
    require(arm in ("G", "GS"), "selected arm is not G/GS")
    recipe = {
        "id": arm,
        "label": "gauge-with-shared-world-clipping" if arm == "G" else "gauge-with-separate-branch-transformer-clipping",
        **manifest["selected_recipe_contract"]["common"],
        **manifest["selected_recipe_contract"][arm],
    }
    return recipe


def load_exp20_binding(
    manifest: Mapping[str, Any],
    path: str | Path = BINDING_PATH,
    *,
    verify_external_files: bool = True,
) -> dict[str, Any]:
    binding = read_json(path)
    require(binding.get("schema_version") == 1, "binding schema differs")
    require(binding.get("campaign_id") == manifest["campaign_id"], "binding campaign differs")
    require(binding.get("status") == "sealed_exp20_acceptance", "Exp20 acceptance is not sealed")
    require(binding.get("launch_allowed") is True, "binding does not authorize launch")
    claimed = binding.get("binding_sha256")
    body = dict(binding)
    body.pop("binding_sha256", None)
    require(SHA256.fullmatch(str(claimed or "")) is not None and claimed == stable_hash(body), "binding self-hash differs")
    _reject_forbidden_ancestry(binding.get("exp20"), manifest["prerequisite"]["forbidden_ancestry_tokens"], "Exp20 binding")
    arm = binding.get("selected_arm")
    recipe = selected_recipe(manifest, str(arm))
    require(binding.get("selected_recipe") == recipe, "bound selected recipe differs")
    require(binding.get("selected_recipe_sha256") == stable_hash(recipe), "bound recipe hash differs")
    exp20 = binding.get("exp20") or {}
    prerequisite = manifest["prerequisite"]
    for key in ("manifest_sha256", "package_protocol_sha256", "source_sha256", "runtime_sha256", "actual_evaluation_bank_sha256"):
        require(exp20.get(key) == prerequisite[key], f"bound Exp20 {key} differs")
    evidence = exp20.get("raw_evidence")
    raw = prerequisite["raw_recomputation"]
    expected_evidence = {
        (setting, arm, seed)
        for setting in raw["settings"]
        for arm in raw["arms"]
        for seed in raw["seeds"]
    }
    require(isinstance(evidence, list) and len(evidence) == raw["stage_5000_runs"], "bound Exp20 raw evidence count differs")
    actual_evidence = {
        (str(row.get("setting_id")), str(row.get("arm_id")), int(row.get("seed", -1)))
        for row in evidence if isinstance(row, dict)
    }
    require(actual_evidence == expected_evidence and len(actual_evidence) == len(evidence), "bound Exp20 raw evidence matrix differs")
    require(exp20.get("raw_evidence_sha256") == stable_hash(evidence), "bound Exp20 raw evidence hash differs")
    if verify_external_files:
        for path_key, hash_key in (("stage_5000_gate_path", "stage_5000_gate_file_sha256"), ("acceptance_path", "acceptance_file_sha256")):
            artifact = Path(str(exp20.get(path_key, "")))
            require(artifact.is_file() and not artifact.is_symlink(), f"bound Exp20 artifact missing/symlinked: {artifact}")
            require(file_sha256(artifact) == exp20.get(hash_key), f"bound Exp20 exact bytes differ: {artifact}")
        for row in evidence:
            launch_path = Path(str(row.get("launch_path", "")))
            require(launch_path.is_file() and not launch_path.is_symlink(), f"bound Exp20 launch evidence missing/symlinked: {launch_path}")
            require(file_sha256(launch_path) == row.get("launch_file_sha256"), f"bound Exp20 launch evidence bytes differ: {launch_path}")
            launch = read_json(launch_path)
            claimed_launch = launch.get("launch_sha256")
            launch_body = dict(launch)
            launch_body.pop("launch_sha256", None)
            require(claimed_launch == row.get("launch_sha256") == stable_hash(launch_body), f"bound Exp20 launch self-hash differs: {launch_path}")
            _reject_forbidden_ancestry(launch, prerequisite["forbidden_ancestry_tokens"], "bound Exp20 launch evidence")
            event_files = row.get("event_files")
            require(isinstance(event_files, list) and event_files, "bound Exp20 event evidence list empty")
            for event in event_files:
                path = Path(str(event.get("path", "")))
                require(path.is_file() and not path.is_symlink(), f"bound Exp20 event evidence missing/symlinked: {path}")
                require(path.stat().st_size == event.get("size") and file_sha256(path) == event.get("sha256"), f"bound Exp20 event evidence bytes differ: {path}")
    return binding


def expand_runs(manifest: Mapping[str, Any]) -> list[RunSpec]:
    validate_manifest(manifest)
    runs: list[RunSpec] = []
    for setting_index, setting in enumerate(manifest["settings"]):
        for seed_index, seed in enumerate(SEEDS):
            index = setting_index * len(SEEDS) + seed_index
            name = f"gaugebridge-v2-{setting['id']}-seed{seed}"
            wandb_id = stable_hash({"campaign_id": manifest["campaign_id"], "setting_id": setting["id"], "seed": seed})[:32]
            runs.append(RunSpec(index, setting_index, seed_index, setting["id"], setting["env_config"], seed, name, wandb_id))
    require(len(runs) == RUNS and len({run.run_name for run in runs}) == RUNS, "run expansion differs")
    return runs


def run_at(manifest: Mapping[str, Any], index: int) -> RunSpec:
    require(0 <= index < RUNS, "array index must be in [0,20)")
    return expand_runs(manifest)[index]


def setting_for(manifest: Mapping[str, Any], run: RunSpec) -> Mapping[str, Any]:
    setting = manifest["settings"][run.setting_index]
    require(setting["id"] == run.setting_id, "run/setting identity differs")
    return setting


def run_directory(manifest: Mapping[str, Any], run: RunSpec) -> Path:
    return Path(manifest["paths"]["run_root"]) / run.setting_id / "treewm" / run.run_name


def recipe_root(manifest: Mapping[str, Any], setting_id: str) -> Path:
    return Path(manifest["paths"]["compatible_contract_root"]) / "future-recipes" / setting_id


def data_contract_path(manifest: Mapping[str, Any], setting_id: str) -> Path:
    return Path(manifest["paths"]["compatible_contract_root"]) / "data" / f"{setting_id}.json"


def recipe_audit_anchors(anchors: Sequence[int], sample_count: int = 257) -> list[int]:
    values = [int(value) for value in anchors]
    require(values and all(a < b for a, b in zip(values, values[1:])), "recipe anchors not ordered/unique")
    count = min(sample_count, len(values))
    positions = [(index * (len(values) - 1)) // (count - 1) for index in range(count)] if count > 1 else [0]
    selected = [values[position] for position in positions]
    require(len(selected) == len(set(selected)), "recipe audit duplicated anchors")
    return selected


def load_compatible_input(
    manifest: Mapping[str, Any],
    setting_or_run: Mapping[str, Any] | RunSpec,
    *,
    verify_files: bool = False,
) -> dict[str, Any]:
    setting = setting_for(manifest, setting_or_run) if isinstance(setting_or_run, RunSpec) else setting_or_run
    contract = read_json(data_contract_path(manifest, setting["id"]))
    claimed = contract.get("contract_sha256")
    body = dict(contract)
    body.pop("contract_sha256", None)
    require(claimed == stable_hash(body) == setting["input_contract_sha256"], f"{setting['id']}: data contract hash differs")
    legacy = manifest["compatible_v2_recipe_input"]
    expected = {
        "campaign_id": legacy["campaign_id"],
        "objective_version": legacy["objective_version"],
        "campaign_protocol_sha256": legacy["campaign_protocol_sha256"],
        "setting_id": setting["id"],
        "dataset_kind": setting["dataset_kind"],
        "calibration_sha256": setting["calibration_sha256"],
        "future_recipe_sha256": setting["future_recipe_sha256"],
        "raw_cache_read_only": True,
    }
    for key, value in expected.items():
        require(contract.get(key) == value, f"{setting['id']}: compatible input {key} differs")
    require(contract.get("train_manifest_sha256") != contract.get("validation_manifest_sha256"), f"{setting['id']}: split identities overlap")
    source_files = contract.get("source_files") or []
    train = [row for row in source_files if row.get("split") == "train"]
    validation = [row for row in source_files if row.get("split") in {"val", "validation"}]
    require(train and validation, f"{setting['id']}: split coverage incomplete")
    require(not {row.get("path") for row in train}.intersection(row.get("path") for row in validation), f"{setting['id']}: split paths overlap")
    require(not {row.get("sha256") for row in train}.intersection(row.get("sha256") for row in validation), f"{setting['id']}: split hashes overlap")
    root = recipe_root(manifest, setting["id"])
    composite = read_json(root / "manifest.json")
    require(composite.get("recipe_sha256") == setting["future_recipe_sha256"], f"{setting['id']}: recipe hash differs")
    from treewm.data.future_recipe import FutureRecipe, validate_recipe_manifest
    validate_recipe_manifest(
        root,
        composite,
        expected_source_manifest_sha256=contract["data_manifest_sha256"],
        expected_normalizer_sha256=contract["normalizer_sha256"],
        expected_calibration_sha256=contract["calibration_sha256"],
        expected_thresholds=contract["chosen_thresholds"],
        expected_train_manifest_sha256=contract["train_manifest_sha256"],
        expected_validation_manifest_sha256=contract["validation_manifest_sha256"],
        expected_code_sha256=legacy["recipe_code_sha256"],
        expected_runtime_sha256=legacy["recipe_runtime_sha256"],
        verify_file_hash=verify_files,
    )
    audits: dict[str, Any] = {}
    for split, key, count in (("train", "train_manifest", setting["published_union_train_anchors"]), ("validation", "validation_manifest", setting["published_union_validation_anchors"])):
        recipe = FutureRecipe(root / Path(composite[key]).parent)
        require(len(recipe.anchors) == count, f"{setting['id']}: {split} union count differs")
        selected = recipe_audit_anchors(recipe.anchors)
        require(recipe.contains_all(selected), f"{setting['id']}: {split} audit failed")
        audits[split] = {"recipe_anchor_count": len(recipe.anchors), "audit_anchor_sha256": stable_hash(selected)}
    result = dict(contract)
    result["recipe_coverage_audit"] = audits
    return result


def source_contract(repo_root: str | Path = REPOSITORY_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from treewm.utils.provenance import runtime_fingerprint, trainer_code_fingerprint
    source = trainer_code_fingerprint(root)
    runtime = runtime_fingerprint()
    return {"source_sha256": source["manifest_sha256"], "source_files": source["files"], "runtime_sha256": runtime["sha256"], "runtime": runtime}


def snapshot_identity_sha256(source: Mapping[str, Any], package_protocol_sha256: str) -> str:
    return stable_hash({"source_sha256": source["source_sha256"], "runtime_sha256": source["runtime_sha256"], "package_protocol_sha256": package_protocol_sha256})


def assert_snapshot_files_read_only(root: str | Path) -> int:
    files = [path for path in Path(root).resolve().rglob("*") if path.is_file()]
    require(files and all(not path.is_symlink() for path in files), "snapshot files missing/symlinked")
    require(all(path.stat().st_mode & 0o222 == 0 for path in files), "snapshot contains writable source")
    return len(files)


def verify_source_snapshot(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    marker = read_json(root.parent / "SNAPSHOT.json")
    require(marker.get("schema_version") == 1 and marker.get("status") == "sealed_read_only", "snapshot is not sealed")
    source = source_contract(root)
    protocol = verify_protocol_lock(root / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2")
    identity = snapshot_identity_sha256(source, protocol)
    require(marker.get("trainer_source_sha256") == source["source_sha256"], "snapshot source differs")
    require(marker.get("runtime_sha256") == source["runtime_sha256"], "snapshot runtime differs")
    require(marker.get("package_protocol_sha256") == protocol, "snapshot protocol differs")
    require(marker.get("snapshot_identity_sha256") == identity and root.parent.name == identity, "snapshot identity differs")
    assert_snapshot_files_read_only(root)
    return {"source_sha256": source["source_sha256"], "runtime_sha256": source["runtime_sha256"], "package_protocol_sha256": protocol, "snapshot_identity_sha256": identity}


def _override(name: str, value: object) -> str:
    if isinstance(value, bool):
        rendered = str(value).lower()
    elif isinstance(value, (list, tuple)):
        rendered = "[" + ",".join(str(item).lower() if isinstance(item, bool) else str(item) for item in value) + "]"
    else:
        rendered = str(value)
    return f"{name}={rendered}"


def actual_evaluation_bank(manifest: Mapping[str, Any]) -> dict[str, Any]:
    scientific = manifest["scientific_contract"]
    base = int(scientific["evaluation_seed"])
    episodes = int(scientific["periodic_episodes_per_task"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "policy": "prospective_monitor_25000_fixed_cfg_eval_seed_fallback",
        "task_ids": list(TASK_IDS),
        "episodes_per_task": episodes,
        "seeds": [[base + 1000 * task_index + episode for episode in range(episodes)] for task_index, _ in enumerate(TASK_IDS)],
    }
    payload["sha256"] = stable_hash(payload)
    return payload


def scientific_overrides(
    manifest: Mapping[str, Any],
    run: RunSpec,
    contract: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> list[str]:
    setting = setting_for(manifest, run)
    method = manifest["method"]
    scientific = manifest["scientific_contract"]
    future = scientific["future_config"]
    recipe = binding["selected_recipe"]
    chosen = contract["chosen_thresholds"]
    return [
        _override("env", run.env_config),
        _override("experiment", method["experiment_config"]),
        _override("arm", method["arm"]),
        _override("objective_version", method["objective_version"]),
        _override("seed", run.seed),
        _override("train.steps", scientific["optimizer_updates"]),
        _override("train.scheduler_total_steps", scientific["scheduler_total_steps"]),
        _override("train.log_every", scientific["training_log_every_updates"]),
        _override("train.ckpt_every", scientific["checkpoint_every_updates"]),
        _override("train.val_every", scientific["validation_every_updates"]),
        _override("train.diag_every", scientific["diagnostics_every_updates"]),
        _override("train.eval_every", scientific["periodic_evaluation_every_updates"]),
        _override("train.validation_sample_seed", scientific["validation_sample_seed"]),
        _override("train.max_train_anchors", setting["published_union_train_anchors"]),
        _override("train.max_val_anchors", setting["published_union_validation_anchors"]),
        _override("train.num_workers", scientific["data_loader_workers"]),
        _override("train.lr", recipe["world_lr"]),
        _override("train.weight_decay", scientific["world_weight_decay"]),
        _override("train.gradient_checkpointing", scientific["gradient_checkpointing"]),
        _override("train.separate_gain_grad_clip", True),
        _override("train.separate_branch_transformer_grad_clip", recipe["separate_branch_transformer_grad_clip"]),
        _override("train.world_grad_clip", 1.0),
        _override("train.gain_grad_clip", 1.0),
        _override("train.branch_transformer_grad_clip", recipe["branch_transformer_grad_clip"]),
        _override("train.gain_loss_every", scientific["gain_loss_every"]),
        _override("train.gain_lr", scientific["gain_lr"]),
        _override("train.gain_weight_decay", scientific["gain_weight_decay"]),
        _override("train.gain_training_scorers", scientific["gain_training_scorers"]),
        _override("model.dropout", scientific["model_dropout"]),
        _override("model.max_depth", scientific["model_max_depth"]),
        _override("tree.max_depth", scientific["tree_max_depth"]),
        _override("tree.node_budget", method["node_budget"]),
        _override("tree.keep_threshold", scientific["keep_threshold"]),
        _override("tree.scorer", method["scorer"]),
        _override("model.branch_factor", method["branch_factor"]),
        _override("planner.decoded_metric", scientific["planner_decoded_metric"]),
        _override("planner.execute_mode", scientific["planner_execute_mode"]),
        _override("planner.execute_steps", scientific["planner_execute_steps"]),
        _override("planner.max_env_steps", setting["max_episode_steps"]),
        _override("planner.require_first_edge_improvement", method["require_first_edge_improvement"]),
        _override("planner.min_first_edge_improvement", scientific["min_first_edge_improvement"]),
        *[_override(f"future_sets.{key}", value) for key, value in future.items()],
        _override("future_sets.relative_endpoints", setting["relative_endpoints"]),
        _override("future_sets.retrieval_radius", chosen["retrieval_radius"]),
        _override("future_sets.displacement_threshold", chosen["displacement_threshold"]),
        _override("future_sets.cluster_threshold", chosen["cluster_threshold"]),
        _override("+env.task_metric_dims", setting["task_metric_dims"]),
        _override("losses.keep_balance", scientific["keep_balance"]),
        _override("losses.enabled.multistep", scientific["multistep_enabled"]),
        _override("losses.weights.multistep", scientific["multistep_weight"]),
        _override("losses.scheduled_sampling_p", scientific["scheduled_sampling_p"]),
        _override("losses.scheduled_sampling_warmup", scientific["scheduled_sampling_warmup"]),
        _override("losses.scheduled_sampling_granularity", scientific["scheduled_sampling_granularity"]),
        _override("losses.multistep_transition_mode", recipe["transition_mode"]),
        _override("losses.grounded_select_action_weight", recipe["grounded_select_action_weight"]),
        _override("losses.grounded_select_endpoint_weight", recipe["grounded_select_endpoint_weight"]),
        _override("losses.grounded_select_horizon_weight", recipe["grounded_select_horizon_weight"]),
        _override("losses.grounded_loss_latent_weight", recipe["grounded_loss_latent_weight"]),
        _override("losses.grounded_loss_action_weight", recipe["grounded_loss_action_weight"]),
        _override("losses.grounded_loss_horizon_weight", recipe["grounded_loss_horizon_weight"]),
        _override("losses.grounded_loss_endpoint_weight", recipe["grounded_loss_endpoint_weight"]),
        _override("losses.grounded_detach_self_fed_parent", scientific["grounded_detach_self_fed_parent"]),
        _override("losses.multistep_depth_weights", scientific["multistep_depth_weights"]),
        _override("losses.enabled.latent_gauge", recipe["latent_gauge_enabled"]),
        _override("losses.weights.latent_gauge", recipe["latent_gauge_weight"]),
        _override("losses.latent_gauge_epsilon", scientific["latent_gauge_epsilon"]),
        _override("losses.latent_gauge_min_reference_scale", scientific["latent_gauge_min_reference_scale"]),
        _override("eval.task_split", scientific["task_split"]),
        _override("eval.episodes_per_task", scientific["periodic_episodes_per_task"]),
        _override("eval.final_episodes_per_task", scientific["final_episodes_per_task"]),
        _override("eval.seed", scientific["evaluation_seed"]),
        _override("+campaign_input_contract_sha256", contract["contract_sha256"]),
        _override("+campaign_calibration_sha256", contract["calibration_sha256"]),
        _override("+campaign_future_recipe_sha256", contract["future_recipe_sha256"]),
        _override("+campaign_compatible_recipe_code_sha256", manifest["compatible_v2_recipe_input"]["recipe_code_sha256"]),
        _override("+campaign_factorial_arm", f"exp20-{binding['selected_arm']}-all-ten"),
        _override("+campaign_prerequisite_binding_sha256", binding["binding_sha256"]),
        _override("+campaign_exp20_binding_sha256", binding["binding_sha256"]),
        _override("+campaign_selected_recipe_sha256", binding["selected_recipe_sha256"]),
    ]


def trainer_command(
    manifest: Mapping[str, Any],
    run: RunSpec,
    *,
    repo_root: str | Path = REPOSITORY_ROOT,
    verify_recipe_files: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    package = root / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2"
    protocol = verify_protocol_lock(package)
    binding = load_exp20_binding(manifest, package / "exp20_binding.json", verify_external_files=True)
    contract = load_compatible_input(manifest, run, verify_files=verify_recipe_files)
    source = source_contract(root)
    overrides = scientific_overrides(manifest, run, contract, binding)
    config_sha = stable_hash({"schema_version": 1, "overrides": overrides})
    scientific = manifest["scientific_contract"]
    from treewm.evaluation.rollout import build_evaluation_seed_tables
    seed_tables = build_evaluation_seed_tables(
        scientific["evaluation_seed_protocol_sha256"],
        run.seed,
        TASK_IDS,
        scientific["periodic_episodes_per_task"],
        scientific["final_episodes_per_task"],
    )
    actual_bank = actual_evaluation_bank(manifest)
    run_protocol = stable_hash({
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "package_protocol_sha256": protocol,
        "source_sha256": source["source_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "config_sha256": config_sha,
        "exp20_binding_sha256": binding["binding_sha256"],
        "selected_recipe_sha256": binding["selected_recipe_sha256"],
        "input_contract_sha256": contract["contract_sha256"],
        "data_manifest_sha256": contract["data_manifest_sha256"],
        "normalizer_sha256": contract["normalizer_sha256"],
        "train_manifest_sha256": contract["train_manifest_sha256"],
        "validation_manifest_sha256": contract["validation_manifest_sha256"],
        "calibration_sha256": contract["calibration_sha256"],
        "future_recipe_sha256": contract["future_recipe_sha256"],
        "actual_evaluation_bank_sha256": actual_bank["sha256"],
    })
    output = run_directory(manifest, run)
    argv = [
        manifest["paths"]["python"],
        str(package / "train_entry.py"),
        *overrides,
        _override("run_root", manifest["paths"]["run_root"]),
        _override("run_name", run.run_name),
        _override("resume", "auto"),
        _override("+campaign_source_sha256", source["source_sha256"]),
        _override("+campaign_protocol_sha256", protocol),
        _override("+campaign_config_sha256", config_sha),
        _override("hydra.run.dir", output / "hydra"),
        _override("hydra.job.chdir", False),
    ]
    environment = {
        "TREEWM_PROTOCOL_SHA256": run_protocol,
        "TREEWM_CODE_SHA256": source["source_sha256"],
        "TREEWM_ACTIVE_SOURCE_SHA256": source["source_sha256"],
        "TREEWM_RUNTIME_SHA256": source["runtime_sha256"],
        "TREEWM_RECIPE_CODE_SHA256": manifest["compatible_v2_recipe_input"]["recipe_code_sha256"],
        "TREEWM_RECIPE_RUNTIME_SHA256": manifest["compatible_v2_recipe_input"]["recipe_runtime_sha256"],
        "TREEWM_CONFIG_SHA256": config_sha,
        "TREEWM_DATA_SHA256": contract["data_manifest_sha256"],
        "TREEWM_CALIBRATION_SHA256": contract["calibration_sha256"],
        "TREEWM_FUTURE_RECIPE_SHA256": contract["future_recipe_sha256"],
        "TREEWM_DATA_CONTRACT_SHA256": contract["contract_sha256"],
        "TREEWM_PREREQUISITE_BINDING_SHA256": binding["binding_sha256"],
        "TREEWM_SELECTED_RECIPE_SHA256": binding["selected_recipe_sha256"],
        "TREEWM_DATA_ROOT": manifest["paths"]["data_root"],
        "TREEWM_CACHE": manifest["paths"]["raw_cache_root"],
        "TREEWM_FUTURE_RECIPE_ROOT": str(recipe_root(manifest, run.setting_id)),
        "TREEWM_EVALUATION_SEED_PROTOCOL_SHA256": scientific["evaluation_seed_protocol_sha256"],
        "TREEWM_EXPECTED_FINAL_SEED_TABLE_SHA256": seed_tables["final"]["sha256"],
        "TREEWM_RUN_NAME": run.run_name,
        "WANDB_PROJECT": manifest["logging"]["wandb_project"],
        "WANDB_RUN_GROUP": manifest["logging"]["wandb_group"],
        "WANDB_RUN_ID": run.wandb_id,
        "WANDB_MODE": manifest["logging"]["wandb_mode"],
        "OMP_NUM_THREADS": str(scientific["loader_thread_limit"]),
        "MKL_NUM_THREADS": str(scientific["loader_thread_limit"]),
        "OPENBLAS_NUM_THREADS": str(scientific["loader_thread_limit"]),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    metric_boundary_required_tags = list(manifest["stage_acceptance"]["training_exact_target_tags"])
    if binding["selected_arm"] == "GS":
        metric_boundary_required_tags.extend(SEPARATE_CLIP_TAGS)
    launch: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "classification": manifest["classification"],
        "formal_validation": False,
        "run": {**asdict(run), "selected_arm": binding["selected_arm"], "run_directory": str(output)},
        "metric_boundary_required_tags": metric_boundary_required_tags,
        "hashes": {
            "manifest_sha256": manifest_sha256(manifest),
            "source_sha256": source["source_sha256"],
            "runtime_sha256": source["runtime_sha256"],
            "package_protocol_sha256": protocol,
            "config_sha256": config_sha,
            "run_protocol_sha256": run_protocol,
            "exp20_binding_sha256": binding["binding_sha256"],
            "selected_recipe_sha256": binding["selected_recipe_sha256"],
            "input_contract_sha256": contract["contract_sha256"],
            "data_manifest_sha256": contract["data_manifest_sha256"],
            "normalizer_sha256": contract["normalizer_sha256"],
            "train_manifest_sha256": contract["train_manifest_sha256"],
            "validation_manifest_sha256": contract["validation_manifest_sha256"],
            "calibration_sha256": contract["calibration_sha256"],
            "future_recipe_sha256": contract["future_recipe_sha256"],
            "recipe_code_sha256": manifest["compatible_v2_recipe_input"]["recipe_code_sha256"],
            "recipe_runtime_sha256": manifest["compatible_v2_recipe_input"]["recipe_runtime_sha256"],
            "evaluation_seed_tables_sha256": seed_tables["sha256"],
            "final_seed_table_sha256": seed_tables["final"]["sha256"],
            "actual_evaluation_bank_sha256": actual_bank["sha256"],
        },
        "argv": argv,
        "environment": environment,
    }
    launch["launch_sha256"] = stable_hash(launch)
    return launch


def verify_all(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path = REPOSITORY_ROOT,
    verify_files: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    protocol = verify_protocol_lock(root / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2")
    binding = load_exp20_binding(manifest, root / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2/exp20_binding.json")
    audits = {setting["id"]: load_compatible_input(manifest, setting, verify_files=verify_files)["recipe_coverage_audit"] for setting in manifest["settings"]}
    launches = [trainer_command(manifest, run, repo_root=root) for run in expand_runs(manifest)]
    require(len({launch["launch_sha256"] for launch in launches}) == RUNS, "launch identities collide")
    return {
        "schema_version": 1,
        "status": "verified_bounded_all_ten_gauge_bridge",
        "formal_validation": False,
        "campaign_id": manifest["campaign_id"],
        "runs": len(launches),
        "selected_arm": binding["selected_arm"],
        "selected_recipe_sha256": binding["selected_recipe_sha256"],
        "binding_sha256": binding["binding_sha256"],
        "package_protocol_sha256": protocol,
        "source_sha256": launches[0]["hashes"]["source_sha256"],
        "runtime_sha256": launches[0]["hashes"]["runtime_sha256"],
        "recipe_files_verified": bool(verify_files),
        "recipe_coverage_audits": audits,
        "actual_evaluation_bank": actual_evaluation_bank(manifest),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("protocol-hash", "verify", "snapshot", "runs", "launch"))
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.command == "protocol-hash":
        print(protocol_sha256(args.repo_root / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2"))
    elif args.command == "snapshot":
        print(json.dumps(verify_source_snapshot(args.repo_root), sort_keys=True, indent=2))
    elif args.command == "runs":
        print(json.dumps([asdict(run) for run in expand_runs(manifest)], sort_keys=True, indent=2))
    elif args.command == "launch":
        print(json.dumps(trainer_command(manifest, run_at(manifest, args.index), repo_root=args.repo_root), sort_keys=True, indent=2))
    else:
        print(json.dumps(verify_all(manifest, repo_root=args.repo_root, verify_files=args.verify_files), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"gauge all-ten bridge contract error: {exc}", file=sys.stderr)
        raise SystemExit(2)
