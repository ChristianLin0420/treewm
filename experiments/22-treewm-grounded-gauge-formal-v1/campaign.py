#!/usr/bin/env python3
"""Sealed contracts and deterministic mappings for fresh gauge formal Exp22."""

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
PROTOCOL_LOCK_PATH = CAMPAIGN_DIR / "protocol.sha256"
SEED_TABLE_PATH = CAMPAIGN_DIR / "eval_seed_table.json"
PREREQUISITE_BINDINGS_PATH = CAMPAIGN_DIR / "prerequisite_bindings.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TRAINING_RUNS = 40
FINAL_EVAL_TASKS = 200
PINNED_FORMAL_PYTHON = "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python"
SEEDS = (220, 221, 222, 223)
TASK_IDS = (1, 2, 3, 4, 5)
STAGE_TARGETS = (2_000, 25_000, 100_000, 1_000_000)
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
PROTOCOL_FILES = (
    "manifest.json",
    "campaign.py",
    "raw_exp20_recompute.py",
    "worker.py",
    "stage_gate.py",
    "submit.py",
    "final_eval.py",
    "aggregate.py",
    "train_entry.py",
    "train.slurm",
    "final_eval.slurm",
    "gate.slurm",
    "aggregate.slurm",
    "eval_seed_table.json",
    "prerequisite_bindings.json",
    "bind_prerequisites.py",
    "README.md",
    "tests/conftest.py",
    "tests/test_formal_campaign.py",
    "tests/test_stage_gate.py",
    "tests/test_lifecycle_and_eval.py",
)
EVALUATION_SOURCE_FILES = (
    "scripts/__init__.py",
    "scripts/eval.py",
)


class ContractError(RuntimeError):
    """A scientific, provenance, lifecycle, or evaluation contract is not exact."""


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


@dataclass(frozen=True)
class EvalSpec:
    index: int
    training_index: int
    task_id: int
    run: RunSpec


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


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
    try:
        descriptor = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = read_json(path)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the immutable Exp22 design without consulting outcome artifacts."""
    require(manifest.get("schema_version") == 1, "manifest schema drifted")
    require(manifest.get("campaign_id") == "treewm-grounded-gauge-formal-v1", "campaign ID drifted")
    require(manifest.get("classification") == "fresh_formal_validation", "classification drifted")
    require(manifest.get("expected_training_runs") == TRAINING_RUNS, "training count drifted")
    require(manifest.get("expected_final_eval_tasks") == FINAL_EVAL_TASKS, "eval count drifted")

    authority = manifest.get("promotion_authority") or {}
    require(authority.get("pilot_gate") == "accepted_exp20_gauge_pilot_v2_raw_recomputed", "Exp20 authority drifted")
    require(authority.get("recipe_gate") == "accepted_exp21_all_ten_bridge_v2_raw_recomputed", "Exp21 authority drifted")
    require(authority.get("bindings_file") == "prerequisite_bindings.json", "binding filename drifted")
    require(authority.get("old_checkpoints_allowed") is False, "old checkpoints became allowed")
    require(authority.get("no_outcome_based_recipe_selection_within_formal") is True, "formal recipe policy drifted")
    forbidden = tuple(authority.get("forbidden_ancestry_tokens") or ())
    require(forbidden and forbidden == tuple((manifest.get("prerequisites") or {}).get("forbidden_ancestry_tokens") or ()), "forbidden ancestry set drifted")
    require(all(any(token in old for old in forbidden) for token in ("exp14", "exp15", "exp16", "exp17", "exp18")), "old experiment ancestry is not fully forbidden")

    method = manifest.get("method") or {}
    require(method.get("arm") == "treewm" and method.get("model_class") == "TreeWM", "method identity drifted")
    require(method.get("experiment_config") == "treewm_v2_grounded_gauge_formal_v1", "config identity drifted")
    require(method.get("objective_version") == "treewm_v2_grounded_gauge_formal_v1", "objective identity drifted")
    require(method.get("final_eval_rails") == ["learned", "bfs"], "final rails drifted")
    grounded = method.get("grounded_multistep") or {}
    require(grounded.get("enabled") is True and grounded.get("weight") == 1, "grounded method disabled")
    require(grounded.get("depth_weights") == [1, 1, 1], "depth weights drifted")
    require(grounded.get("scheduled_sampling_granularity") == "sequence", "sampling granularity drifted")
    require(grounded.get("transition_mode") == "grounded_execution_v2", "transition mode drifted")
    require(grounded.get("selector_weights") == {"action": 1, "endpoint": 1, "horizon": 0.25}, "grounded selector weights drifted")
    require(grounded.get("keep_balance") is True and grounded.get("detach_self_fed_parent") is True, "grounded safeguards drifted")
    require(grounded.get("selected_recipe_source") == "exp21_raw_recomputed_acceptance", "recipe source drifted")
    gauge = method.get("latent_gauge") or {}
    require(gauge == {
        "enabled": True,
        "weight": 1,
        "epsilon": 1e-8,
        "min_reference_scale": 1e-4,
        "reference_update": 0,
        "reference_sealed": 1,
        "selected_clipping_mode_source": "exp21_raw_recomputed_acceptance",
    }, "latent-gauge contract drifted")

    prerequisites = manifest.get("prerequisites") or {}
    exp20 = prerequisites.get("exp20") or {}
    exp21 = prerequisites.get("exp21") or {}
    require(exp20.get("campaign_id") == "treewm-grounded-gauge-pilot-v2-launch2", "Exp20 identity drifted")
    require(exp21.get("campaign_id") == "treewm-grounded-gauge-all-ten-bridge-v2", "Exp21 identity drifted")
    require(exp20.get("allowed_selected_arms") == ["G", "GS"] and exp20.get("selection_precedence") == ["G", "GS"], "Exp20 arm contract drifted")
    require(exp21.get("allowed_selected_arms") == ["G", "GS"], "Exp21 arm contract drifted")
    require(exp20.get("required_stage_5000_status") == "accepted_for_selected_continuation", "Exp20 5k status drifted")
    require(exp20.get("required_acceptance_status") == "accepted_for_fresh_formal_campaign_design", "Exp20 25k status drifted")
    require(exp21.get("required_status") == "accepted_for_later_1m_formal_campaign_design", "Exp21 status drifted")
    for dependency, keys in ((exp20, ("manifest_sha256", "package_protocol_sha256", "source_sha256", "runtime_sha256", "actual_evaluation_bank_sha256")), (exp21, ("manifest_sha256", "preacceptance_package_protocol_sha256", "source_sha256", "runtime_sha256", "actual_evaluation_bank_sha256"))):
        for key in keys:
            require(SHA256.fullmatch(str(dependency.get(key, ""))) is not None, f"prerequisite {key} is not pinned")
    for key in ("stage_5000_gate_path", "acceptance_path"):
        require(Path(str(exp20.get(key, ""))).is_absolute(), f"Exp20 {key} is not absolute")
    for key in ("exp20_binding_path", "acceptance_path"):
        require(Path(str(exp21.get(key, ""))).is_absolute(), f"Exp21 {key} is not absolute")
    recipes = prerequisites.get("allowed_selected_recipes") or {}
    require(set(recipes) == {"G", "GS"}, "allowed recipe set drifted")
    common_recipe = {
        "world_lr": 3e-5,
        "transition_mode": "grounded_execution_v2",
        "grounded_select_action_weight": 1,
        "grounded_select_endpoint_weight": 1,
        "grounded_select_horizon_weight": 0.25,
        "grounded_loss_latent_weight": 0.25,
        "grounded_loss_action_weight": 0.5,
        "grounded_loss_horizon_weight": 0.25,
        "grounded_loss_endpoint_weight": 0.5,
        "latent_gauge_enabled": True,
        "latent_gauge_weight": 1,
        "branch_transformer_grad_clip": 1,
    }
    for arm in ("G", "GS"):
        recipe = recipes[arm]
        expected_recipe = {
            "id": arm,
            "label": (
                "gauge-with-shared-world-clipping"
                if arm == "G"
                else "gauge-with-separate-branch-transformer-clipping"
            ),
            **common_recipe,
            "separate_branch_transformer_grad_clip": arm == "GS",
        }
        require(recipe == expected_recipe, f"{arm}: full selected recipe drifted")

    design = manifest.get("design") or {}
    require(design.get("seeds") == list(SEEDS), "fresh training seed bank drifted")
    require(not set(SEEDS).intersection(set(range(0, 4)) | set(range(100, 112))), "training seed bank overlaps method selection")
    require(design.get("task_ids") == list(TASK_IDS), "task IDs drifted")
    scientific = manifest.get("scientific_contract") or {}
    exact_scientific = {
        "optimizer_updates": 1_000_000,
        "scheduler_total_steps": 1_000_000,
        "checkpoint_every_updates": 1_000,
        "training_log_every_updates": 50,
        "validation_every_updates": 2_000,
        "periodic_evaluation_every_updates": 25_000,
        "periodic_episodes_per_task": 1,
        "final_episodes_per_task": 50,
        "planner_decoded_metric": "domain_raw",
        "planner_execute_steps": 4,
        "latent_gauge_weight": 1,
        "branch_transformer_grad_clip": 1,
    }
    for key, value in exact_scientific.items():
        require(scientific.get(key) == value, f"scientific contract {key} drifted")
    require(scientific.get("future_config", {}).get("recipe_anchor_policy") == "published_union", "recipe anchor policy drifted")

    lifecycle = manifest.get("lifecycle") or {}
    require(lifecycle.get("stage_targets") == list(STAGE_TARGETS), "stage targets drifted")
    require(lifecycle.get("resume_policy") == "auto only inside this fresh Exp22 campaign run directory; no external or old checkpoint is accepted", "resume policy drifted")
    require(
        lifecycle.get("post_update_cadence_policy")
        == "Every committed optimizer update completes its ordered logging, diagnostics, validation, periodic checkpoint/evaluation, and visualization cadence before a normal graceful-stop checkpoint; an interrupted evaluation or visualization is durable only with explicit replay intent that completes before the next optimizer update.",
        "post-update cadence policy drifted",
    )
    gate = manifest.get("stage_acceptance") or {}
    require(gate.get("scientific_gate_stage") == 25_000 and gate.get("outcome_sanity_stage") == 100_000, "stage gate placement drifted")
    require(gate.get("post_100k_policy") == "integrity_and_numerical_health_only", "post-100k policy drifted")
    require(gate.get("clipping_fraction_policy") == "per_tag_maximum", "clipping policy drifted")
    require(gate.get("min_scale_ratio") == 0.8 and gate.get("reference_update") == 0 and gate.get("reference_sealed") == 1, "gauge gate drifted")

    final = manifest.get("final_evaluation") or {}
    require(final.get("array") == "0-199%40" and final.get("aggregate_requires") == FINAL_EVAL_TASKS, "final-eval matrix drifted")
    require(final.get("episodes_per_task_per_rail") == 50 and final.get("rails") == ["learned", "bfs"], "paired final protocol drifted")
    require(final.get("adaptive_selection") is False and final.get("heldout_disjointness_required") is True, "heldout policy drifted")
    promotion = final.get("promotion_criterion") or {}
    require(promotion.get("primary_inference_unit") == "training_seed" and promotion.get("training_seed_replicates") == 4, "paired inference unit drifted")
    require(promotion.get("t_critical_975_df3") == 3.182446, "paired t critical drifted")

    settings = manifest.get("settings") or []
    require(len(settings) == len(SETTING_IDS) and [s.get("id") for s in settings] == list(SETTING_IDS), "setting order/coverage drifted")
    for setting in settings:
        require(setting.get("published_union_train_anchors") == EXPECTED_UNION_COUNTS[setting["id"]][0], f"{setting['id']}: train union drifted")
        require(setting.get("published_union_validation_anchors") == EXPECTED_UNION_COUNTS[setting["id"]][1], f"{setting['id']}: validation union drifted")
        for key in ("input_contract_sha256", "calibration_sha256", "future_recipe_sha256", "evaluation_seed_protocol_sha256"):
            require(SHA256.fullmatch(str(setting.get(key, ""))) is not None, f"{setting['id']}: malformed {key}")
    paths = manifest.get("paths") or {}
    require(paths.get("python") == PINNED_FORMAL_PYTHON, "Python runtime drifted")
    require(str(paths.get("run_root", "")).endswith("/outputs/treewm-grounded-gauge-formal-v1"), "run namespace drifted")
    require(str(paths.get("final_eval_root", "")).endswith("/outputs/treewm-grounded-gauge-formal-v1-final-eval"), "eval namespace drifted")
    logging = manifest.get("logging") or {}
    require(logging.get("wandb_project") == logging.get("wandb_group") == "treewm-grounded-gauge-formal-v1", "W&B namespace drifted")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    return stable_hash(manifest)


def protocol_sha256(campaign_dir: str | Path = CAMPAIGN_DIR) -> str:
    root = Path(campaign_dir).resolve()
    repository = root.parents[1]
    require(len(PROTOCOL_FILES) == len(set(PROTOCOL_FILES)), "protocol file inventory contains duplicates")
    files: dict[str, str] = {}
    for relative in PROTOCOL_FILES:
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"protocol file missing/symlinked: {path}")
        files[f"experiments/22-treewm-grounded-gauge-formal-v1/{relative}"] = file_sha256(path)
    config = repository / "configs/experiment/treewm_v2_grounded_gauge_formal_v1.yaml"
    require(config.is_file() and not config.is_symlink(), "Exp22 config missing/symlinked")
    files["configs/experiment/treewm_v2_grounded_gauge_formal_v1.yaml"] = file_sha256(config)
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


def load_seed_table(
    manifest: Mapping[str, Any], path: str | Path = SEED_TABLE_PATH
) -> dict[str, Any]:
    payload = read_json(path)
    claimed = payload.get("sha256")
    body = dict(payload)
    body.pop("sha256", None)
    require(claimed == stable_hash(body), "locked evaluation seed-table hash differs")
    require(payload.get("schema_version") == 1, "seed-table schema differs")
    require(payload.get("campaign_id") == manifest["campaign_id"], "seed-table campaign differs")
    require(payload.get("task_ids") == list(TASK_IDS), "seed-table tasks differ")
    require(payload.get("episodes_per_task") == 50, "seed-table episode count differs")
    tables = payload.get("settings") or {}
    require(set(tables) == set(SETTING_IDS) and len(tables) == len(SETTING_IDS), "seed-table setting coverage differs")
    from treewm.evaluation.rollout import validate_evaluation_seed_table

    all_final: set[int] = set()
    all_monitor: set[int] = set()
    for setting in manifest["settings"]:
        table = tables.get(setting["id"]) or {}
        validate_evaluation_seed_table(
            table,
            split="final",
            task_ids=TASK_IDS,
            episodes_per_task=50,
        )
        require(
            table.get("protocol_sha256") == setting["evaluation_seed_protocol_sha256"],
            f"{setting['id']}: evaluation seed protocol differs",
        )
        from treewm.evaluation.rollout import build_evaluation_seed_tables
        generated = build_evaluation_seed_tables(
            setting["evaluation_seed_protocol_sha256"],
            SEEDS[0],
            TASK_IDS,
            manifest["scientific_contract"]["periodic_episodes_per_task"],
            manifest["scientific_contract"]["final_episodes_per_task"],
        )
        require(generated["final"] == table, f"{setting['id']}: final seed bank is not generator-exact")
        require(all(build_evaluation_seed_tables(setting["evaluation_seed_protocol_sha256"], seed, TASK_IDS, 1, 50)["final"] == table for seed in SEEDS), f"{setting['id']}: final bank is not common across model seeds")
        final_values = {int(seed) for row in table["seeds"] for seed in row}
        monitor_values = {int(seed) for row in generated["monitor"]["seeds"] for seed in row}
        require(not final_values.intersection(monitor_values), f"{setting['id']}: monitor/final seeds overlap")
        require(not all_final.intersection(final_values), f"{setting['id']}: final seeds overlap another setting")
        require(not all_monitor.intersection(monitor_values), f"{setting['id']}: monitor seeds overlap another setting")
        require(not all_final.intersection(monitor_values) and not all_monitor.intersection(final_values), f"{setting['id']}: cross-setting monitor/final overlap")
        all_final.update(final_values)
        all_monitor.update(monitor_values)
    require(len(all_final) == len(SETTING_IDS) * len(TASK_IDS) * 50, "heldout bank cardinality differs")
    return payload


def _contains_forbidden(value: object, tokens: Sequence[str]) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_forbidden(key, tokens) or _contains_forbidden(item, tokens) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden(item, tokens) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(token.lower() in lowered for token in tokens)
    return False


def load_prerequisite_bindings(
    manifest: Mapping[str, Any],
    path: str | Path = PREREQUISITE_BINDINGS_PATH,
    *,
    verify_external_files: bool = True,
) -> dict[str, Any]:
    """Load byte-bound raw recomputations of Exp20 and Exp21, never old ancestry."""
    payload = read_json(path)
    require(payload.get("schema_version") == 1, "prerequisite binding schema differs")
    require(payload.get("campaign_id") == manifest["campaign_id"], "prerequisite binding campaign differs")
    require(
        payload.get("status") == "sealed_accepted_exp20_and_exp21_raw_recomputed",
        "formal campaign is blocked: accepted Exp20+Exp21 raw-evidence binding is not sealed",
    )
    require(payload.get("formal_submission_allowed") is True, "prerequisite binding forbids submission")
    require(
        payload.get("selection_policy") == "derive_exact_exp21_recipe_after_independent_exp20_and_exp21_raw_recomputation",
        "formal recipe selection policy differs",
    )
    claimed = payload.get("binding_sha256")
    body = dict(payload)
    body.pop("binding_sha256", None)
    require(SHA256.fullmatch(str(claimed or "")) is not None and claimed == stable_hash(body), "prerequisite binding self-hash differs")
    tokens = tuple(manifest["prerequisites"]["forbidden_ancestry_tokens"])
    require(not _contains_forbidden(payload, tokens), "prerequisite binding contains forbidden Exp14-18 ancestry")

    selected_arm = payload.get("selected_arm")
    require(selected_arm in ("G", "GS"), "selected arm is not G/GS")
    selected_recipe = payload.get("selected_recipe")
    require(isinstance(selected_recipe, dict), "selected recipe is missing")
    require(selected_recipe == manifest["prerequisites"]["allowed_selected_recipes"][selected_arm], "selected recipe differs from sealed arm recipe")
    require(payload.get("selected_recipe_sha256") == stable_hash(selected_recipe), "selected recipe hash differs")

    exp20 = payload.get("exp20") or {}
    exp21 = payload.get("exp21") or {}
    exp20_contract = manifest["prerequisites"]["exp20"]
    exp21_contract = manifest["prerequisites"]["exp21"]
    require(exp20.get("campaign_id") == exp20_contract["campaign_id"], "bound Exp20 identity differs")
    require(exp21.get("campaign_id") == exp21_contract["campaign_id"], "bound Exp21 identity differs")
    require(exp20.get("stage_5000_status") == exp20_contract["required_stage_5000_status"], "bound Exp20 5k status differs")
    require(exp20.get("acceptance_status") == exp20_contract["required_acceptance_status"], "bound Exp20 25k status differs")
    require(exp21.get("acceptance_status") == exp21_contract["required_status"], "bound Exp21 status differs")
    require(exp20.get("selected_arm") == exp21.get("selected_arm") == selected_arm, "cross-gate selected arm differs")
    require(exp20.get("selected_recipe") == exp21.get("selected_recipe") == selected_recipe, "cross-gate selected recipe differs")
    require(exp20.get("selected_recipe_sha256") == exp21.get("selected_recipe_sha256") == payload["selected_recipe_sha256"], "cross-gate recipe hash differs")
    for key in ("manifest_sha256", "package_protocol_sha256", "source_sha256", "runtime_sha256", "actual_evaluation_bank_sha256"):
        require(exp20.get(key) == exp20_contract[key], f"bound Exp20 {key} differs")
    for key in ("manifest_sha256", "source_sha256", "runtime_sha256", "actual_evaluation_bank_sha256"):
        require(exp21.get(key) == exp21_contract[key], f"bound Exp21 {key} differs")
    require(SHA256.fullmatch(str(exp21.get("package_protocol_sha256", ""))) is not None, "bound Exp21 accepted protocol malformed")
    for dependency in (exp20, exp21):
        for key in ("raw_recomputation_sha256", "acceptance_file_sha256"):
            require(SHA256.fullmatch(str(dependency.get(key, ""))) is not None, f"bound prerequisite has malformed {key}")
    require(SHA256.fullmatch(str(exp20.get("stage_5000_gate_file_sha256", ""))) is not None, "Exp20 5k gate hash malformed")
    require(SHA256.fullmatch(str(exp21.get("exp20_binding_file_sha256", ""))) is not None, "Exp21 upstream-binding hash malformed")
    require(len(exp20.get("stage_5000_run_evidence") or {}) == 30, "Exp20 5k evidence is not all 30 runs")
    require(len(exp20.get("stage_25000_run_evidence") or {}) == 10, "Exp20 25k evidence is not all 10 selected runs")
    require(len(exp20.get("stage_25000_skipped_runs") or []) == 10, "Exp20 skipped-arm evidence is not all 10 runs")
    require(len(exp21.get("run_evidence") or {}) == 20, "Exp21 evidence is not all 20 runs")
    for records, expected in ((exp20["stage_5000_run_evidence"], 30), (exp20["stage_25000_run_evidence"], 10), (exp21["run_evidence"], 20)):
        require(len(records) == expected and all(isinstance(record, dict) for record in records.values()), "raw evidence record shape differs")
        for record in records.values():
            for key in ("launch_file_sha256", "event_evidence_sha256", "checkpoint_sha256"):
                require(SHA256.fullmatch(str(record.get(key, ""))) is not None, f"raw evidence {key} malformed")
    bound_files = payload.get("bound_files") or {}
    require(isinstance(bound_files, dict) and len(bound_files) >= 4, "bound prerequisite file inventory is incomplete")
    require(all(Path(str(name)).is_absolute() and SHA256.fullmatch(str(digest)) is not None for name, digest in bound_files.items()), "bound prerequisite file inventory malformed")
    required_paths = {
        str(Path(exp20_contract["stage_5000_gate_path"]).resolve()),
        str(Path(exp20_contract["acceptance_path"]).resolve()),
        str(Path(exp21_contract["exp20_binding_path"]).resolve()),
        str(Path(exp21_contract["acceptance_path"]).resolve()),
    }
    require(required_paths.issubset(set(bound_files)), "bound prerequisite decisions are not all byte-inventoried")
    if verify_external_files:
        for name, digest in bound_files.items():
            artifact = Path(name)
            require(artifact.is_file() and not artifact.is_symlink(), f"bound prerequisite file disappeared/symlinked: {artifact}")
            require(file_sha256(artifact) == digest, f"bound prerequisite bytes changed: {artifact}")
    return payload


def expand_runs(manifest: Mapping[str, Any]) -> list[RunSpec]:
    validate_manifest(manifest)
    runs: list[RunSpec] = []
    for setting_index, setting in enumerate(manifest["settings"]):
        for seed_index, seed in enumerate(SEEDS):
            index = setting_index * len(SEEDS) + seed_index
            name = f"grounded-gauge-formal-{setting['id']}-seed{seed}"
            wandb_id = stable_hash({"campaign_id": manifest["campaign_id"], "setting_id": setting["id"], "seed": seed})[:32]
            runs.append(RunSpec(index, setting_index, seed_index, setting["id"], setting["env_config"], seed, name, wandb_id))
    require(len(runs) == TRAINING_RUNS and len({r.run_name for r in runs}) == TRAINING_RUNS, "run expansion differs")
    return runs


def run_at(manifest: Mapping[str, Any], index: int) -> RunSpec:
    require(0 <= index < TRAINING_RUNS, "training index must be in [0,40)")
    return expand_runs(manifest)[index]


def eval_at(manifest: Mapping[str, Any], index: int) -> EvalSpec:
    require(0 <= index < FINAL_EVAL_TASKS, "final-eval index must be in [0,200)")
    training_index, task_offset = divmod(index, len(TASK_IDS))
    return EvalSpec(index, training_index, TASK_IDS[task_offset], run_at(manifest, training_index))


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
    """Select deterministic audit anchors from the sealed recipe, never a dataset prefix."""
    values = [int(value) for value in anchors]
    require(values and all(a < b for a, b in zip(values, values[1:])), "recipe anchors not ordered/unique")
    count = min(int(sample_count), len(values))
    require(count > 0, "recipe audit sample count must be positive")
    if count == 1:
        return [values[0]]
    positions = [(i * (len(values) - 1)) // (count - 1) for i in range(count)]
    selected = [values[position] for position in positions]
    require(len(selected) == len(set(selected)), "recipe audit selection duplicated anchors")
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
    require(
        contract.get("train_manifest_sha256") != contract.get("validation_manifest_sha256"),
        f"{setting['id']}: train/validation split manifest identities overlap",
    )
    source_files = contract.get("source_files") or []
    train_sources = [row for row in source_files if row.get("split") == "train"]
    validation_sources = [row for row in source_files if row.get("split") in {"val", "validation"}]
    require(train_sources and validation_sources, f"{setting['id']}: source split coverage is incomplete")
    train_paths = {str(row.get("path")) for row in train_sources}
    validation_paths = {str(row.get("path")) for row in validation_sources}
    train_hashes = {str(row.get("sha256")) for row in train_sources}
    validation_hashes = {str(row.get("sha256")) for row in validation_sources}
    require(not train_paths.intersection(validation_paths), f"{setting['id']}: train/validation source paths overlap")
    require(not train_hashes.intersection(validation_hashes), f"{setting['id']}: train/validation source hashes overlap")

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
    for split, manifest_key, expected_count in (
        ("train", "train_manifest", setting["published_union_train_anchors"]),
        ("val", "validation_manifest", setting["published_union_validation_anchors"]),
    ):
        split_root = root / Path(composite[manifest_key]).parent
        split_recipe = FutureRecipe(split_root)
        require(len(split_recipe.anchors) == expected_count, f"{setting['id']}: {split} union count differs")
        selected = recipe_audit_anchors(split_recipe.anchors)
        require(split_recipe.contains_all(selected), f"{setting['id']}: recipe-derived audit coverage failed")
        audits[split] = {
            "source": "sealed_recipe_union",
            "recipe_anchor_count": len(split_recipe.anchors),
            "audit_anchor_count": len(selected),
            "audit_anchor_sha256": stable_hash(selected),
        }
    result = dict(contract)
    result["source_split_audit"] = {
        "train_manifest_sha256": contract["train_manifest_sha256"],
        "validation_manifest_sha256": contract["validation_manifest_sha256"],
        "train_source_file_count": len(train_sources),
        "validation_source_file_count": len(validation_sources),
        "path_overlap_count": 0,
        "sha256_overlap_count": 0,
    }
    result["recipe_coverage_audit"] = audits
    return result


def evaluation_source_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    files: dict[str, str] = {}
    for relative in EVALUATION_SOURCE_FILES:
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"evaluation source missing/symlinked: {path}")
        files[relative] = file_sha256(path)
    return {
        "sha256": stable_hash({"schema_version": 1, "files": files}),
        "files": files,
    }


def source_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from treewm.utils.provenance import runtime_fingerprint, trainer_code_fingerprint

    source = trainer_code_fingerprint(root)
    runtime = runtime_fingerprint()
    evaluation_source = evaluation_source_contract(root)
    return {
        "source_sha256": source["manifest_sha256"],
        "source_files": source["files"],
        "evaluation_source_sha256": evaluation_source["sha256"],
        "evaluation_source_files": evaluation_source["files"],
        "runtime_sha256": runtime["sha256"],
        "runtime": runtime,
    }


def snapshot_identity_sha256(source: Mapping[str, Any], package_protocol_sha256: str) -> str:
    return stable_hash({
        "source_sha256": source["source_sha256"],
        "evaluation_source_sha256": source["evaluation_source_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "package_protocol_sha256": package_protocol_sha256,
    })


def assert_snapshot_files_read_only(root: str | Path) -> int:
    snapshot_root = Path(root).resolve()
    regular_files = [path for path in snapshot_root.rglob("*") if path.is_file()]
    require(regular_files, "snapshot repository contains no source files")
    require(all(not path.is_symlink() for path in regular_files), "snapshot contains symlinked source")
    require(all(path.stat().st_mode & 0o222 == 0 for path in regular_files), "snapshot has writable source files")
    return len(regular_files)


def verify_source_snapshot(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    marker_path = root.parent / "SNAPSHOT.json"
    marker = read_json(marker_path)
    require(marker.get("schema_version") == 1, "snapshot schema differs")
    require(marker.get("status") == "sealed_read_only", "source snapshot is not sealed")
    require(marker.get("repo_subdirectory") == root.name == "repo", "snapshot repo path differs")
    require(marker.get("repo_files_writable") is False, "snapshot permits writable files")
    source = source_contract(root)
    protocol = verify_protocol_lock(root / "experiments" / "22-treewm-grounded-gauge-formal-v1")
    snapshot_identity = snapshot_identity_sha256(source, protocol)
    require(marker.get("trainer_source_sha256") == source["source_sha256"], "snapshot trainer source differs")
    require(marker.get("evaluation_source_sha256") == source["evaluation_source_sha256"], "snapshot evaluation source differs")
    require(marker.get("runtime_sha256") == source["runtime_sha256"], "snapshot runtime differs")
    require(marker.get("package_protocol_sha256") == protocol, "snapshot package protocol differs")
    require(marker.get("snapshot_identity_sha256") == snapshot_identity, "snapshot identity differs")
    require(root.parent.name == snapshot_identity, "snapshot directory identity differs")
    assert_snapshot_files_read_only(root)
    return {
        "marker": str(marker_path),
        "source_sha256": source["source_sha256"],
        "evaluation_source_sha256": source["evaluation_source_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "package_protocol_sha256": protocol,
        "snapshot_identity_sha256": snapshot_identity,
    }


def _override(name: str, value: object) -> str:
    if isinstance(value, bool):
        rendered = str(value).lower()
    elif isinstance(value, (list, tuple)):
        rendered = "[" + ",".join(str(v).lower() if isinstance(v, bool) else str(v) for v in value) + "]"
    else:
        rendered = str(value)
    return f"{name}={rendered}"


def scientific_overrides(
    manifest: Mapping[str, Any],
    run: RunSpec,
    contract: Mapping[str, Any],
    prerequisite_binding: Mapping[str, Any],
) -> list[str]:
    setting = setting_for(manifest, run)
    method = manifest["method"]
    scientific = manifest["scientific_contract"]
    future = scientific["future_config"]
    grounded = method["grounded_multistep"]
    regularization = method["regularization"]
    selected = prerequisite_binding["selected_recipe"]
    chosen = contract["chosen_thresholds"]
    return [
        _override("env", run.env_config),
        _override("experiment", method["experiment_config"]),
        _override("arm", method["arm"]),
        _override("objective_version", method["objective_version"]),
        _override("seed", run.seed),
        _override("train.steps", scientific["optimizer_updates"]),
        _override("train.scheduler_total_steps", scientific["scheduler_total_steps"]),
        _override("train.ckpt_every", scientific["checkpoint_every_updates"]),
        _override("train.log_every", scientific["training_log_every_updates"]),
        _override("train.val_every", scientific["validation_every_updates"]),
        _override("train.diag_every", scientific["validation_every_updates"]),
        _override("train.eval_every", scientific["periodic_evaluation_every_updates"]),
        _override("train.validation_sample_seed", scientific["validation_sample_seed"]),
        _override("train.max_train_anchors", setting["published_union_train_anchors"]),
        _override("train.max_val_anchors", setting["published_union_validation_anchors"]),
        _override("train.num_workers", scientific["data_loader_workers"]),
        _override("train.lr", selected["world_lr"]),
        _override("train.weight_decay", regularization["weight_decay"]),
        _override("train.gradient_checkpointing", scientific["gradient_checkpointing"]),
        _override("train.separate_gain_grad_clip", True),
        _override("train.separate_branch_transformer_grad_clip", selected["separate_branch_transformer_grad_clip"]),
        _override("train.world_grad_clip", 1.0),
        _override("train.gain_grad_clip", 1.0),
        _override("train.branch_transformer_grad_clip", selected["branch_transformer_grad_clip"]),
        _override("train.gain_loss_every", 1),
        _override("train.gain_lr", 3e-4),
        _override("train.gain_weight_decay", 0.0),
        _override("train.gain_training_scorers", ["learned", "novelty_q"]),
        _override("train.viz_every", 25_000),
        _override("train.viz_every_early", 2_000),
        _override("train.viz_early_until", 25_000),
        _override("model.dropout", regularization["dropout"]),
        _override("model.max_depth", scientific["model_max_depth"]),
        _override("tree.max_depth", scientific["tree_max_depth"]),
        _override("tree.node_budget", method["node_budget"]),
        _override("tree.scorer", method["scorer"]),
        _override("model.branch_factor", method["branch_factor"]),
        _override("planner.score_space", scientific["planner_score_space"]),
        _override("planner.decoded_metric", scientific["planner_decoded_metric"]),
        _override("planner.execute_mode", scientific["planner_execute_mode"]),
        _override("planner.execute_steps", scientific["planner_execute_steps"]),
        _override("planner.max_env_steps", setting["max_episode_steps"]),
        _override("planner.require_first_edge_improvement", scientific["require_first_edge_improvement"]),
        _override("planner.min_first_edge_improvement", scientific["min_first_edge_improvement"]),
        *[_override(f"future_sets.{key}", value) for key, value in future.items()],
        _override("future_sets.relative_endpoints", setting["relative_endpoints"]),
        _override("future_sets.retrieval_radius", chosen["retrieval_radius"]),
        _override("future_sets.displacement_threshold", chosen["displacement_threshold"]),
        _override("future_sets.cluster_threshold", chosen["cluster_threshold"]),
        _override("+env.task_metric_dims", setting["task_metric_dims"]),
        _override("losses.enabled.multistep", grounded["enabled"]),
        _override("losses.weights.multistep", grounded["weight"]),
        _override("losses.scheduled_sampling_p", grounded["scheduled_sampling_p"]),
        _override("losses.scheduled_sampling_warmup", grounded["scheduled_sampling_warmup"]),
        _override("losses.scheduled_sampling_granularity", grounded["scheduled_sampling_granularity"]),
        _override("losses.multistep_transition_mode", selected["transition_mode"]),
        _override("losses.grounded_select_action_weight", selected["grounded_select_action_weight"]),
        _override("losses.grounded_select_endpoint_weight", selected["grounded_select_endpoint_weight"]),
        _override("losses.grounded_select_horizon_weight", selected["grounded_select_horizon_weight"]),
        _override("losses.grounded_loss_latent_weight", selected["grounded_loss_latent_weight"]),
        _override("losses.grounded_loss_action_weight", selected["grounded_loss_action_weight"]),
        _override("losses.grounded_loss_horizon_weight", selected["grounded_loss_horizon_weight"]),
        _override("losses.grounded_loss_endpoint_weight", selected["grounded_loss_endpoint_weight"]),
        _override("losses.grounded_detach_self_fed_parent", grounded["detach_self_fed_parent"]),
        _override("losses.keep_balance", grounded["keep_balance"]),
        _override("losses.multistep_depth_weights", grounded["depth_weights"]),
        _override("losses.enabled.latent_gauge", selected["latent_gauge_enabled"]),
        _override("losses.weights.latent_gauge", selected["latent_gauge_weight"]),
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
        _override("+campaign_factorial_arm", f"exp21-{prerequisite_binding['selected_arm']}"),
        _override("+campaign_prerequisite_binding_sha256", prerequisite_binding["binding_sha256"]),
        _override("+campaign_selected_recipe_sha256", prerequisite_binding["selected_recipe_sha256"]),
    ]


def trainer_command(manifest: Mapping[str, Any], run: RunSpec, *, repo_root: str | Path = REPOSITORY_ROOT, verify_recipe_files: bool = False) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    package = root / "experiments" / "22-treewm-grounded-gauge-formal-v1"
    protocol = verify_protocol_lock(package)
    prerequisite_binding = load_prerequisite_bindings(
        manifest,
        package / "prerequisite_bindings.json",
        # The independent binder and submit preflight hash every raw prerequisite
        # byte once.  Runtime jobs consume the protocol-bound local receipt; they
        # must not reread tens of checkpoints and event streams per array cell.
        verify_external_files=False,
    )
    contract = load_compatible_input(manifest, run, verify_files=verify_recipe_files)
    source = source_contract(root)
    overrides = scientific_overrides(manifest, run, contract, prerequisite_binding)
    config_sha = stable_hash({"schema_version": 1, "overrides": overrides})
    setting = setting_for(manifest, run)
    seed_bundle = load_seed_table(manifest, package / "eval_seed_table.json")
    final_seed_table = seed_bundle["settings"][run.setting_id]
    from treewm.evaluation.rollout import build_evaluation_seed_tables

    expected_seed_tables = build_evaluation_seed_tables(
        setting["evaluation_seed_protocol_sha256"],
        run.seed,
        TASK_IDS,
        manifest["scientific_contract"]["periodic_episodes_per_task"],
        manifest["scientific_contract"]["final_episodes_per_task"],
    )
    require(
        expected_seed_tables["final"] == final_seed_table,
        f"{run.setting_id}: locked final seed table differs from trainer generator",
    )
    run_protocol = stable_hash({
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "package_protocol_sha256": protocol,
        "source_sha256": source["source_sha256"],
        "evaluation_source_sha256": source["evaluation_source_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "config_sha256": config_sha,
        "prerequisite_binding_sha256": prerequisite_binding["binding_sha256"],
        "selected_recipe_sha256": prerequisite_binding["selected_recipe_sha256"],
        "selected_arm": prerequisite_binding["selected_arm"],
        "input_contract_sha256": contract["contract_sha256"],
        "future_recipe_sha256": contract["future_recipe_sha256"],
        "evaluation_seed_protocol_sha256": setting["evaluation_seed_protocol_sha256"],
        "final_seed_table_sha256": final_seed_table["sha256"],
        "monitor_seed_table_sha256": expected_seed_tables["monitor"]["sha256"],
        "evaluation_seed_tables_sha256": expected_seed_tables["sha256"],
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
        "TREEWM_EVALUATION_SOURCE_SHA256": source["evaluation_source_sha256"],
        "TREEWM_RUNTIME_SHA256": source["runtime_sha256"],
        "TREEWM_RECIPE_CODE_SHA256": manifest["compatible_v2_recipe_input"]["recipe_code_sha256"],
        "TREEWM_RECIPE_RUNTIME_SHA256": manifest["compatible_v2_recipe_input"]["recipe_runtime_sha256"],
        "TREEWM_CONFIG_SHA256": config_sha,
        "TREEWM_PREREQUISITE_BINDING_SHA256": prerequisite_binding["binding_sha256"],
        "TREEWM_SELECTED_RECIPE_SHA256": prerequisite_binding["selected_recipe_sha256"],
        "TREEWM_DATA_SHA256": contract["data_manifest_sha256"],
        "TREEWM_CALIBRATION_SHA256": contract["calibration_sha256"],
        "TREEWM_FUTURE_RECIPE_SHA256": contract["future_recipe_sha256"],
        "TREEWM_DATA_CONTRACT_SHA256": contract["contract_sha256"],
        "TREEWM_DATA_ROOT": manifest["paths"]["data_root"],
        "TREEWM_CACHE": manifest["paths"]["raw_cache_root"],
        "TREEWM_FUTURE_RECIPE_ROOT": str(recipe_root(manifest, run.setting_id)),
        "TREEWM_EVALUATION_SEED_PROTOCOL_SHA256": setting["evaluation_seed_protocol_sha256"],
        "TREEWM_EXPECTED_FINAL_SEED_TABLE_SHA256": final_seed_table["sha256"],
        "TREEWM_RUN_NAME": run.run_name,
        "WANDB_PROJECT": manifest["logging"]["wandb_project"],
        "WANDB_RUN_GROUP": manifest["logging"]["wandb_group"],
        "WANDB_RUN_ID": run.wandb_id,
        "WANDB_MODE": manifest["logging"]["wandb_mode"],
        "OMP_NUM_THREADS": str(manifest["scientific_contract"]["loader_thread_limit"]),
        "MKL_NUM_THREADS": str(manifest["scientific_contract"]["loader_thread_limit"]),
        "OPENBLAS_NUM_THREADS": str(manifest["scientific_contract"]["loader_thread_limit"]),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    launch: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "formal_validation": True,
        "run": {**asdict(run), "run_directory": str(output)},
        "hashes": {
            "manifest_sha256": manifest_sha256(manifest),
            "source_sha256": source["source_sha256"],
            "evaluation_source_sha256": source["evaluation_source_sha256"],
            "runtime_sha256": source["runtime_sha256"],
            "package_protocol_sha256": protocol,
            "config_sha256": config_sha,
            "prerequisite_binding_sha256": prerequisite_binding["binding_sha256"],
            "selected_recipe_sha256": prerequisite_binding["selected_recipe_sha256"],
            "selected_arm": prerequisite_binding["selected_arm"],
            "run_protocol_sha256": run_protocol,
            "input_contract_sha256": contract["contract_sha256"],
            "calibration_sha256": contract["calibration_sha256"],
            "future_recipe_sha256": contract["future_recipe_sha256"],
            "evaluation_seed_protocol_sha256": setting["evaluation_seed_protocol_sha256"],
            "package_seed_table_sha256": seed_bundle["sha256"],
            "final_seed_table_sha256": final_seed_table["sha256"],
            "monitor_seed_table_sha256": expected_seed_tables["monitor"]["sha256"],
            "evaluation_seed_tables_sha256": expected_seed_tables["sha256"],
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
    verify_prerequisite_files: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    protocol = verify_protocol_lock(root / "experiments" / "22-treewm-grounded-gauge-formal-v1")
    prerequisite_binding = load_prerequisite_bindings(
        manifest,
        root / "experiments" / "22-treewm-grounded-gauge-formal-v1" / "prerequisite_bindings.json",
        verify_external_files=verify_prerequisite_files,
    )
    seed_table = load_seed_table(
        manifest,
        root / "experiments" / "22-treewm-grounded-gauge-formal-v1" / "eval_seed_table.json",
    )
    recipe_audits = {}
    for setting in manifest["settings"]:
        contract = load_compatible_input(manifest, setting, verify_files=verify_files)
        recipe_audits[setting["id"]] = contract["recipe_coverage_audit"]
    runs = expand_runs(manifest)
    source = source_contract(root)
    return {
        "schema_version": 1,
        "status": "verified",
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "package_protocol_sha256": protocol,
        "prerequisite_binding_sha256": prerequisite_binding["binding_sha256"],
        "selected_recipe_sha256": prerequisite_binding["selected_recipe_sha256"],
        "selected_arm": prerequisite_binding["selected_arm"],
        "source_sha256": source["source_sha256"],
        "evaluation_source_sha256": source["evaluation_source_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "evaluation_seed_table_sha256": seed_table["sha256"],
        "training_runs": len(runs),
        "final_eval_tasks": len([eval_at(manifest, i) for i in range(FINAL_EVAL_TASKS)]),
        "recipe_anchor_policy": manifest["scientific_contract"]["future_config"]["recipe_anchor_policy"],
        "recipe_coverage_audits": recipe_audits,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("protocol-hash", "verify", "snapshot", "runs", "evals"))
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "protocol-hash":
        print(protocol_sha256(args.repo_root / "experiments" / "22-treewm-grounded-gauge-formal-v1"))
        return 0
    manifest = load_manifest(args.repo_root / "experiments" / "22-treewm-grounded-gauge-formal-v1" / "manifest.json")
    if args.command == "snapshot":
        print(json.dumps(verify_source_snapshot(args.repo_root), sort_keys=True, indent=2))
    elif args.command == "verify":
        print(json.dumps(verify_all(manifest, repo_root=args.repo_root, verify_files=args.verify_files), sort_keys=True, indent=2))
    elif args.command == "runs":
        print(json.dumps([asdict(run) for run in expand_runs(manifest)], sort_keys=True, indent=2))
    else:
        print(json.dumps([asdict(eval_at(manifest, i)) for i in range(FINAL_EVAL_TASKS)], sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"grounded-formal campaign error: {exc}", file=sys.stderr)
        raise SystemExit(2)
