#!/usr/bin/env python3
"""Evaluate one training-run/task cell on paired learned and BFS rails."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Mapping, Sequence

import torch

from campaign import (
    CAMPAIGN_DIR,
    ContractError,
    REPOSITORY_ROOT,
    atomic_json,
    eval_at,
    load_manifest,
    load_seed_table,
    read_json,
    run_directory,
    stable_hash,
    trainer_command,
)
from worker import verify_stage_marker


REQUEUE_EXIT_CODE = 75
CANCEL_EXIT_CODE = 143


class StopState:
    def __init__(self) -> None:
        self.reason: str | None = None
        self.cancel = False

    def request_requeue(self, signum: int, _frame: object) -> None:
        if self.reason is None:
            self.reason = signal.Signals(signum).name

    def request_cancel(self, signum: int, _frame: object) -> None:
        self.cancel = True
        if self.reason is None:
            self.reason = signal.Signals(signum).name


def single_task_seed_table(table: Mapping[str, Any], task_id: int) -> dict[str, Any]:
    from treewm.evaluation.rollout import validate_evaluation_seed_table

    validate_evaluation_seed_table(table, split="final", task_ids=[1, 2, 3, 4, 5], episodes_per_task=50)
    position = list(table["task_ids"]).index(int(task_id))
    result: dict[str, Any] = {
        "schema_version": 1,
        "split": "final",
        "protocol_sha256": table["protocol_sha256"],
        "task_ids": [int(task_id)],
        "episodes_per_task": 50,
        "seeds": [list(table["seeds"][position])],
    }
    result["sha256"] = stable_hash(result)
    validate_evaluation_seed_table(result, split="final", task_ids=[task_id], episodes_per_task=50)
    return result


def encode_generator_state(generator: torch.Generator) -> list[int]:
    return [int(value) for value in generator.get_state().detach().cpu().tolist()]


def restore_generator_state(generator: torch.Generator, state: Sequence[int]) -> None:
    if not state:
        raise ContractError("resumed rail lacks planner generator state")
    tensor = torch.tensor([int(value) for value in state], dtype=torch.uint8, device="cpu")
    generator.set_state(tensor)


def seal_progress(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach a canonical self-hash after every durable progress mutation."""
    payload.pop("progress_sha256", None)
    payload["progress_sha256"] = stable_hash(payload)
    return payload


def validate_progress(
    payload: Mapping[str, Any],
    identity: Mapping[str, Any],
    seed_table: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject non-prefix, cross-rail, or byte-corrupted resume state."""
    body = dict(payload)
    claimed = body.pop("progress_sha256", None)
    if claimed != stable_hash(body):
        raise ContractError("existing final-eval progress self-hash differs")
    if (
        set(payload) != {
            "schema_version",
            "status",
            "identity",
            "rails",
            "progress_sha256",
        }
        or payload.get("schema_version") != 1
        or payload.get("status") != "in_progress"
        or payload.get("identity") != dict(identity)
    ):
        raise ContractError("existing final-eval progress identity differs")
    rails = payload.get("rails")
    if not isinstance(rails, dict) or set(rails) != {"learned", "bfs"}:
        raise ContractError("existing final-eval progress rail coverage differs")
    expected_seeds = list(seed_table["seeds"][0])
    for rail in ("learned", "bfs"):
        rail_state = rails[rail]
        if not isinstance(rail_state, dict) or not {
            "episodes",
            "metrics",
        }.issubset(rail_state) or not set(rail_state).issubset(
            {"episodes", "metrics", "planner_generator_state"}
        ):
            raise ContractError(f"{rail}: final-eval progress structure differs")
        episodes = rail_state.get("episodes")
        if not isinstance(episodes, list) or len(episodes) > 50:
            raise ContractError(f"{rail}: final-eval progress episode count differs")
        if any(not isinstance(row, dict) for row in episodes):
            raise ContractError(f"{rail}: final-eval progress episode row differs")
        actual_prefix = [
            (row.get("task_id"), row.get("episode_index"), row.get("episode_seed"))
            for row in episodes
        ]
        expected_prefix = [
            (int(identity["task_id"]), episode, seed)
            for episode, seed in enumerate(expected_seeds)
        ]
        if actual_prefix != expected_prefix[: len(actual_prefix)]:
            raise ContractError(f"{rail}: final-eval progress is not the locked seed prefix")
        generator_state = rail_state.get("planner_generator_state")
        if episodes and (
            not isinstance(generator_state, list)
            or not generator_state
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 255
                for value in generator_state
            )
        ):
            raise ContractError(f"{rail}: final-eval progress generator state differs")
        metrics = rail_state.get("metrics")
        if metrics is not None and (
            not isinstance(metrics, dict)
            or len(episodes) != 50
            or metrics.get("eval/num_episodes") != 50.0
        ):
            raise ContractError(f"{rail}: completed progress metrics/coverage differ")
    if rails["bfs"].get("episodes") and rails["learned"].get("metrics") is None:
        raise ContractError("BFS progress exists before the learned rail completed")
    return dict(payload)


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    seal_progress(payload)
    atomic_json(path, payload)


def verify_final_stage_gate(manifest: Mapping[str, Any], launch: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(manifest["paths"]["run_root"]) / "state" / "stage-gates" / "STAGE_GATE_1000000.json"
    gate = read_json(path)
    claimed = gate.get("gate_sha256")
    body = dict(gate)
    body.pop("gate_sha256", None)
    if claimed != stable_hash(body):
        raise ContractError("final training-stage gate hash differs")
    if gate.get("status") != "accepted" or int(gate.get("stage_target", -1)) != 1_000_000 or len(gate.get("runs") or []) != 40:
        raise ContractError("final evaluation requires an accepted exact 40-run 1M gate")
    if gate.get("evaluation_source_sha256") != launch["hashes"]["evaluation_source_sha256"]:
        raise ContractError("final training-stage gate evaluation source differs")
    if gate.get("prerequisite_binding_sha256") != launch["hashes"]["prerequisite_binding_sha256"]:
        raise ContractError("final training-stage gate prerequisite binding differs")
    if gate.get("selected_recipe_sha256") != launch["hashes"]["selected_recipe_sha256"]:
        raise ContractError("final training-stage gate selected recipe differs")
    row = next((value for value in gate["runs"] if int(value.get("index", -1)) == int(launch["run"]["index"])), None)
    if not (
        row is not None
        and row.get("launch_sha256") == launch["launch_sha256"]
        and row.get("identity_sha256") == checkpoint["identity_sha256"]
        and row.get("checkpoint_sha256") == checkpoint["checkpoint_sha256"]
        and row.get("final_seed_table_sha256") == launch["hashes"]["final_seed_table_sha256"]
    ):
        raise ContractError("final stage gate does not bind this run/checkpoint")
    return gate


def _progress_identity(manifest: Mapping[str, Any], spec, launch: Mapping[str, Any], checkpoint: Mapping[str, Any], seed_table: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "eval_index": spec.index,
        "training_index": spec.training_index,
        "setting_id": spec.run.setting_id,
        "training_seed": spec.run.seed,
        "task_id": spec.task_id,
        "rails": ["learned", "bfs"],
        "episodes_per_task_per_rail": 50,
        "launch_sha256": launch["launch_sha256"],
        "identity_sha256": checkpoint["identity_sha256"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "evaluation_seed_protocol_sha256": seed_table["protocol_sha256"],
        "task_seed_table_sha256": seed_table["sha256"],
        "full_final_seed_table_sha256": launch["hashes"]["final_seed_table_sha256"],
        "package_seed_table_sha256": launch["hashes"]["package_seed_table_sha256"],
        "package_protocol_sha256": launch["hashes"]["package_protocol_sha256"],
        "source_sha256": launch["hashes"]["source_sha256"],
        "evaluation_source_sha256": launch["hashes"]["evaluation_source_sha256"],
        "runtime_sha256": launch["hashes"]["runtime_sha256"],
        "prerequisite_binding_sha256": launch["hashes"]["prerequisite_binding_sha256"],
        "selected_recipe_sha256": launch["hashes"]["selected_recipe_sha256"],
        "selected_arm": launch["hashes"]["selected_arm"],
    }
    identity["eval_contract_sha256"] = stable_hash(identity)
    return identity


def _validate_existing_result(path: Path, identity: Mapping[str, Any], seed_table: Mapping[str, Any]) -> dict[str, Any]:
    result = read_json(path)
    claimed = result.get("result_sha256")
    body = dict(result)
    body.pop("result_sha256", None)
    if claimed != stable_hash(body):
        raise ContractError(f"existing final result hash differs: {path}")
    identity_body = dict(identity)
    identity_claim = identity_body.pop("eval_contract_sha256", None)
    if (
        result.get("status") != "complete"
        or result.get("identity") != dict(identity)
        or identity_claim != stable_hash(identity_body)
    ):
        raise ContractError(f"existing final result identity differs: {path}")
    expected_seeds = list(seed_table["seeds"][0])
    rails = result.get("rails") or {}
    if set(rails) != {"learned", "bfs"} or identity.get("rails") != ["learned", "bfs"]:
        raise ContractError("final result rail order/coverage differs")
    for rail in ("learned", "bfs"):
        episodes = rails[rail].get("episodes") or []
        if len(episodes) != 50 or [row.get("episode_seed") for row in episodes] != expected_seeds:
            raise ContractError(f"{rail}: result lacks exact locked 50-episode coverage")
        episode_count = (rails[rail].get("metrics") or {}).get("eval/num_episodes")
        if isinstance(episode_count, bool) or episode_count != 50:
            raise ContractError(f"{rail}: result metrics episode count differs")
    return result


def run_final_eval(args: argparse.Namespace) -> int:
    stop = StopState()
    signal.signal(signal.SIGUSR1, stop.request_requeue)
    signal.signal(signal.SIGTERM, stop.request_cancel)
    signal.signal(signal.SIGINT, stop.request_cancel)
    root = args.repo_root.resolve()
    manifest = load_manifest(args.manifest)
    spec = eval_at(manifest, args.index)
    launch = trainer_command(manifest, spec.run, repo_root=root)
    for env_name, hash_name in (
        ("TREEWM_EXPECTED_SOURCE_SHA256", "source_sha256"),
        ("TREEWM_EXPECTED_EVALUATION_SOURCE_SHA256", "evaluation_source_sha256"),
        ("TREEWM_EXPECTED_RUNTIME_SHA256", "runtime_sha256"),
        ("TREEWM_EXPECTED_PACKAGE_PROTOCOL_SHA256", "package_protocol_sha256"),
        ("TREEWM_EXPECTED_SEED_TABLE_SHA256", "package_seed_table_sha256"),
        ("TREEWM_EXPECTED_PREREQUISITE_BINDING_SHA256", "prerequisite_binding_sha256"),
        ("TREEWM_EXPECTED_SELECTED_RECIPE_SHA256", "selected_recipe_sha256"),
    ):
        expected = os.environ.get(env_name)
        if expected is None or expected != launch["hashes"][hash_name]:
            raise ContractError(f"{env_name} differs from sealed launch")
    run_dir = run_directory(manifest, spec.run)
    checkpoint_record = verify_stage_marker(run_dir, 1_000_000, launch)
    verify_final_stage_gate(manifest, launch, checkpoint_record)
    seed_bundle = load_seed_table(manifest, root / "experiments" / "22-treewm-grounded-gauge-formal-v1" / "eval_seed_table.json")
    full_seed_table = seed_bundle["settings"][spec.run.setting_id]
    if full_seed_table["sha256"] != launch["hashes"]["final_seed_table_sha256"]:
        raise ContractError("locked final seed table differs from training identity")
    task_seed_table = single_task_seed_table(full_seed_table, spec.task_id)
    identity = _progress_identity(manifest, spec, launch, checkpoint_record, task_seed_table)

    result_root = Path(manifest["paths"]["final_eval_root"]) / "results"
    result_path = result_root / f"{spec.index:03d}.json"
    progress_path = result_root / f"{spec.index:03d}.progress.json"
    if result_path.exists():
        _validate_existing_result(result_path, identity, task_seed_table)
        return 0

    progress: dict[str, Any] = {
        "schema_version": 1,
        "status": "in_progress",
        "identity": identity,
        "rails": {
            "learned": {"episodes": [], "metrics": None},
            "bfs": {"episodes": [], "metrics": None},
        },
    }
    if progress_path.exists():
        progress = validate_progress(
            read_json(progress_path), identity, task_seed_table
        )
    else:
        write_progress(progress_path, progress)

    def publish_requeue_ready() -> None:
        value = os.environ.get("TREEWM_EVAL_REQUEUE_READY")
        if not value:
            raise ContractError("final-eval requeue has no sealed READY path")
        current = validate_progress(
            read_json(progress_path), identity, task_seed_table
        )
        record = {
            "schema_version": 1,
            "status": "final_eval_progress_verified_for_requeue",
            "eval_contract_sha256": identity["eval_contract_sha256"],
            "progress_sha256": current["progress_sha256"],
            "unix_time": time.time(),
        }
        atomic_json(Path(value), record)

    from scripts.eval import load_run
    from treewm.data.ogbench_dataset import load_ogbench
    from treewm.evaluation.domains import get_domain
    from treewm.evaluation.rollout import EvaluationInterrupted, evaluate
    from treewm.evaluation.tasks import build_tasks
    from treewm.models.baselines import tree_config_for
    from treewm.planning.goal_planner import GoalPlanner
    from treewm.utils import config as cfg_utils
    from treewm.utils.seeding import seed_everything

    checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(0)
    model, normalizer, cfg, payload = load_run(str(checkpoint_path), device)
    if payload.get("identity_sha256") != checkpoint_record["identity_sha256"]:
        raise ContractError("loaded final checkpoint identity differs")
    env = load_ogbench(cfg.env.name, dataset_dir=cfg.env.dataset_dir, env_only=True)
    all_tasks = build_tasks(
        env,
        str(cfg.eval.task_split),
        int(cfg.eval.num_hard_tasks),
        float(cfg.eval.hard_percentile),
        int(cfg.eval.seed),
    )
    selected = [task for index, task in enumerate(all_tasks) if int(task.get("task_id", index + 1)) == spec.task_id]
    if len(selected) != 1:
        raise ContractError(f"task {spec.task_id} is not unique in checkpoint task split")
    domain = get_domain(cfg.env.name)
    base_tree = tree_config_for(cfg.arm, cfg_utils.tree_config(cfg), model)

    try:
        for rail in ("learned", "bfs"):
            rail_state = progress["rails"][rail]
            if rail_state.get("metrics") is not None:
                continue
            tree_cfg = replace(base_tree, scorer=rail)
            planner = GoalPlanner(model, normalizer, tree_cfg, cfg_utils.planner_config(cfg), device, domain=domain)
            completed_episodes = list(rail_state.get("episodes") or [])
            if completed_episodes:
                restore_generator_state(planner.generator, rail_state.get("planner_generator_state") or [])

            def persist_episode(row: dict[str, Any], *, _rail: str = rail) -> None:
                progress["rails"][_rail]["episodes"].append(row)
                progress["rails"][_rail]["planner_generator_state"] = encode_generator_state(planner.generator)
                write_progress(progress_path, progress)

            metrics = evaluate(
                env,
                planner,
                selected,
                50,
                int(cfg.planner.max_env_steps),
                0,
                domain=domain,
                stop_callback=lambda: stop.reason is not None,
                completed_results=completed_episodes,
                episode_callback=persist_episode,
                episode_seed_table=task_seed_table,
                expected_episode_seed_split="final",
            )
            progress["rails"][rail]["metrics"] = metrics
            progress["rails"][rail]["planner_generator_state"] = encode_generator_state(planner.generator)
            write_progress(progress_path, progress)
    except EvaluationInterrupted:
        if not stop.cancel:
            publish_requeue_ready()
        return CANCEL_EXIT_CODE if stop.cancel else REQUEUE_EXIT_CODE
    if stop.reason is not None:
        if not stop.cancel:
            publish_requeue_ready()
        return CANCEL_EXIT_CODE if stop.cancel else REQUEUE_EXIT_CODE

    validate_progress(progress, identity, task_seed_table)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "identity": identity,
        "rails": progress["rails"],
        "completed_unix_time": time.time(),
    }
    result["result_sha256"] = stable_hash(result)
    atomic_json(result_path, result)
    _validate_existing_result(result_path, identity, task_seed_table)
    progress_path.unlink()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_DIR / "manifest.json")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run_final_eval(_parser().parse_args()))
    except ContractError as exc:
        print(f"grounded-formal final-eval error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
