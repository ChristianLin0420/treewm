from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CAMPAIGN_DIR.parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))
sys.path.insert(0, str(REPO_ROOT))

# Experiment 12 also has top-level modules named campaign/report. Load this package's
# standalone scripts under the names they expect, then restore the process module table
# so combined-suite collection cannot cross-wire the two experiments.
_LOCAL_NAMES = ("campaign", "report", "submit", "worker")
_PREVIOUS = {name: sys.modules.get(name) for name in _LOCAL_NAMES}
for _name in _LOCAL_NAMES:
    sys.modules.pop(_name, None)
try:
    import campaign  # noqa: E402
    import report  # noqa: E402
    import submit  # noqa: E402
    import worker  # noqa: E402
finally:
    for _name in _LOCAL_NAMES:
        sys.modules.pop(_name, None)
        if _PREVIOUS[_name] is not None:
            sys.modules[_name] = _PREVIOUS[_name]


@pytest.fixture(scope="module")
def manifest():
    return campaign.load_manifest(CAMPAIGN_DIR / "manifest.json")


def test_exact_2x2_mapping_is_deterministic_and_complete(manifest):
    runs = campaign.expand_runs(manifest)
    assert len(runs) == 32
    assert [(run.index, run.setting_id, run.arm_id, run.seed) for run in runs[:8]] == [
        (0, "antmaze-large", "r0-g0", 0),
        (1, "antmaze-large", "r0-g0", 1),
        (2, "antmaze-large", "r0-g1", 0),
        (3, "antmaze-large", "r0-g1", 1),
        (4, "antmaze-large", "r1-g0", 0),
        (5, "antmaze-large", "r1-g0", 1),
        (6, "antmaze-large", "r1-g1", 0),
        (7, "antmaze-large", "r1-g1", 1),
    ]
    assert runs[-1].index == 31
    assert runs[-1].setting_id == "scene"
    assert len({run.run_name for run in runs}) == len({run.wandb_id for run in runs}) == 32


def test_run_directory_matches_the_trainer_arm_layout(manifest):
    run = campaign.run_at(manifest, 3)
    path = campaign.run_directory(manifest, run)
    assert path.parts[-3:] == (run.setting_id, "treewm", run.run_name)
    assert run.arm_id in run.run_name


def test_factorial_axes_are_the_only_arm_differences(manifest):
    runs = campaign.expand_runs(manifest)
    contracts = {
        run.setting_id: campaign.load_compatible_input(manifest, run)
        for run in runs
        if run.setting_id == "antmaze-large"
    }
    by_arm = {}
    for run in runs[:8:2]:
        overrides = set(campaign.scientific_overrides(manifest, run, contracts[run.setting_id]))
        by_arm[run.arm_id] = overrides
        for invariant in (
            "train.steps=12000",
            "train.ckpt_every=2000",
            "train.eval_every=6000",
            "planner.decoded_metric=domain_raw",
            "planner.execute_steps=4",
            "tree.max_depth=3",
            "model.max_depth=3",
            "future_sets.cache=false",
            "future_sets.shared_cache=true",
            "future_sets.retrieval_pool=50000",
            "eval.final_episodes_per_task=1",
        ):
            assert invariant in overrides
    assert "train.lr=0.0003" in by_arm["r0-g0"]
    assert "model.dropout=0.1" in by_arm["r1-g1"]
    assert "losses.enabled.multistep=false" in by_arm["r1-g0"]
    assert "losses.enabled.multistep=true" in by_arm["r0-g1"]
    assert "losses.weights.multistep=1.0" in by_arm["r1-g1"]
    assert "losses.scheduled_sampling_p=0.25" in by_arm["r1-g1"]


def test_current_trainer_and_recipe_producer_identities_are_separate(manifest):
    launch = campaign.trainer_command(manifest, campaign.expand_runs(manifest)[0], repo_root=REPO_ROOT)
    env = launch["environment"]
    assert env["TREEWM_CODE_SHA256"] == launch["hashes"]["source_sha256"]
    assert env["TREEWM_RUNTIME_SHA256"] == launch["hashes"]["runtime_sha256"]
    assert env["TREEWM_RECIPE_CODE_SHA256"] == "4cb70b4421d3eae1a6e947e3b0359336bd64248897ffcf52d4a672ca4adcd30c"
    assert env["TREEWM_RECIPE_RUNTIME_SHA256"] == "77da91d49a1db99850fbf0632dc02ec58a3209f1a87949d6f5640ae6bf505c6b"
    assert launch["hashes"]["compatible_recipe_runtime_sha256"] == env["TREEWM_RECIPE_RUNTIME_SHA256"]
    assert env["TREEWM_RECIPE_CODE_SHA256"] != env["TREEWM_CODE_SHA256"]
    assert launch["formal_validation"] is False
    assert launch["hashes"]["config_sha256"] in " ".join(launch["argv"])


def test_protocol_and_slurm_lifecycle_are_locked(manifest):
    assert campaign.verify_protocol_lock(CAMPAIGN_DIR) == (CAMPAIGN_DIR / "protocol.sha256").read_text().strip()
    submit.validate_slurm(CAMPAIGN_DIR / "pilot.slurm")
    text = (CAMPAIGN_DIR / "pilot.slurm").read_text()
    assert "/cm/shared/apps/slurm/current/bin/srun" in text
    assert "/cm/shared/apps/slurm/current/bin/scontrol" in text
    assert '"$SCONTROL" requeue "$REQUEUE_TARGET"' in text
    assert "CANCEL_REQUESTED" in text and "READY_FOR_REQUEUE.json" in text
    assert "while true; do" in text and 'kill -0 "$step_pid"' in text
    assert "#SBATCH --array=0-31%32" in text
    assert "#SBATCH --gpus-per-node=1" in text
    assert 'f"--output={run_root}/logs/%x_%A_%a.out"' in (CAMPAIGN_DIR / "submit.py").read_text()
    submit_text = (CAMPAIGN_DIR / "submit.py").read_text()
    assert '"--parsable"' in submit_text
    assert "SUBMISSION_RECEIPT.json" in submit_text
    assert "os.O_EXCL" in submit_text
    assert "TREEWM_EXPECTED_RUNTIME_SHA256" in submit_text
    assert "verify_files=True" in submit_text


def test_corrupt_existing_launch_identity_fails_closed(tmp_path):
    path = tmp_path / "launch.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(campaign.ContractError, match="unreadable"):
        worker.read_json(path)


def test_submit_path_is_single_array_and_persists_job_receipt(
    manifest, tmp_path, monkeypatch
):
    isolated = copy.deepcopy(manifest)
    run_root = tmp_path / "corrected-pilot-output"
    isolated["paths"]["run_root"] = str(run_root)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(isolated), encoding="utf-8")
    plan = {
        "schema_version": 1,
        "status": "sealed_dry_run",
        "campaign_id": isolated["campaign_id"],
        "source_sha256": "1" * 64,
        "runtime_sha256": "2" * 64,
        "package_protocol_sha256": "3" * 64,
        "runs": [],
    }
    monkeypatch.setattr(submit, "launch_plan", lambda _manifest, _root: plan)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="123456\n", stderr="")

    monkeypatch.setattr(submit.subprocess, "run", fake_run)
    assert submit.main(
        ["--manifest", str(manifest_path), "--repo-root", str(REPO_ROOT), "--submit"]
    ) == 0
    receipt = json.loads(
        (run_root / "state" / "SUBMISSION_RECEIPT.json").read_text(encoding="utf-8")
    )
    assert receipt["job_id"] == "123456"
    assert calls[0][0][1] == "--parsable"
    with pytest.raises(campaign.ContractError, match="already submitted"):
        submit.main(
            ["--manifest", str(manifest_path), "--repo-root", str(REPO_ROOT), "--submit"]
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["shared_contract"].update(optimizer_updates=20_001),
        lambda value: value["shared_contract"].update(planner_decoded_metric="normalized_l2"),
        lambda value: value["factorial"]["arms"][3].update(dropout=0.2),
        lambda value: value["compatible_v2_recipe_input"].update(read_only=False),
        lambda value: value["execution"].update(array="0-31%1"),
    ],
)
def test_manifest_rejects_scientific_or_execution_drift(manifest, mutation):
    changed = copy.deepcopy(manifest)
    mutation(changed)
    with pytest.raises(campaign.ContractError):
        campaign.validate_manifest(changed)


def _synthetic_records(manifest):
    records = []
    for run in campaign.expand_runs(manifest):
        control_progress = 0.10
        progress = 0.20 if run.arm_id == "r1-g1" else control_progress
        records.append(
            {
                "setting_id": run.setting_id,
                "arm_id": run.arm_id,
                "seed": run.seed,
                "internal_pass": run.arm_id == "r1-g1",
                "gates": {"complete_identity": True},
                "metrics": {
                    "progress_6k": 0.10 if run.arm_id == "r1-g1" else 0.08,
                    "progress_12k": progress,
                    "success_12k": 0.2 if run.arm_id == "r1-g1" else 0.1,
                },
            }
        )
    return records


def test_report_accepts_only_next_bounded_pilot_and_never_formal(manifest):
    result = report.aggregate_acceptance(manifest, _synthetic_records(manifest))
    assert result["accepted"] is True
    assert result["status"] == "accepted_for_next_bounded_pilot"
    assert result["formal_validation"] is False
    assert "never formal validation" in result["claim"]


def test_report_fails_closed_on_missing_run_or_control_regression(manifest):
    records = _synthetic_records(manifest)
    assert report.aggregate_acceptance(manifest, records[:-1])["accepted"] is False
    for record in records:
        if record["arm_id"] == "r1-g1":
            record["metrics"]["progress_12k"] = -0.2
    result = report.aggregate_acceptance(manifest, records)
    assert result["accepted"] is False
    assert result["comparison_gates"]["candidate_progress_vs_regularized_control"] is False
