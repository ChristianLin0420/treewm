#!/bin/bash
#SBATCH --job-name=treewm-all10-bridge-v1
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@420
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --array=0-39%40
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --open-mode=append

set -Eeuo pipefail

PROJECT_ROOT="${TREEWM_BRIDGE_REPO_ROOT:?launch through guarded submit.py with a sealed snapshot}"
CAMPAIGN_DIR="$PROJECT_ROOT/experiments/16-treewm-grounded-repair-all-ten-bridge-v1"
PYTHON_EXECUTABLE=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
SRUN=/cm/shared/apps/slurm/current/bin/srun
SCONTROL=/cm/shared/apps/slurm/current/bin/scontrol
: "${SLURM_ARRAY_JOB_ID:?all-ten bridge requires the 40-element array}"
: "${SLURM_ARRAY_TASK_ID:?all-ten bridge requires the 40-element array}"
: "${TREEWM_EXPECTED_SOURCE_SHA256:?launch through guarded submit.py}"
: "${TREEWM_EXPECTED_RUNTIME_SHA256:?launch through guarded submit.py}"
: "${TREEWM_EXPECTED_PACKAGE_PROTOCOL_SHA256:?launch through guarded submit.py}"
: "${TREEWM_EXPECTED_EXP15_PREREQUISITE_SHA256:?exp15 acceptance must be validated by guarded submit.py}"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= 40 )); then
    exit 2
fi
if [[ ! -x "$PYTHON_EXECUTABLE" || ! -x "$SRUN" || ! -x "$SCONTROL" ]]; then
    echo "pinned Python/Slurm executable is unavailable" >&2
    exit 2
fi
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/campaign.py" snapshot --repo-root "$PROJECT_ROOT" >/dev/null

REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RUN_ROOT=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/treewm/outputs/treewm-grounded-repair-all-ten-bridge-v1
TASK_STATE="$RUN_ROOT/state/arrays/array-job-$REQUEUE_TARGET/task-$(printf '%02d' "$SLURM_ARRAY_TASK_ID")"
GENERATION_DIR="$TASK_STATE/requeue/${SLURM_RESTART_COUNT:-0}"
CANCEL_LATCH="$TASK_STATE/CANCEL_REQUESTED"
READY_PATH="$TASK_STATE/READY_FOR_REQUEUE.json"
COMPLETE_PATH="$TASK_STATE/WORKER_COMPLETE.json"
REQUEUE_CALLING="$GENERATION_DIR/REQUEUE_CALLING.json"
mkdir -p "$GENERATION_DIR"

step_pid=""
on_usr1() {
    if [[ -n "$step_pid" ]] && kill -0 "$step_pid" 2>/dev/null; then
        kill -USR1 "$step_pid" 2>/dev/null || true
    fi
}
on_cancel() {
    if [[ -s "$REQUEUE_CALLING" ]]; then
        return
    fi
    mkdir -p "$TASK_STATE"
    touch "$CANCEL_LATCH"
}
trap on_usr1 USR1
trap on_cancel TERM INT
if [[ -e "$CANCEL_LATCH" ]]; then
    exit 143
fi

cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PATH="/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin"

set +e
"$SRUN" --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 --cpus-per-task=12 \
    --kill-on-bad-exit=0 \
    "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/worker.py" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --state-dir "$TASK_STATE" \
    --repo-root "$PROJECT_ROOT" \
    --manifest "$CAMPAIGN_DIR/manifest.json" &
step_pid=$!
status=0
while true; do
    wait "$step_pid"
    status=$?
    if ! kill -0 "$step_pid" 2>/dev/null; then
        break
    fi
done
step_pid=""
set -e

if [[ -e "$CANCEL_LATCH" ]] || [[ "$status" -eq 143 ]]; then
    exit 143
fi
if [[ "$status" -eq 0 ]]; then
    if [[ ! -s "$COMPLETE_PATH" ]]; then
        echo "worker returned success without a durable completion artifact" >&2
        exit 2
    fi
    exit 0
fi
if [[ "$status" -ne 75 || ! -s "$READY_PATH" ]]; then
    echo "worker failed without a verified exact-resume checkpoint (status=$status)" >&2
    exit "$status"
fi

job_record="$($SCONTROL show job "$REQUEUE_TARGET" --oneliner 2>&1)" || exit 2
if [[ "$job_record" =~ JobState=(CANCELLED|COMPLETING|COMPLETED|FAILED|TIMEOUT) ]]; then
    exit 143
fi
if [[ -e "$CANCEL_LATCH" ]]; then
    exit 143
fi
"$PYTHON_EXECUTABLE" - "$REQUEUE_CALLING" "$REQUEUE_TARGET" "${SLURM_RESTART_COUNT:-0}" "$READY_PATH" <<'PY'
import json
import os
from pathlib import Path
import sys
import time

path = Path(sys.argv[1])
ready = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
if ready.get("status") != "checkpoint_verified":
    raise SystemExit("invalid READY_FOR_REQUEUE record")
payload = {
    "status": "scontrol_requeue_calling",
    "job_id": sys.argv[2],
    "restart_count": int(sys.argv[3]),
    "completed_updates": int(ready["completed_updates"]),
    "identity_sha256": ready["identity_sha256"],
    "launch_sha256": ready["launch_sha256"],
    "unix_time": time.time(),
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
if ! "$SCONTROL" requeue "$REQUEUE_TARGET"; then
    mv "$REQUEUE_CALLING" "$GENERATION_DIR/REQUEUE_FAILED.json"
    exit 2
fi
exit 75
