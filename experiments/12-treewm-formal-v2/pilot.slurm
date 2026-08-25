#!/bin/bash
#SBATCH --job-name=treewm-v2-pilot
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@420
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --array=0-9%10
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append

set -Eeuo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?submit from repository root with submit.py}"
CAMPAIGN_DIR="$PROJECT_ROOT/experiments/12-treewm-formal-v2"
PYTHON_EXECUTABLE="${TREEWM_PYTHON:-/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python}"
: "${TREEWM_DATA_ROOT:?set immutable OGBench source root}"
: "${TREEWM_CACHE:?set approved reused raw cache root}"
: "${TREEWM_PILOT_ROOT:?set isolated v2 pilot root}"
: "${SLURM_ARRAY_JOB_ID:?pilot requires a ten-setting array}"
: "${SLURM_ARRAY_TASK_ID:?pilot requires a ten-setting array}"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= 10 )); then
    exit 2
fi
if [[ "${TREEWM_PILOT_ROOT,,}" == *v1* ]]; then
    echo "v2 pilot refuses a v1 namespace" >&2
    exit 2
fi

REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
PILOT_STATE="$TREEWM_PILOT_ROOT/state/arrays/array-job-$REQUEUE_TARGET/task-$SLURM_ARRAY_TASK_ID"
CANCEL_LATCH="$PILOT_STATE/CANCEL_REQUESTED"
GENERATION_DIR="$PILOT_STATE/requeue/${SLURM_RESTART_COUNT:-0}"
REQUEUE_CALLING="$GENERATION_DIR/REQUEUE_CALLING.json"
mkdir -p "$GENERATION_DIR"

cancel_requested=0
requeue_requested=0
step_pid=""
on_usr1() {
    requeue_requested=1
    if [[ -n "$step_pid" ]] && kill -0 "$step_pid" 2>/dev/null; then
        kill -USR1 "$step_pid" 2>/dev/null || true
    fi
}
on_cancel() {
    if [[ -s "$REQUEUE_CALLING" ]]; then
        return
    fi
    cancel_requested=1
    touch "$CANCEL_LATCH"
    if [[ -n "$step_pid" ]] && kill -0 "$step_pid" 2>/dev/null; then
        kill -TERM "$step_pid" 2>/dev/null || true
    fi
}
trap on_usr1 USR1
trap on_cancel TERM INT
if [[ -e "$CANCEL_LATCH" ]]; then
    exit 143
fi

cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/submit.py" \
    --verify-source-snapshot --repo-root "$PROJECT_ROOT"
export WANDB_MODE=online
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

run_step() {
    set +e
    srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 --cpus-per-task=12 \
        --kill-on-bad-exit=0 "$@" &
    step_pid=$!
    wait "$step_pid"
    local status=$?
    step_pid=""
    set -e
    return "$status"
}

AUDIT_DIR="$TREEWM_PILOT_ROOT/state/gradient-audits"
mkdir -p "$AUDIT_DIR"
status=0
run_step "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/validate_pilot.py" \
    --run-setting --setting-index "$SLURM_ARRAY_TASK_ID" \
    --manifest "$CAMPAIGN_DIR/manifest.json" \
    --repo-root "$PROJECT_ROOT" \
    --data-root "$TREEWM_DATA_ROOT" \
    --cache-root "$TREEWM_CACHE" \
    --pilot-root "$TREEWM_PILOT_ROOT" || status=$?
if [[ "$status" -eq 0 ]] && (( ! requeue_requested )) && (( ! cancel_requested )); then
    # Audit the trained weights restored from the exact 5k checkpoint, not random
    # initialization. Three representative batches are fixed and fully recorded.
    run_step "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/gpu_preflight.py" \
        --audit-setting-index "$SLURM_ARRAY_TASK_ID" --post-training-checkpoint \
        --output-dir "$AUDIT_DIR" \
        --manifest "$CAMPAIGN_DIR/manifest.json" \
        --repo-root "$PROJECT_ROOT" \
        --data-root "$TREEWM_DATA_ROOT" \
        --cache-root "$TREEWM_CACHE" || status=$?
fi
if [[ "$status" -eq 0 ]] && (( ! requeue_requested )) && (( ! cancel_requested )); then
    run_step "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/validate_pilot.py" \
        --validate-setting --setting-index "$SLURM_ARRAY_TASK_ID" \
        --manifest "$CAMPAIGN_DIR/manifest.json" \
        --repo-root "$PROJECT_ROOT" \
        --data-root "$TREEWM_DATA_ROOT" \
        --cache-root "$TREEWM_CACHE" \
        --pilot-root "$TREEWM_PILOT_ROOT" || status=$?
fi

if (( cancel_requested )) || [[ -e "$CANCEL_LATCH" ]]; then
    exit 143
fi
if (( requeue_requested )) || [[ "$status" -eq 75 ]]; then
    job_record="$(scontrol show job "$REQUEUE_TARGET" --oneliner 2>&1)" || exit 2
    if [[ "$job_record" =~ JobState=(CANCELLED|COMPLETING|COMPLETED|FAILED|TIMEOUT) ]]; then
        exit 143
    fi
    "$PYTHON_EXECUTABLE" - "$REQUEUE_CALLING" "$REQUEUE_TARGET" "${SLURM_RESTART_COUNT:-0}" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump({"job_id": sys.argv[2], "restart_count": int(sys.argv[3]),
               "status": "pilot_scontrol_requeue_calling", "unix_time": time.time()},
              handle, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(fd)
finally:
    os.close(fd)
PY
    if ! scontrol requeue "$REQUEUE_TARGET"; then
        rm -f "$REQUEUE_CALLING"
        exit 2
    fi
    exit 75
fi
exit "$status"
