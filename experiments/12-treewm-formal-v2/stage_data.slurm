#!/bin/bash
#SBATCH --job-name=treewm-v2-calibrate
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
: "${TREEWM_CONTRACT_ROOT:?set fresh v2 contract/calibration root}"
: "${SLURM_ARRAY_JOB_ID:?v2 stage is a ten-element array}"
: "${SLURM_ARRAY_TASK_ID:?v2 stage is a ten-element array}"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= 10 )); then
    exit 2
fi
STAGE_PHASE="${TREEWM_STAGE_PHASE:-calibrate}"
if [[ "$STAGE_PHASE" != "calibrate" && "$STAGE_PHASE" != "recipe" ]]; then
    echo "TREEWM_STAGE_PHASE must be calibrate or recipe" >&2
    exit 2
fi
if [[ "${TREEWM_CONTRACT_ROOT,,}" == *treewm-50task-full-cache-v1* ]]; then
    echo "v2 contracts may not overwrite reused raw cache content" >&2
    exit 2
fi

REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
STAGE_STATE="$TREEWM_CONTRACT_ROOT/stage-state/$STAGE_PHASE/array-$SLURM_ARRAY_JOB_ID/task-$SLURM_ARRAY_TASK_ID"
CANCEL_LATCH="$STAGE_STATE/CANCEL_REQUESTED"
GENERATION_DIR="$STAGE_STATE/requeue/${SLURM_RESTART_COUNT:-0}"
REQUEUE_CALLING="$GENERATION_DIR/REQUEUE_CALLING.json"
mkdir -p "$GENERATION_DIR"

cancel_requested=0
stage_step_pid=""
on_cancel() {
    if [[ -s "$REQUEUE_CALLING" ]]; then
        return
    fi
    cancel_requested=1
    touch "$CANCEL_LATCH"
    if [[ -n "$stage_step_pid" ]] && kill -0 "$stage_step_pid" 2>/dev/null; then
        kill -TERM "$stage_step_pid" 2>/dev/null || true
    fi
}
trap on_cancel TERM INT
if [[ -e "$CANCEL_LATCH" ]]; then
    exit 143
fi

cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/submit.py" \
    --verify-source-snapshot --repo-root "$PROJECT_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
# prepare_cache reads the reused cache but publishes only atomic v2 namespaced
# contract/calibration artifacts.  It commits a safe boundary before returning 75.
set +e
srun --ntasks=1 --cpus-per-task=12 --kill-on-bad-exit=0 \
    "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/prepare_cache.py" \
        --manifest "$CAMPAIGN_DIR/manifest.json" \
        --phase "$STAGE_PHASE" \
        --setting-index "$SLURM_ARRAY_TASK_ID" \
        --data-root "$TREEWM_DATA_ROOT" \
        --cache-root "$TREEWM_CACHE" \
        --contract-root "$TREEWM_CONTRACT_ROOT" &
stage_step_pid=$!
wait "$stage_step_pid"
stage_status=$?
set -e
stage_step_pid=""

if (( cancel_requested )) || [[ -e "$CANCEL_LATCH" ]]; then
    exit 143
fi
if [[ "$stage_status" -eq 75 ]]; then
    for _ in 1 2 3 4 5; do
        if (( cancel_requested )) || [[ -e "$CANCEL_LATCH" ]]; then
            exit 143
        fi
        sleep 0.2
    done
    job_record="$(scontrol show job "$REQUEUE_TARGET" --oneliner 2>&1)" || exit 2
    if [[ "$job_record" =~ JobState=(CANCELLED|COMPLETING|COMPLETED|FAILED|TIMEOUT) ]]; then
        exit 143
    fi
    # fsync a current-generation intent marker before scontrol's TERM teardown.
    "$PYTHON_EXECUTABLE" - "$REQUEUE_CALLING" "$REQUEUE_TARGET" "${SLURM_RESTART_COUNT:-0}" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump({"job_id": sys.argv[2], "restart_count": int(sys.argv[3]),
               "status": "stage_scontrol_requeue_calling", "unix_time": time.time()},
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
exit "$stage_status"
