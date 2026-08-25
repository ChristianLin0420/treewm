#!/bin/bash
#SBATCH --job-name=treewm-data
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

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root with submit.py}"
CAMPAIGN_DIR="$PROJECT_ROOT/experiments/11-treewm-formal"
PYTHON_EXECUTABLE="${TREEWM_PYTHON:-/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python}"
: "${TREEWM_DATA_ROOT:?set TREEWM_DATA_ROOT to the immutable OGBench source root}"
: "${TREEWM_CACHE:?set TREEWM_CACHE to durable cache storage}"
: "${SLURM_ARRAY_JOB_ID:?cache staging must be a ten-element array}"
: "${SLURM_ARRAY_TASK_ID:?cache staging must be a ten-element array}"

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= 10 )); then
    echo "invalid cache setting index $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
STAGE_STATE="$TREEWM_CACHE/stage-state/array-$SLURM_ARRAY_JOB_ID/task-$SLURM_ARRAY_TASK_ID"
CANCEL_LATCH="$STAGE_STATE/CANCEL_REQUESTED"
STAGE_GENERATION="$STAGE_STATE/requeue/${SLURM_RESTART_COUNT:-0}"
REQUEUE_CALLING="$STAGE_GENERATION/REQUEUE_CALLING.json"
mkdir -p "$STAGE_STATE" "$STAGE_GENERATION"

cancel_requested=0
stage_step_pid=""
on_cancel() {
    # The scheduler tears down a successful explicit requeue with TERM. This
    # current-generation marker distinguishes that teardown from scancel.
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
    echo "cache-stage cancellation latch is set; refusing restart" >&2
    exit 143
fi

cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

# prepare_cache handles USR1/TERM only at safe source/cache boundaries and returns
# 75 after its durable progress is committed. Run srun in the background so a
# concurrent scancel can latch no-requeue in this batch shell.
set +e
srun \
    --ntasks=1 \
    --cpus-per-task=12 \
    --kill-on-bad-exit=0 \
    "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/prepare_cache.py" \
        --manifest "$CAMPAIGN_DIR/manifest.json" \
        --setting-index "$SLURM_ARRAY_TASK_ID" \
        --data-root "$TREEWM_DATA_ROOT" \
        --cache-root "$TREEWM_CACHE" &
stage_step_pid=$!
stage_status=0
while true; do
    wait "$stage_step_pid"
    stage_status=$?
    if ! kill -0 "$stage_step_pid" 2>/dev/null; then
        break
    fi
done
set -e
stage_step_pid=""

if (( cancel_requested )) || [[ -e "$CANCEL_LATCH" ]]; then
    echo "cache-stage cancellation observed; no requeue" >&2
    exit 143
fi
if [[ "$stage_status" -eq 75 ]]; then
    for _ in 1 2 3 4 5; do
        if (( cancel_requested )) || [[ -e "$CANCEL_LATCH" ]]; then
            exit 143
        fi
        sleep 0.2
    done
    job_record="$(scontrol show job "$REQUEUE_TARGET" --oneliner 2>&1)" || {
        echo "cannot verify cache element before requeue: $job_record" >&2
        exit 2
    }
    if [[ "$job_record" =~ JobState=(CANCELLED|COMPLETING|COMPLETED|FAILED|TIMEOUT) ]]; then
        echo "cache element state forbids requeue: $job_record" >&2
        exit 143
    fi
    if (( cancel_requested )) || [[ -e "$CANCEL_LATCH" ]]; then
        exit 143
    fi
    # Commit intent before scontrol: its successful TERM teardown must not create
    # the generation-independent cancellation latch seen by the successor.
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
    json.dump(
        {
            "job_id": sys.argv[2],
            "restart_count": int(sys.argv[3]),
            "status": "stage_scontrol_requeue_calling",
            "unix_time": time.time(),
        },
        handle,
        sort_keys=True,
    )
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
    if ! scontrol requeue "$REQUEUE_TARGET"; then
        rm -f "$REQUEUE_CALLING"
        echo "explicit cache-stage requeue failed for $REQUEUE_TARGET" >&2
        exit 2
    fi
    exit 75
fi
exit "$stage_status"
