#!/bin/bash
#SBATCH --job-name=treewm50-v2-1m
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@420
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append

set -Eeuo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root with submit.py}"
CAMPAIGN_DIR="$PROJECT_ROOT/experiments/12-treewm-formal-v2"
PYTHON_EXECUTABLE="${TREEWM_PYTHON:-/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python}"
: "${TREEWM_DATA_ROOT:?set TREEWM_DATA_ROOT to the immutable OGBench source root}"
: "${TREEWM_CACHE:?set TREEWM_CACHE to the approved read-only raw cache/content root}"
: "${TREEWM_RUN_ROOT:?set TREEWM_RUN_ROOT to fresh v2 formal run storage}"
: "${SLURM_ARRAY_JOB_ID:?formal training requires the three-element array}"
: "${SLURM_ARRAY_TASK_ID:?formal training requires the three-element array}"

ALLOCATION_SHARDS=3
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ALLOCATION_SHARDS )); then
    echo "invalid array task $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
if [[ "${TREEWM_RUN_ROOT,,}" == *v1* ]]; then
    echo "v2 formal training refuses a v1 run namespace" >&2
    exit 2
fi

REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
STATE_BASE="$TREEWM_RUN_ROOT/state/arrays"
printf -v SHARD_PADDED '%02d' "$SLURM_ARRAY_TASK_ID"
ELEMENT_STATE="$STATE_BASE/array-job-$REQUEUE_TARGET/allocation-shard-${SHARD_PADDED}-of-03"
CANCEL_LATCH="$ELEMENT_STATE/CANCEL_REQUESTED"
GENERATION_DIR="$ELEMENT_STATE/requeue/${SLURM_RESTART_COUNT:-0}"
FALLBACK_REQUEUE_CALLING="$GENERATION_DIR/FALLBACK_REQUEUE_CALLING.json"
BATCH_REQUEUE_CALLING="$GENERATION_DIR/BATCH_REQUEUE_CALLING.json"
mkdir -p "$TREEWM_RUN_ROOT" "$ELEMENT_STATE"

cancel_requested=0
dispatcher_step_pid=""
on_cancel() {
    # Successful scontrol teardown sends TERM too; only a marker from this exact
    # generation may suppress conversion into a persistent cancellation latch.
    if [[ -s "$FALLBACK_REQUEUE_CALLING" || -s "$BATCH_REQUEUE_CALLING" ]]; then
        return
    fi
    cancel_requested=1
    touch "$CANCEL_LATCH"
    if [[ -n "$dispatcher_step_pid" ]] && kill -0 "$dispatcher_step_pid" 2>/dev/null; then
        kill -TERM "$dispatcher_step_pid" 2>/dev/null || true
    fi
}
trap on_cancel TERM INT
if [[ -e "$CANCEL_LATCH" ]]; then
    echo "persistent cancellation latch is set for $REQUEUE_TARGET" >&2
    exit 143
fi

cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/submit.py" \
    --verify-source-snapshot --repo-root "$PROJECT_ROOT"
export WANDB_PROJECT="${WANDB_PROJECT:-treewm-50task-formal-v2}"
export WANDB_MODE="${WANDB_MODE:-online}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Stop at 3h40m, thirteen minutes before USR1 and twenty before hard termination.
export TREEWM_ALLOCATION_STOP_EPOCH="$(( $(date +%s) + 13200 ))"
export TREEWM_ALLOCATION_SHARDS="$ALLOCATION_SHARDS"

# Topology and CUDA are host-generation properties, so prove them every restart.
QUICK_DIR="$TREEWM_RUN_ROOT/state/gpu_preflight/quick/job${SLURM_JOB_ID}.restart${SLURM_RESTART_COUNT:-0}"
srun \
    --ntasks=16 \
    --ntasks-per-node=8 \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --cpus-per-task=12 \
    --kill-on-bad-exit=1 \
    "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/gpu_preflight.py" \
        --output-dir "$QUICK_DIR" --repo-root "$PROJECT_ROOT" --quick
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/gpu_preflight.py" \
    --output-dir "$QUICK_DIR" --verify --verify-level quick --workers 16

if (( cancel_requested )) || [[ -e "$CANCEL_LATCH" ]]; then
    exit 143
fi

# The heavy gate is shared only when protocol, objective source, runtime, launch
# scripts, calibrated contracts, and pilot acceptance all have the same fingerprints.
PREFLIGHT_KEY="$(
    "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/gpu_preflight.py" \
        --print-cache-key --protocol-lock "$CAMPAIGN_DIR/protocol.sha256" \
        --repo-root "$PROJECT_ROOT" --cache-root "$TREEWM_CACHE"
)"
PREFLIGHT_CACHE="$TREEWM_RUN_ROOT/state/gpu_preflight/cache/$PREFLIGHT_KEY"
PREFLIGHT_SUCCESS="$PREFLIGHT_CACHE/SUCCESS.json"
mkdir -p "$PREFLIGHT_CACHE"
exec 9>"$PREFLIGHT_CACHE/.full-preflight.lock"
flock 9
if ! "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/gpu_preflight.py" \
    --check-success-sentinel --success-sentinel "$PREFLIGHT_SUCCESS" \
    --cache-key-value "$PREFLIGHT_KEY" --workers 16; then
    FULL_DIR="$PREFLIGHT_CACHE/job${SLURM_JOB_ID}.restart${SLURM_RESTART_COUNT:-0}"
    srun \
        --ntasks=16 \
        --ntasks-per-node=8 \
        --gpus-per-task=1 \
        --gpu-bind=single:1 \
        --cpus-per-task=12 \
        --kill-on-bad-exit=1 \
        "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/gpu_preflight.py" \
            --output-dir "$FULL_DIR" --repo-root "$PROJECT_ROOT" \
            --manifest "$CAMPAIGN_DIR/manifest.json" \
            --data-root "$TREEWM_DATA_ROOT" --cache-root "$TREEWM_CACHE"
    "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/gpu_preflight.py" \
        --output-dir "$FULL_DIR" --verify --workers 16 \
        --cache-key-value "$PREFLIGHT_KEY" --success-sentinel "$PREFLIGHT_SUCCESS"
else
    echo "reusing exact TreeWM-v2 heavy preflight $PREFLIGHT_KEY"
fi
flock -u 9
exec 9>&-

if (( cancel_requested )) || [[ -e "$CANCEL_LATCH" ]]; then
    exit 143
fi

# Background srun leaves the batch trap live while every trainer commits its exact
# model/optimizer/scheduler/loader/RNG checkpoint.
set +e
srun \
    --ntasks=16 \
    --ntasks-per-node=8 \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --cpus-per-task=12 \
    --kill-on-bad-exit=0 \
    --label \
    "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/dispatcher.py" \
        --manifest "$CAMPAIGN_DIR/manifest.json" \
        --repo-root "$PROJECT_ROOT" \
        --run-root "$TREEWM_RUN_ROOT" \
        --data-root "$TREEWM_DATA_ROOT" \
        --cache-root "$TREEWM_CACHE" \
        --state-root "$STATE_BASE" \
        --cancel-latch "$CANCEL_LATCH" \
        --job-id "$REQUEUE_TARGET" \
        --workers 16 \
        --allocation-shard "$SLURM_ARRAY_TASK_ID" \
        --allocation-shards "$ALLOCATION_SHARDS" &
dispatcher_step_pid=$!
wait "$dispatcher_step_pid"
dispatch_status=$?
set -e
dispatcher_step_pid=""

if (( cancel_requested )) || [[ -e "$CANCEL_LATCH" ]]; then
    exit 143
fi
resolution="$("$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/dispatcher.py" \
    --resolution-status --manifest "$CAMPAIGN_DIR/manifest.json" \
    --repo-root "$PROJECT_ROOT" --run-root "$TREEWM_RUN_ROOT" \
    --data-root "$TREEWM_DATA_ROOT" --cache-root "$TREEWM_CACHE" \
    --state-root "$STATE_BASE" --cancel-latch "$CANCEL_LATCH" \
    --job-id "$REQUEUE_TARGET" --restart-count "${SLURM_RESTART_COUNT:-0}" \
    --worker-index 0 --workers 16 --allocation-shard "$SLURM_ARRAY_TASK_ID" \
    --allocation-shards "$ALLOCATION_SHARDS")"

if [[ "$resolution" == "cancelled" ]]; then
    exit 143
fi
if [[ "$resolution" == "called" ]]; then
    exit 75
fi
if [[ "$resolution" == "authorized" ]]; then
    # Briefly expose a racing scancel before the irreversible scheduler request.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        if (( cancel_requested )) || [[ -e "$CANCEL_LATCH" ]]; then
            exit 143
        fi
        sleep 0.2
    done
    job_record="$(scontrol show job "$REQUEUE_TARGET" --oneliner 2>&1)" || {
        echo "cannot verify $REQUEUE_TARGET before requeue: $job_record" >&2
        exit 2
    }
    if [[ "$job_record" =~ JobState=(CANCELLED|COMPLETING|COMPLETED|FAILED|TIMEOUT) ]]; then
        echo "scheduler state forbids requeue: $job_record" >&2
        exit 143
    fi
    "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/dispatcher.py" \
        --mark-batch-requeue-calling --manifest "$CAMPAIGN_DIR/manifest.json" \
        --repo-root "$PROJECT_ROOT" --run-root "$TREEWM_RUN_ROOT" \
        --data-root "$TREEWM_DATA_ROOT" --cache-root "$TREEWM_CACHE" \
        --state-root "$STATE_BASE" --cancel-latch "$CANCEL_LATCH" \
        --job-id "$REQUEUE_TARGET" --restart-count "${SLURM_RESTART_COUNT:-0}" \
        --worker-index 0 --workers 16 --allocation-shard "$SLURM_ARRAY_TASK_ID" \
        --allocation-shards "$ALLOCATION_SHARDS"
    if ! scontrol requeue "$REQUEUE_TARGET"; then
        rm -f "$BATCH_REQUEUE_CALLING"
        exit 2
    fi
    exit 75
fi
if [[ "$dispatch_status" -eq 75 ]]; then
    echo "exit 75 lacked a valid current-generation authorization" >&2
    exit 2
fi
exit "$dispatch_status"
