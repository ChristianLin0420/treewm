#!/bin/bash
#SBATCH --job-name=rql50-1m
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
#SBATCH --array=0-12%13
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append

set -Eeuo pipefail

# submit.py always invokes sbatch from the repository root and creates logs/
# before submission (Slurm opens --output before this shell starts).
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root with submit.py}"
CAMPAIGN_DIR="$PROJECT_ROOT/experiments/10-rql-reproduction"
PYTHON_EXECUTABLE="${RQL_PYTHON:-python}"
: "${RQL_DATA_ROOT:?set RQL_DATA_ROOT to the preflighted OGBench cache}"
: "${RQL_RUN_ROOT:?set RQL_RUN_ROOT to durable checkpoint storage}"
: "${SLURM_ARRAY_TASK_ID:?formal training must be submitted as a Slurm array}"
: "${SLURM_ARRAY_JOB_ID:?formal training requires a Slurm array master ID}"

ALLOCATION_SHARDS=13
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ALLOCATION_SHARDS )); then
    echo "invalid array task $SLURM_ARRAY_TASK_ID; expected 0-$((ALLOCATION_SHARDS - 1))" >&2
    exit 2
fi
REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
SHARD_STATE_ROOT="$RQL_RUN_ROOT/state/arrays/$SLURM_ARRAY_JOB_ID/task$SLURM_ARRAY_TASK_ID"

cd "$PROJECT_ROOT"
mkdir -p "$RQL_RUN_ROOT" "$SHARD_STATE_ROOT"

# One allocation-wide deadline, inherited by every persistent dispatcher.
# 13,200 s (3h40m) leaves 13 minutes before Slurm's USR1 and 20 minutes before
# hard termination, including the <=240 s ready-rank rendezvous.
export RQL_ALLOCATION_STOP_EPOCH="$(( $(date +%s) + 13200 ))"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# Downloads are intentionally not performed inside a billed four-hour GPU
# allocation.  This single-process check fails before the 16 ranks start.
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/prepare_data.py" \
    --manifest "$CAMPAIGN_DIR/manifest.json" \
    --data-root "$RQL_DATA_ROOT" \
    --check

# Requeues may land on different nodes.  Always prove that this allocation has
# exactly two hosts with eight ranks each and one working JAX GPU per rank.
# This intentionally performs only a tiny device operation, not an RQL compile.
GPU_QUICK_DIR="$RQL_RUN_ROOT/state/gpu_preflight/quick/job$SLURM_JOB_ID.restart${SLURM_RESTART_COUNT:-0}"
srun \
    --ntasks=16 \
    --ntasks-per-node=8 \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --cpus-per-task=12 \
    --kill-on-bad-exit=1 \
    "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/gpu_preflight.py" \
        --output-dir "$GPU_QUICK_DIR" \
        --upstream-dir "$CAMPAIGN_DIR/upstream_rql" \
        --quick
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/gpu_preflight.py" \
    --output-dir "$GPU_QUICK_DIR" \
    --verify \
    --verify-level quick \
    --workers 16

# The expensive compile/EGL/checkpoint/auth gate is cached durably by protocol,
# source, Python, and package-version fingerprint.  Requeues do not repeat it
# unless something relevant changed; the current-allocation probe above never
# uses this cache.
GPU_PREFLIGHT_KEY="$(
    "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/gpu_preflight.py" \
        --print-cache-key \
        --protocol-lock "$CAMPAIGN_DIR/protocol.sha256" \
        --upstream-dir "$CAMPAIGN_DIR/upstream_rql"
)"
GPU_PREFLIGHT_CACHE="$RQL_RUN_ROOT/state/gpu_preflight/cache/$GPU_PREFLIGHT_KEY"
GPU_PREFLIGHT_SUCCESS="$GPU_PREFLIGHT_CACHE/SUCCESS.json"
mkdir -p "$GPU_PREFLIGHT_CACHE"
# Array elements may start together.  Serialize only the expensive first-run
# smoke so one 16-GPU allocation produces the shared fingerprinted sentinel;
# every allocation already passed its own quick binding/JAX/topology gate.
exec 9>"$GPU_PREFLIGHT_CACHE/.full-preflight.lock"
flock 9
if [[ ! -s "$GPU_PREFLIGHT_SUCCESS" ]]; then
    GPU_PREFLIGHT_DIR="$GPU_PREFLIGHT_CACHE/job$SLURM_JOB_ID.restart${SLURM_RESTART_COUNT:-0}"
    srun \
        --ntasks=16 \
        --ntasks-per-node=8 \
        --gpus-per-task=1 \
        --gpu-bind=single:1 \
        --cpus-per-task=12 \
        --kill-on-bad-exit=1 \
        "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/gpu_preflight.py" \
            --output-dir "$GPU_PREFLIGHT_DIR" \
            --upstream-dir "$CAMPAIGN_DIR/upstream_rql"
    "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/gpu_preflight.py" \
        --output-dir "$GPU_PREFLIGHT_DIR" \
        --verify \
        --workers 16 \
        --cache-key-value "$GPU_PREFLIGHT_KEY" \
        --success-sentinel "$GPU_PREFLIGHT_SUCCESS"
else
    echo "Reusing successful protocol/code/package-keyed GPU preflight $GPU_PREFLIGHT_KEY"
fi
flock -u 9
exec 9>&-

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
        --upstream-main "$CAMPAIGN_DIR/upstream_rql/main.py" \
        --run-root "$RQL_RUN_ROOT" \
        --data-root "$RQL_DATA_ROOT" \
        --state-root "$SHARD_STATE_ROOT" \
        --job-id "$REQUEUE_TARGET" \
        --workers 16 \
        --allocation-shard "$SLURM_ARRAY_TASK_ID" \
        --allocation-shards "$ALLOCATION_SHARDS"
dispatch_status=$?
set -e

# Rank 0 normally calls `scontrol requeue` from inside the still-live srun
# step after all 16 ready markers exist.  RequeueExit is unset on this cluster,
# so neither --requeue nor exit 75 is treated as an automatic requeue request.
if [[ "$dispatch_status" -ne 0 ]]; then
    exit "$dispatch_status"
fi

# A zero status means this array element's disjoint run set is complete.
# submit.py creates a separate afterok aggregation job that runs only after all
# 13 array elements have completed their exact 1M-step/final-50 contracts.
echo "allocation shard $SLURM_ARRAY_TASK_ID/$ALLOCATION_SHARDS complete"
