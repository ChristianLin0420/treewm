#!/bin/bash
#SBATCH --job-name=treewm-executable-prefix-pilot-v1-launch8
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@420
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --array=0-19%20
#SBATCH --export=NONE
#SBATCH --open-mode=append

set -Eeuo pipefail
umask 077

PYTHON_EXECUTABLE=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
export PATH=/usr/bin:/bin

if [[ "$#" -ne 5 ]]; then
    echo "usage: train.slurm SNAPSHOT_ROOT SUBMISSION_ROOT SUBMISSION_SHA256 WAVE_INDEX PREDECESSOR_ARRAY_JOB_ID" >&2
    exit 2
fi
SNAPSHOT_ROOT=$1
SUBMISSION_ROOT=$2
SUBMISSION_SHA256=$3
WAVE_INDEX=$4
PREDECESSOR_ARRAY_JOB_ID=$5
case "$SNAPSHOT_ROOT" in /*) ;; *) echo "snapshot root must be absolute" >&2; exit 2 ;; esac
case "$SUBMISSION_ROOT" in /*) ;; *) echo "submission root must be absolute" >&2; exit 2 ;; esac
[[ "$SUBMISSION_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "submission SHA256 is malformed" >&2; exit 2; }
[[ "$WAVE_INDEX" == 0 || "$WAVE_INDEX" == 1 ]] || { echo "wave index must be 0 or 1" >&2; exit 2; }
if [[ "$WAVE_INDEX" == 0 ]]; then
    [[ "$PREDECESSOR_ARRAY_JOB_ID" == none ]] || { echo "wave zero must not name a predecessor" >&2; exit 2; }
else
    [[ "$PREDECESSOR_ARRAY_JOB_ID" =~ ^[0-9]+$ ]] || { echo "wave one predecessor job ID is malformed" >&2; exit 2; }
fi
[[ -d "$SNAPSHOT_ROOT" && ! -L "$SNAPSHOT_ROOT" ]] || { echo "snapshot root is unavailable or symlinked" >&2; exit 2; }
[[ -d "$SUBMISSION_ROOT" && ! -L "$SUBMISSION_ROOT" ]] || { echo "submission root is unavailable or symlinked" >&2; exit 2; }
[[ -f "$SUBMISSION_ROOT/SUBMISSION_CONTRACT.json" && ! -L "$SUBMISSION_ROOT/SUBMISSION_CONTRACT.json" ]] || {
    echo "submission contract is unavailable or symlinked" >&2
    exit 2
}
[[ -x "$PYTHON_EXECUTABLE" ]] || {
    echo "pinned Python executable is unavailable" >&2
    exit 2
}
[[ -z "${TREEWM_STOP_AFTER_UPDATE+x}" ]] || {
    echo "TREEWM_STOP_AFTER_UPDATE is forbidden" >&2
    exit 2
}

: "${SLURM_ARRAY_JOB_ID:?Exp23 requires a Slurm array job}"
: "${SLURM_ARRAY_TASK_ID:?Exp23 requires a Slurm array task}"
[[ "$SLURM_ARRAY_JOB_ID" =~ ^[0-9]+$ ]] || { echo "array job ID is malformed" >&2; exit 2; }
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || { echo "array task ID is malformed" >&2; exit 2; }
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 20 )) || {
    echo "array task is outside the exact 20-cell matrix" >&2
    exit 2
}
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
[[ "$RESTART_COUNT" == 0 ]] || { echo "within-wave scheduler requeue is forbidden" >&2; exit 2; }
if [[ "$WAVE_INDEX" == 1 && "$SLURM_ARRAY_JOB_ID" == "$PREDECESSOR_ARRAY_JOB_ID" ]]; then
    echo "wave one must be a distinct array job" >&2
    exit 2
fi

PACKAGE_DIR="$SNAPSHOT_ROOT/experiments/23-treewm-executable-prefix-repair-pilot-v1"
WORKER="$PACKAGE_DIR/worker.py"
[[ -f "$WORKER" && ! -L "$WORKER" ]] || { echo "snapshot worker is unavailable or symlinked" >&2; exit 2; }
CELL_LABEL=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
TASK_ROOT="$SUBMISSION_ROOT/tasks/cell-$CELL_LABEL"
GENERATION_ROOT="$TASK_ROOT/waves/$WAVE_INDEX"
SIGNAL_READY="$GENERATION_ROOT/WORKER_SIGNAL_READY.json"
CONTINUATION_READY="$GENERATION_ROOT/CONTINUATION_READY.json"
WAVE_INCOMPLETE="$GENERATION_ROOT/WAVE_INCOMPLETE.json"
TASK_COMPLETE="$TASK_ROOT/WORKER_COMPLETE.json"
TASK_CANCEL="$TASK_ROOT/CANCEL_REQUESTED.json"
GLOBAL_CANCEL="$SUBMISSION_ROOT/CANCEL_REQUESTED.json"
COMMON_ARGS=(
    --submission-root "$SUBMISSION_ROOT"
    --submission-sha256 "$SUBMISSION_SHA256"
    --cell-index "$SLURM_ARRAY_TASK_ID"
    --wave-index "$WAVE_INDEX"
    --array-job-id "$SLURM_ARRAY_JOB_ID"
    --array-task-id "$SLURM_ARRAY_TASK_ID"
    --predecessor-array-job-id "$PREDECESSOR_ARRAY_JOB_ID"
)

worker_pid=
signal_record_failed=0

cancel_is_latched() {
    [[ -e "$GLOBAL_CANCEL" || -L "$GLOBAL_CANCEL" || -e "$TASK_CANCEL" || -L "$TASK_CANCEL" ]]
}

record_and_forward() {
    local requested_signal=$1
    local status=0
    if "$PYTHON_EXECUTABLE" -I -S -B "$WORKER" record-signal "${COMMON_ARGS[@]}" --signal "$requested_signal"; then
        status=0
    else
        status=$?
    fi
    if [[ "$status" -ne 0 ]]; then
        signal_record_failed=1
        echo "failed to durably record $requested_signal (status=$status)" >&2
        return
    fi
    if [[ -n "$worker_pid" && -f "$SIGNAL_READY" && ! -L "$SIGNAL_READY" ]]; then
        if [[ "$requested_signal" == TERM ]]; then
            kill -TERM "$worker_pid" 2>/dev/null || true
        else
            kill -USR1 "$worker_pid" 2>/dev/null || true
        fi
    fi
}

on_usr1() {
    cancel_is_latched || record_and_forward USR1
}

on_term() {
    record_and_forward TERM
}

trap on_usr1 USR1
trap on_term TERM INT

cd "$SNAPSHOT_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

# Ignore lifecycle signals only across Python's exec/import window. worker.py installs
# handlers before validating the full context and train_entry uses an inherited blocked
# mask for its own exec-to-handler bridge.
(
    trap '' USR1 TERM INT
    exec "$PYTHON_EXECUTABLE" -I -S -B "$WORKER" run \
        --snapshot-root "$SNAPSHOT_ROOT" \
        "${COMMON_ARGS[@]}"
) &
worker_pid=$!

set +e
while true; do
    wait "$worker_pid"
    worker_status=$?
    if kill -0 "$worker_pid" 2>/dev/null; then
        continue
    fi
    break
done
set -e
worker_pid=

if [[ "$worker_status" -eq 0 ]]; then
    [[ -f "$TASK_COMPLETE" && ! -L "$TASK_COMPLETE" ]] || {
        echo "worker returned zero without immutable WORKER_COMPLETE.json" >&2
        exit 2
    }
    exit 0
fi

if [[ "$worker_status" -eq 76 ]]; then
    exit 76
fi

if [[ "$signal_record_failed" -ne 0 ]]; then
    echo "a lifecycle signal was not durably recorded" >&2
    exit 2
fi

if [[ "$WAVE_INDEX" != 0 || "$worker_status" -ne 75 || ! -f "$CONTINUATION_READY" || -L "$CONTINUATION_READY" ]]; then
    if [[ "$WAVE_INDEX" == 1 && -f "$WAVE_INCOMPLETE" && ! -L "$WAVE_INCOMPLETE" ]]; then
        echo "wave one exhausted incomplete; the fixed DAG has no successor" >&2
        exit 2
    fi
    echo "worker failed without valid wave-zero continuation readiness (status=$worker_status)" >&2
    exit "$worker_status"
fi
if cancel_is_latched; then
    echo "cancellation raced with wave-zero continuation readiness" >&2
    exit 76
fi
# The controller predeclared wave one before releasing this array.  Compute-side
# execution never contacts the scheduler; exit zero satisfies the afterok edge.
exit 0
