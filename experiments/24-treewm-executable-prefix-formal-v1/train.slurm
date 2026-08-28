#!/bin/bash
#SBATCH --job-name=treewm-exp24-formal-stage
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
#SBATCH --export=NONE
#SBATCH --open-mode=append

set -Eeuo pipefail
umask 077

PYTHON_EXECUTABLE=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

if [[ "$#" -ne 5 ]]; then
    echo "train.slurm requires snapshot_root submission_root submission_sha256 node stage_target" >&2
    exit 2
fi
SNAPSHOT_ROOT=$1
SUBMISSION_ROOT=$2
SUBMISSION_SHA256=$3
NODE=$4
STAGE_TARGET=$5
case "$SNAPSHOT_ROOT" in /*) ;; *) echo "snapshot root must be absolute" >&2; exit 2 ;; esac
case "$SUBMISSION_ROOT" in /*) ;; *) echo "submission root must be absolute" >&2; exit 2 ;; esac
[[ "$SUBMISSION_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "submission SHA256 is malformed" >&2; exit 2; }
[[ "$NODE" == "train_$STAGE_TARGET" ]] || { echo "stage node identity differs" >&2; exit 2; }
case "$STAGE_TARGET" in 2000|25000|100000|1000000) ;; *) echo "stage target differs" >&2; exit 2 ;; esac
[[ -d "$SNAPSHOT_ROOT" && ! -L "$SNAPSHOT_ROOT" ]] || { echo "snapshot root differs" >&2; exit 2; }
[[ -d "$SUBMISSION_ROOT" && ! -L "$SUBMISSION_ROOT" ]] || { echo "submission root differs" >&2; exit 2; }
[[ -x "$PYTHON_EXECUTABLE" ]] || { echo "pinned Python unavailable" >&2; exit 2; }
: "${SLURM_ARRAY_JOB_ID:?Exp24 stage requires an array job}"
: "${SLURM_ARRAY_TASK_ID:?Exp24 stage requires an array task}"
[[ "$SLURM_ARRAY_JOB_ID" =~ ^[1-9][0-9]*$ ]] || { echo "array job ID differs" >&2; exit 2; }
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || { echo "array task ID differs" >&2; exit 2; }
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 40 )) || { echo "array task outside 40-run matrix" >&2; exit 2; }
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
[[ "$RESTART_COUNT" =~ ^[0-9]+$ ]] || { echo "restart count differs" >&2; exit 2; }

WORKER="$SNAPSHOT_ROOT/experiments/24-treewm-executable-prefix-formal-v1/worker.py"
[[ -f "$WORKER" && ! -L "$WORKER" ]] || { echo "snapshot worker differs" >&2; exit 2; }
COMMON=(--snapshot-root "$SNAPSHOT_ROOT" --submission-root "$SUBMISSION_ROOT" --submission-sha256 "$SUBMISSION_SHA256" --node "$NODE" --cell-index "$SLURM_ARRAY_TASK_ID" --restart-count "$RESTART_COUNT" --array-job-id "$SLURM_ARRAY_JOB_ID" --array-task-id "$SLURM_ARRAY_TASK_ID")
worker_pid=

forward_signal() {
    local requested=$1
    "$PYTHON_EXECUTABLE" -I -S -B "$WORKER" record-signal "${COMMON[@]}" --signal "$requested" || return
    if [[ -n "$worker_pid" ]]; then
        kill -"$requested" "$worker_pid" 2>/dev/null || true
    fi
}
trap 'forward_signal USR1' USR1
trap 'forward_signal TERM' TERM INT

cd "$SNAPSHOT_ROOT"
(
    # Do not inherit the wrapper's trapped dispositions across exec: the
    # parent forwards the scheduler signal to this exact worker PID.
    trap - USR1 TERM INT
    exec "$PYTHON_EXECUTABLE" -I -S -B "$WORKER" run "${COMMON[@]}"
) &
worker_pid=$!
set +e
while true; do
    wait "$worker_pid"
    status=$?
    kill -0 "$worker_pid" 2>/dev/null || break
done
set -e
worker_pid=
if [[ "$status" -eq 0 || "$status" -eq 76 ]]; then
    exit "$status"
fi
if [[ "$status" -ne 75 ]]; then
    exit "$status"
fi
exec "$PYTHON_EXECUTABLE" -I -S -B "$WORKER" requeue "${COMMON[@]}"
