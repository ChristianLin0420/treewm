#!/bin/bash
#SBATCH --job-name=treewm-grounded-gauge-all-ten-bridge-v2
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
#SBATCH --array=0-19%20
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --open-mode=append

set -Eeuo pipefail

PROJECT_ROOT="${TREEWM_GAUGE_BRIDGE_REPO_ROOT:?launch only through guarded Exp21 submit.py}"
CAMPAIGN_DIR="$PROJECT_ROOT/experiments/21-treewm-grounded-gauge-all-ten-bridge-v2"
PYTHON_EXECUTABLE=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
SRUN=/cm/shared/apps/slurm/current/bin/srun
SCONTROL=/cm/shared/apps/slurm/current/bin/scontrol
: "${SLURM_ARRAY_JOB_ID:?Exp21 requires an array job}"
: "${SLURM_ARRAY_TASK_ID:?Exp21 requires an array task}"
: "${TREEWM_EXPECTED_SOURCE_SHA256:?missing source seal}"
: "${TREEWM_EXPECTED_RUNTIME_SHA256:?missing runtime seal}"
: "${TREEWM_EXPECTED_PACKAGE_PROTOCOL_SHA256:?missing protocol seal}"
: "${TREEWM_EXPECTED_EXP20_BINDING_SHA256:?missing Exp20 binding seal}"
: "${TREEWM_EXPECTED_SELECTED_RECIPE_SHA256:?missing selected recipe seal}"
: "${TREEWM_EXPECTED_STAGE_TARGET:?missing lifecycle target}"
[[ "$TREEWM_EXPECTED_STAGE_TARGET" == "25000" ]] || exit 2
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 20 )) || exit 2
if [[ ! -x "$PYTHON_EXECUTABLE" || ! -x "$SRUN" || ! -x "$SCONTROL" ]]; then
    echo "pinned Python/Slurm executable is unavailable" >&2
    exit 2
fi

REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RUN_ROOT=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/treewm/outputs/treewm-grounded-gauge-all-ten-bridge-v2
TASK_STATE="$RUN_ROOT/state/arrays/stage-25000/array-job-$REQUEUE_TARGET/task-$(printf '%02d' "$SLURM_ARRAY_TASK_ID")"
GENERATION_DIR="$TASK_STATE/requeue/${SLURM_RESTART_COUNT:-0}"
CANCEL_LATCH="$TASK_STATE/CANCEL_REQUESTED"
REQUEUE_LATCH="$TASK_STATE/REQUEUE_REQUESTED.json"
READY_PATH="$TASK_STATE/READY_FOR_REQUEUE.json"
WORKER_COMPLETE_PATH="$TASK_STATE/WORKER_COMPLETE.json"
CANCELLED_PATH="$TASK_STATE/CANCELLED.json"
REQUEUE_CALLING="$GENERATION_DIR/REQUEUE_CALLING.json"
mkdir -p "$GENERATION_DIR"
[[ ! -e "$READY_PATH" ]] || mv "$READY_PATH" "$GENERATION_DIR/READY_FROM_PREVIOUS_GENERATION.json"
[[ ! -e "$REQUEUE_LATCH" ]] || mv "$REQUEUE_LATCH" "$GENERATION_DIR/REQUEST_FROM_PREVIOUS_GENERATION.json"

on_usr1() {
    [[ -e "$CANCEL_LATCH" ]] || touch "$REQUEUE_LATCH"
}
on_cancel() {
    [[ ! -s "$REQUEUE_CALLING" ]] || return
    mkdir -p "$TASK_STATE"
    touch "$CANCEL_LATCH"
}
trap on_usr1 USR1
trap on_cancel TERM INT

cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export TREEWM_REQUEUE_CALLING_MARKER="$REQUEUE_CALLING"
export PATH="/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin"
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/campaign.py" snapshot --repo-root "$PROJECT_ROOT"

set +e
"$SRUN" --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 --cpus-per-task=12 \
    --kill-on-bad-exit=0 \
    "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/worker.py" \
    --index "$SLURM_ARRAY_TASK_ID" --stage-target 25000 \
    --state-dir "$TASK_STATE" --repo-root "$PROJECT_ROOT" \
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
set -e

if [[ "$status" -eq 0 ]]; then
    [[ -s "$WORKER_COMPLETE_PATH" ]] && exit 0
    echo "worker returned success without durable completion" >&2
    exit 2
fi
if [[ -e "$CANCEL_LATCH" || "$status" -eq 143 ]]; then
    if [[ ! -s "$CANCELLED_PATH" ]]; then
        set +e
        "$SRUN" --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 --cpus-per-task=12 \
            --kill-on-bad-exit=0 \
            "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/worker.py" \
            --index "$SLURM_ARRAY_TASK_ID" --stage-target 25000 \
            --state-dir "$TASK_STATE" --repo-root "$PROJECT_ROOT" \
            --manifest "$CAMPAIGN_DIR/manifest.json"
        set -e
    fi
    [[ -s "$CANCELLED_PATH" ]] || { echo "cancellation lacks durable remote-worker CANCELLED.json" >&2; exit 2; }
    exit 143
fi
if [[ "$status" -ne 75 || ! -s "$READY_PATH" ]]; then
    echo "worker failed without a verified exact-resume checkpoint (status=$status)" >&2
    exit "$status"
fi

job_record="$($SCONTROL show job "$REQUEUE_TARGET" --oneliner 2>&1)" || exit 2
[[ ! "$job_record" =~ JobState=(CANCELLED|COMPLETING|COMPLETED|FAILED|TIMEOUT) ]] || exit 143
[[ ! -e "$CANCEL_LATCH" ]] || { echo "cancellation raced with requeue" >&2; exit 2; }
printf '{"status":"scontrol_requeue_calling","job_id":"%s","restart_count":%s}\n' \
    "$REQUEUE_TARGET" "${SLURM_RESTART_COUNT:-0}" > "$REQUEUE_CALLING"
if ! "$SCONTROL" requeue "$REQUEUE_TARGET"; then
    mv "$REQUEUE_CALLING" "$GENERATION_DIR/REQUEUE_FAILED.json"
    exit 2
fi
exit 75
