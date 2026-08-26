#!/bin/bash
#SBATCH --job-name=treewm-grounded-final-eval-v1
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
#SBATCH --array=0-199%40
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --open-mode=append

set -Eeuo pipefail

PROJECT_ROOT="${TREEWM_FORMAL_REPO_ROOT:?launch only through guarded submit.py from a sealed snapshot}"
CAMPAIGN_DIR="$PROJECT_ROOT/experiments/14-treewm-grounded-formal-v1"
PYTHON_EXECUTABLE=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
SRUN=/cm/shared/apps/slurm/current/bin/srun
SCONTROL=/cm/shared/apps/slurm/current/bin/scontrol
: "${SLURM_ARRAY_JOB_ID:?formal final evaluation requires the 200-element array}"
: "${SLURM_ARRAY_TASK_ID:?formal final evaluation requires the 200-element array}"
: "${TREEWM_EXPECTED_SOURCE_SHA256:?missing sealed source identity}"
: "${TREEWM_EXPECTED_EVALUATION_SOURCE_SHA256:?missing sealed evaluation source identity}"
: "${TREEWM_EXPECTED_RUNTIME_SHA256:?missing sealed runtime identity}"
: "${TREEWM_EXPECTED_PACKAGE_PROTOCOL_SHA256:?missing sealed protocol identity}"
: "${TREEWM_EXPECTED_SEED_TABLE_SHA256:?missing sealed evaluation seed table}"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= 200 )); then
    exit 2
fi
if [[ ! -x "$PYTHON_EXECUTABLE" || ! -x "$SRUN" || ! -x "$SCONTROL" ]]; then
    echo "pinned Python/Slurm executable is unavailable" >&2
    exit 2
fi

REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
EVAL_ROOT=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/treewm/outputs/treewm-grounded-formal-v1-final-eval
TASK_STATE="$EVAL_ROOT/state/arrays/array-job-$REQUEUE_TARGET/task-$(printf '%03d' "$SLURM_ARRAY_TASK_ID")"
GENERATION_DIR="$TASK_STATE/requeue/${SLURM_RESTART_COUNT:-0}"
CANCEL_LATCH="$TASK_STATE/CANCEL_REQUESTED"
READY_PATH="$TASK_STATE/READY_FOR_REQUEUE.json"
REQUEUE_CALLING="$GENERATION_DIR/REQUEUE_CALLING.json"
PROGRESS_PATH="$EVAL_ROOT/results/$(printf '%03d' "$SLURM_ARRAY_TASK_ID").progress.json"
RESULT_PATH="$EVAL_ROOT/results/$(printf '%03d' "$SLURM_ARRAY_TASK_ID").json"
mkdir -p "$GENERATION_DIR"
if [[ -e "$READY_PATH" ]]; then
    mv "$READY_PATH" "$GENERATION_DIR/READY_FROM_PREVIOUS_GENERATION.json"
fi

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
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PATH="/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin"
export TREEWM_EVAL_REQUEUE_READY="$READY_PATH"
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/campaign.py" snapshot --repo-root "$PROJECT_ROOT"

set +e
"$SRUN" --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 --cpus-per-task=12 \
    --kill-on-bad-exit=0 \
    "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/final_eval.py" \
    --index "$SLURM_ARRAY_TASK_ID" \
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

if [[ -e "$CANCEL_LATCH" || "$status" -eq 143 ]]; then
    exit 143
fi
if [[ "$status" -eq 0 ]]; then
    if [[ -s "$RESULT_PATH" ]]; then
        exit 0
    fi
    echo "final evaluator returned success without a durable result" >&2
    exit 2
fi
if [[ "$status" -ne 75 || ! -s "$PROGRESS_PATH" || ! -s "$READY_PATH" || -e "$RESULT_PATH" ]]; then
    echo "final evaluator failed without resumable exact progress (status=$status)" >&2
    exit "$status"
fi
job_record="$($SCONTROL show job "$REQUEUE_TARGET" --oneliner 2>&1)" || exit 2
if [[ "$job_record" =~ JobState=(CANCELLED|COMPLETING|COMPLETED|FAILED|TIMEOUT) ]]; then
    exit 143
fi
if [[ -e "$CANCEL_LATCH" ]]; then
    exit 143
fi
printf '{"status":"scontrol_requeue_calling","job_id":"%s","restart_count":%s}\n' \
    "$REQUEUE_TARGET" "${SLURM_RESTART_COUNT:-0}" > "$REQUEUE_CALLING"
if ! "$SCONTROL" requeue "$REQUEUE_TARGET"; then
    mv "$REQUEUE_CALLING" "$GENERATION_DIR/REQUEUE_FAILED.json"
    exit 2
fi
exit 75
