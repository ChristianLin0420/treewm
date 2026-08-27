#!/bin/bash
#SBATCH --job-name=treewm-support-threshold-v1
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-7%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --open-mode=append

set -Eeuo pipefail

SOURCE_ROOT="${TREEWM_SOURCE_ROOT:-${SLURM_SUBMIT_DIR:?submit from a TreeWM source root}}"
PROJECT_ROOT="${TREEWM_PROJECT_ROOT:-/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/treewm}"
PACKAGE="$SOURCE_ROOT/experiments/diagnostics/treewm-support-threshold-v1"
PYTHON_EXECUTABLE=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
SRUN=/cm/shared/apps/slurm/current/bin/srun
OUTPUT_ROOT="$PROJECT_ROOT/outputs/treewm-support-threshold-diagnostic-v1"

: "${SLURM_ARRAY_TASK_ID:?the diagnostic requires its 8-element GPU array}"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= 8 )); then
    echo "array index must be in [0, 8)" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_EXECUTABLE" || ! -x "$SRUN" ]]; then
    echo "pinned Python or compute-node Slurm client is unavailable" >&2
    exit 2
fi

if (( SLURM_ARRAY_TASK_ID < 4 )); then
    setting=puzzle-3x3
    seed="$SLURM_ARRAY_TASK_ID"
else
    setting=puzzle-4x4-100m
    seed="$((SLURM_ARRAY_TASK_ID - 4))"
fi
run_name="grounded-formal-${setting}-seed${seed}"
checkpoint="$PROJECT_ROOT/outputs/treewm-grounded-formal-v1/$setting/treewm/$run_name/checkpoints/latest.pt"
if [[ ! -f "$checkpoint" ]]; then
    echo "paused formal checkpoint is unavailable: $checkpoint" >&2
    exit 2
fi

if [[ ! -f "$PACKAGE/support_threshold.py" || ! -d "$PROJECT_ROOT/outputs/treewm-grounded-formal-v1" ]]; then
    echo "diagnostic source or paused formal output root is unavailable" >&2
    exit 2
fi

cd "$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export WANDB_MODE=disabled
export PATH="/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin"

exec "$SRUN" --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 --cpus-per-task=4 \
    --kill-on-bad-exit=1 \
    "$PYTHON_EXECUTABLE" "$PACKAGE/support_threshold.py" \
    --checkpoint "$checkpoint" \
    --output-root "$OUTPUT_ROOT" \
    --sample-size 4096 \
    --batch-size 256 \
    --stage-target 25000 \
    --device cuda
