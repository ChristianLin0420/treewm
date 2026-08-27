#!/bin/bash
#SBATCH --job-name=treewm-gradient-audit-v1
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-9%10
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --open-mode=append

set -Eeuo pipefail

SOURCE_ROOT="${TREEWM_SOURCE_ROOT:-${SLURM_SUBMIT_DIR:?submit from a TreeWM source root}}"
PROJECT_ROOT="${TREEWM_PROJECT_ROOT:-/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/treewm}"
PACKAGE="$SOURCE_ROOT/experiments/diagnostics/treewm-gradient-audit-v1"
PYTHON_EXECUTABLE=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
SRUN=/cm/shared/apps/slurm/current/bin/srun
OUTPUT_ROOT="$PROJECT_ROOT/outputs/treewm-gradient-audit-v1"

: "${SLURM_ARRAY_TASK_ID:?the gradient audit requires its 10-element GPU array}"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= 10 )); then
    echo "array index must be in [0, 10)" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_EXECUTABLE" || ! -x "$SRUN" ]]; then
    echo "pinned Python or compute-node Slurm client is unavailable" >&2
    exit 2
fi

settings=(
    scene
    puzzle-3x3
    puzzle-4x4-100m
    cube-double
    cube-triple
    cube-quadruple-100m
    antmaze-large
    antmaze-giant
    humanoidmaze-medium
    humanoidmaze-large
)
setting="${settings[$SLURM_ARRAY_TASK_ID]}"
run_name="grounded-formal-${setting}-seed0"
checkpoint="$PROJECT_ROOT/outputs/treewm-grounded-formal-v1/$setting/treewm/$run_name/checkpoints/latest.pt"
if [[ ! -f "$PACKAGE/gradient_audit.py" || ! -f "$PACKAGE/recipes.json" ]]; then
    echo "immutable gradient-audit source package is unavailable" >&2
    exit 2
fi
if [[ ! -f "$checkpoint" ]]; then
    echo "paused formal checkpoint is unavailable: $checkpoint" >&2
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
    "$PYTHON_EXECUTABLE" "$PACKAGE/gradient_audit.py" \
    --checkpoint "$checkpoint" \
    --output-root "$OUTPUT_ROOT"
