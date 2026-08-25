#!/bin/bash
# Read-only held-out rescore for the 40 paused formal checkpoints. This source file is
# intentionally not submitted; launch explicitly only after reviewing the dry-run map.
#SBATCH --job-name=treewm-v2-rescore
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-39%40
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --open-mode=append

set -Eeuo pipefail

SLURM_SRUN="/cm/shared/apps/slurm/current/bin/srun"
if [[ ! -x "$SLURM_SRUN" ]]; then
    echo "required absolute Slurm client is unavailable: $SLURM_SRUN" >&2
    exit 127
fi

SOURCE_ROOT="${TREEWM_SOURCE_ROOT:-${SLURM_SUBMIT_DIR:?submit from a TreeWM source root}}"
PROJECT_ROOT="${TREEWM_PROJECT_ROOT:-/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/treewm}"
PYTHON_EXECUTABLE="${TREEWM_PYTHON:-/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python}"
CONTRACT_ROOT="${TREEWM_CONTRACT_ROOT:-/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/datasets/treewm-50task-formal-v2-contracts-v1}"
CACHE_ROOT="${TREEWM_CACHE:-/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/datasets/treewm-50task-full-cache-v1}"
OUTPUT_ROOT="${TREEWM_RESCORE_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/treewm-v2-validation-rescore}"
CHECKPOINT_GLOB="${TREEWM_RESCORE_CHECKPOINT_GLOB:-$PROJECT_ROOT/outputs/treewm-50task-1m-v2/*/treewm/*/checkpoints/latest.pt}"
: "${SLURM_ARRAY_TASK_ID:?checkpoint rescore requires an array task ID}"
if (( SLURM_ARRAY_TASK_ID < 0 )); then
    echo "invalid checkpoint work index $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi

cd "$SOURCE_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

"$SLURM_SRUN" \
    --ntasks=1 \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    "$PYTHON_EXECUTABLE" scripts/rescore_checkpoint.py \
        --checkpoint-glob "$CHECKPOINT_GLOB" \
        --work-index "$SLURM_ARRAY_TASK_ID" \
        --device cuda \
        --val-batches 20 \
        --num-workers 8 \
        --contract-root "$CONTRACT_ROOT" \
        --cache-root "$CACHE_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --resume \
        "$@"
