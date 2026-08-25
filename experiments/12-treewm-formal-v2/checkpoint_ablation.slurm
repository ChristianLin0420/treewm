#!/bin/bash
# Read-only inference ablation. The default representative screen has four settings
# and nine arms, hence 4 x 9 = 36 independent work items. Only after the documented
# promotion criterion passes, launch all ten settings with:
#   sbatch --array=0-89%40 checkpoint_ablation.slurm --stage full
# This file is intentionally not submitted by the source tree.
#SBATCH --job-name=treewm-v2-ckpt-ablate
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-35%36
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --open-mode=append

set -Eeuo pipefail

SLURM_BIN="${TREEWM_SLURM_BIN:-/cm/shared/apps/slurm/current/bin}"
if [[ ! -x "$SLURM_BIN/srun" ]]; then
    echo "required Slurm client is unavailable: $SLURM_BIN/srun" >&2
    exit 127
fi
export PATH="$SLURM_BIN:/usr/local/bin:/usr/bin:/bin"

SOURCE_ROOT="${TREEWM_SOURCE_ROOT:-${SLURM_SUBMIT_DIR:?submit from a TreeWM source root}}"
PROJECT_ROOT="${TREEWM_PROJECT_ROOT:-/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/treewm}"
PYTHON_EXECUTABLE="${TREEWM_PYTHON:-/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python}"
CHECKPOINT_GLOB="${TREEWM_ABLATION_CHECKPOINT_GLOB:-$PROJECT_ROOT/outputs/treewm-50task-1m-v2/*/treewm/*-seed0/checkpoints/latest.pt}"
OUTPUT_ROOT="${TREEWM_ABLATION_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/treewm-v2-checkpoint-ablation}"
: "${SLURM_ARRAY_TASK_ID:?checkpoint ablation requires an array task ID}"
if (( SLURM_ARRAY_TASK_ID < 0 )); then
    echo "invalid compact-grid work index $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi

cd "$SOURCE_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

"$SLURM_BIN/srun" --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 --cpus-per-task=8 \
    "$PYTHON_EXECUTABLE" scripts/checkpoint_ablation.py \
        --checkpoint-glob "$CHECKPOINT_GLOB" \
        --output-root "$OUTPUT_ROOT" \
        --work-index "$SLURM_ARRAY_TASK_ID" \
        --resume "$@"
