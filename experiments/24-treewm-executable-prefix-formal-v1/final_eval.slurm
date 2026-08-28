#!/bin/bash
#SBATCH --job-name=treewm-exp24-formal-heldout
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
#SBATCH --export=NONE
#SBATCH --open-mode=append

set -Eeuo pipefail
umask 077
PYTHON_EXECUTABLE=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
[[ "$#" -eq 4 ]] || { echo "final_eval.slurm requires four arguments" >&2; exit 2; }
SNAPSHOT_ROOT=$1; SUBMISSION_ROOT=$2; SUBMISSION_SHA256=$3; NODE=$4
[[ "$SUBMISSION_SHA256" =~ ^[0-9a-f]{64}$ && "$NODE" == heldout_eval ]] || { echo "held-out identity differs" >&2; exit 2; }
: "${SLURM_ARRAY_TASK_ID:?Exp24 held-out evaluation requires an array task}"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || { echo "held-out index differs" >&2; exit 2; }
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 200 )) || { echo "held-out index outside 200 cells" >&2; exit 2; }
PROGRAM="$SNAPSHOT_ROOT/experiments/24-treewm-executable-prefix-formal-v1/final_eval.py"
[[ -f "$PROGRAM" && ! -L "$PROGRAM" && -d "$SUBMISSION_ROOT" && ! -L "$SUBMISSION_ROOT" ]] || { echo "held-out paths differ" >&2; exit 2; }
cd "$SNAPSHOT_ROOT"
exec "$PYTHON_EXECUTABLE" -I -S -B "$PROGRAM" --execute --snapshot-root "$SNAPSHOT_ROOT" --submission-root "$SUBMISSION_ROOT" --submission-sha256 "$SUBMISSION_SHA256" --node "$NODE" --cell-index "$SLURM_ARRAY_TASK_ID"
