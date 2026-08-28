#!/bin/bash
#SBATCH --job-name=treewm-exp24-formal-report
#SBATCH --partition=cpu
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --export=NONE
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -Eeuo pipefail
umask 077
PYTHON_EXECUTABLE=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
[[ "$#" -eq 4 ]] || { echo "report.slurm requires four arguments" >&2; exit 2; }
SNAPSHOT_ROOT=$1; SUBMISSION_ROOT=$2; SUBMISSION_SHA256=$3; NODE=$4
[[ "$SUBMISSION_SHA256" =~ ^[0-9a-f]{64}$ && "$NODE" == formal_report ]] || { echo "report identity differs" >&2; exit 2; }
PROGRAM="$SNAPSHOT_ROOT/experiments/24-treewm-executable-prefix-formal-v1/report.py"
[[ -f "$PROGRAM" && ! -L "$PROGRAM" && -d "$SUBMISSION_ROOT" && ! -L "$SUBMISSION_ROOT" ]] || { echo "report paths differ" >&2; exit 2; }
cd "$SNAPSHOT_ROOT"
exec "$PYTHON_EXECUTABLE" -I -S -B "$PROGRAM" --publish --submission-root "$SUBMISSION_ROOT" --submission-sha256 "$SUBMISSION_SHA256"
