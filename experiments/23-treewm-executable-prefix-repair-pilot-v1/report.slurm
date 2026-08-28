#!/bin/bash
#SBATCH --job-name=treewm-exp23-launch5-terminal-report
#SBATCH --partition=cpu
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --open-mode=append

set -Eeuo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "report.slurm requires snapshot_root submission_root submission_sha256" >&2
    exit 2
fi
SNAPSHOT_ROOT="$1"
SUBMISSION_ROOT="$2"
SUBMISSION_SHA256="$3"
PACKAGE_RELATIVE="experiments/23-treewm-executable-prefix-repair-pilot-v1"
PYTHON_EXECUTABLE="/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python"
REPORT_PROGRAM="$SNAPSHOT_ROOT/$PACKAGE_RELATIVE/report.py"
CONTRACT="$SUBMISSION_ROOT/SUBMISSION_CONTRACT.json"

if [[ ! "$SUBMISSION_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "malformed submission SHA256" >&2
    exit 2
fi
if [[ ! -d "$SNAPSHOT_ROOT" || -L "$SNAPSHOT_ROOT" || ! -d "$SUBMISSION_ROOT" || -L "$SUBMISSION_ROOT" ]]; then
    echo "snapshot/submission root is unavailable or symlinked" >&2
    exit 2
fi
if [[ ! -f "$REPORT_PROGRAM" || -L "$REPORT_PROGRAM" || ! -f "$CONTRACT" || -L "$CONTRACT" ]]; then
    echo "report program/submission contract is unavailable or symlinked" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
    echo "pinned Python is unavailable" >&2
    exit 2
fi

export PATH="/cm/shared/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "$SNAPSHOT_ROOT"
exec "$PYTHON_EXECUTABLE" -I -S -B "$REPORT_PROGRAM" \
    --publish \
    --snapshot-root "$SNAPSHOT_ROOT" \
    --submission-root "$SUBMISSION_ROOT" \
    --submission-sha256 "$SUBMISSION_SHA256"
