#!/bin/bash
#SBATCH --job-name=treewm50-report
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append

set -Eeuo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root with submit.py}"
CAMPAIGN_DIR="$PROJECT_ROOT/experiments/11-treewm-formal"
PYTHON_EXECUTABLE="${TREEWM_PYTHON:-/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python}"
: "${TREEWM_DATA_ROOT:?set TREEWM_DATA_ROOT to the immutable OGBench source root}"
: "${TREEWM_CACHE:?set TREEWM_CACHE to durable cache storage}"
: "${TREEWM_RUN_ROOT:?set TREEWM_RUN_ROOT to durable formal run storage}"

cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/aggregate.py" \
    --manifest "$CAMPAIGN_DIR/manifest.json" \
    --repo-root "$PROJECT_ROOT" \
    --run-root "$TREEWM_RUN_ROOT" \
    --data-root "$TREEWM_DATA_ROOT" \
    --cache-root "$TREEWM_CACHE" \
    --output-dir "$TREEWM_RUN_ROOT/report"
