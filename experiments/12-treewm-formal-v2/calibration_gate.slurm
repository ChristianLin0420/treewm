#!/bin/bash
#SBATCH --job-name=treewm-v2-cal-gate
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
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?submit from repository root with submit.py}"
CAMPAIGN_DIR="$PROJECT_ROOT/experiments/12-treewm-formal-v2"
PYTHON_EXECUTABLE="${TREEWM_PYTHON:-/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python}"
: "${TREEWM_DATA_ROOT:?set immutable OGBench source root}"
: "${TREEWM_CACHE:?set approved reused raw cache root}"
: "${TREEWM_CONTRACT_ROOT:?set fresh v2 contract root}"
cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/submit.py" \
    --verify-source-snapshot --repo-root "$PROJECT_ROOT"
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/prepare_cache.py" \
    --manifest "$CAMPAIGN_DIR/manifest.json" --phase validate-all \
    --data-root "$TREEWM_DATA_ROOT" --cache-root "$TREEWM_CACHE" \
    --contract-root "$TREEWM_CONTRACT_ROOT"
