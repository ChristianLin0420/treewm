#!/bin/bash
#SBATCH --job-name=rql50-report
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
CAMPAIGN_DIR="$PROJECT_ROOT/experiments/10-rql-reproduction"
PYTHON_EXECUTABLE="${RQL_PYTHON:-python}"
: "${RQL_RUN_ROOT:?set RQL_RUN_ROOT to durable checkpoint storage}"

cd "$PROJECT_ROOT"
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/aggregate.py" \
    --manifest "$CAMPAIGN_DIR/manifest.json" \
    --run-root "$RQL_RUN_ROOT" \
    --output-dir "$RQL_RUN_ROOT/report"
