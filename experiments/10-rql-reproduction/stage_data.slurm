#!/bin/bash
#SBATCH --job-name=rql-data
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@420
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
: "${RQL_DATA_ROOT:?set RQL_DATA_ROOT to durable dataset storage}"

cd "$PROJECT_ROOT"
set +e
srun \
    --ntasks=1 \
    --cpus-per-task=4 \
    --kill-on-bad-exit=0 \
    "$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/prepare_data.py" \
        --manifest "$CAMPAIGN_DIR/manifest.json" \
        --data-root "$RQL_DATA_ROOT" \
        --download
stage_status=$?
set -e

if [[ "$stage_status" -eq 75 ]]; then
    # The downloader fsyncs its .part file before exit 75.  RequeueExit is not
    # configured on this cluster, so request the requeue explicitly.
    scontrol requeue "$SLURM_JOB_ID"
fi
exit "$stage_status"
