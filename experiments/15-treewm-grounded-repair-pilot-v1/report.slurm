#!/bin/bash
#SBATCH --job-name=treewm-repair-report-v1
#SBATCH --partition=cpu
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append

set -Eeuo pipefail

PROJECT_ROOT="${TREEWM_PILOT_REPO_ROOT:?launch through guarded submit.py with a sealed snapshot}"
CAMPAIGN_DIR="$PROJECT_ROOT/experiments/15-treewm-grounded-repair-pilot-v1"
PYTHON_EXECUTABLE=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
: "${TREEWM_EXPECTED_SOURCE_SHA256:?launch through guarded submit.py}"
: "${TREEWM_EXPECTED_RUNTIME_SHA256:?launch through guarded submit.py}"
: "${TREEWM_EXPECTED_PACKAGE_PROTOCOL_SHA256:?launch through guarded submit.py}"
if [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
    echo "pinned Python executable is unavailable" >&2
    exit 2
fi
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/campaign.py" snapshot --repo-root "$PROJECT_ROOT" >/dev/null
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/report.py" \
    --repo-root "$PROJECT_ROOT" \
    --manifest "$CAMPAIGN_DIR/manifest.json" \
    --publish
