#!/bin/bash
#SBATCH --job-name=treewm-grounded-gauge-all-ten-gate-v2
#SBATCH --partition=cpu
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append

set -Eeuo pipefail
PROJECT_ROOT="${TREEWM_GAUGE_BRIDGE_REPO_ROOT:?launch only through guarded Exp21 submit.py}"
CAMPAIGN_DIR="$PROJECT_ROOT/experiments/21-treewm-grounded-gauge-all-ten-bridge-v2"
PYTHON_EXECUTABLE=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
: "${TREEWM_GATE_TARGET:?missing sealed gate target}"
[[ "$TREEWM_GATE_TARGET" == "25000" ]] || exit 2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
cd "$PROJECT_ROOT"
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/campaign.py" snapshot --repo-root "$PROJECT_ROOT"
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/stage_gate.py" \
    --repo-root "$PROJECT_ROOT" --manifest "$CAMPAIGN_DIR/manifest.json" --publish
