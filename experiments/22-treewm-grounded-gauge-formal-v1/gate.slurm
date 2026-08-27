#!/bin/bash
#SBATCH --job-name=treewm-grounded-gauge-formal-gate-v1
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
PROJECT_ROOT="${TREEWM_GAUGE_FORMAL_V1_REPO_ROOT:?launch only through guarded submit.py from a sealed snapshot}"
CAMPAIGN_DIR="$PROJECT_ROOT/experiments/22-treewm-grounded-gauge-formal-v1"
PYTHON_EXECUTABLE=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
: "${TREEWM_GATE_TARGET:?missing sealed stage gate target}"
case "$TREEWM_GATE_TARGET" in
    2000|25000|100000|1000000) ;;
    *) echo "invalid stage gate target" >&2; exit 2 ;;
esac
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
cd "$PROJECT_ROOT"
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/campaign.py" snapshot --repo-root "$PROJECT_ROOT"
"$PYTHON_EXECUTABLE" "$CAMPAIGN_DIR/stage_gate.py" \
    --stage-target "$TREEWM_GATE_TARGET" \
    --repo-root "$PROJECT_ROOT" \
    --manifest "$CAMPAIGN_DIR/manifest.json" \
    --publish
