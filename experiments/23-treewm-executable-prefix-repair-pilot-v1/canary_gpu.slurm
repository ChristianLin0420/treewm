#!/bin/bash
#SBATCH --partition=polar4,polar3,polar,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --export=NONE
#SBATCH --open-mode=append

set -Eeuo pipefail
umask 077

PYTHON_EXECUTABLE=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
export PATH=/usr/bin:/bin

[[ "$#" -eq 3 ]] || { echo "usage: canary_gpu.slurm WORKER STATE_ROOT WAVE_ROLE" >&2; exit 2; }
WORKER=$1
STATE_ROOT=$2
WAVE_ROLE=$3
[[ "$WAVE_ROLE" == wave0 || "$WAVE_ROLE" == wave1 ]] || { echo "invalid canary GPU role" >&2; exit 2; }
[[ -f "$WORKER" && ! -L "$WORKER" ]] || { echo "canary worker is unavailable or symlinked" >&2; exit 2; }
[[ -d "$STATE_ROOT" && ! -L "$STATE_ROOT" ]] || { echo "canary state root is unavailable or symlinked" >&2; exit 2; }
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || { echo "canary within-wave requeue is forbidden" >&2; exit 2; }
: "${CUDA_VISIBLE_DEVICES:?canary requires one visible GPU}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo "canary sees more than one GPU" >&2; exit 2; }

exec "$PYTHON_EXECUTABLE" -I -S -B "$WORKER" --state-root "$STATE_ROOT" --role "$WAVE_ROLE"
