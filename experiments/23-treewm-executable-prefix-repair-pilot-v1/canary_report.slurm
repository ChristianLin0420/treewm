#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --qos=normal
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --export=NONE
#SBATCH --open-mode=append

set -Eeuo pipefail
umask 077

PYTHON_EXECUTABLE=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
export PATH=/usr/bin:/bin

[[ "$#" -eq 2 ]] || { echo "usage: canary_report.slurm WORKER STATE_ROOT" >&2; exit 2; }
WORKER=$1
STATE_ROOT=$2
[[ -f "$WORKER" && ! -L "$WORKER" ]] || { echo "canary worker is unavailable or symlinked" >&2; exit 2; }
[[ -d "$STATE_ROOT" && ! -L "$STATE_ROOT" ]] || { echo "canary state root is unavailable or symlinked" >&2; exit 2; }
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || { echo "canary report requeue is forbidden" >&2; exit 2; }

exec "$PYTHON_EXECUTABLE" -I -S -B "$WORKER" --state-root "$STATE_ROOT" --role report
