#!/bin/bash
# Wait for the pointmaze-large training (by PID), then run its distance-bucketed
# evaluation and the cross-environment scaling report.
set -u
cd /localhome/local-chrislin/treewm
PY=~/miniconda3/envs/treewm/bin/python
log() { echo "[large $(date +%H:%M:%S)] $*"; }
TPID=$(cat /tmp/largepid)

while kill -0 "$TPID" 2>/dev/null; do sleep 60; done
log "large training done"

$PY -u scripts/difficulty_curve.py --runs runs_large --tag pointmaze_large \
    --bins 1 3 5 7 9 12 16 21 --per-bin 12 --episodes 2 --budgets 64 128 \
    --include H8 H16 H20 H32 Fh20 > diffcurve_large.log 2>&1
log "large difficulty curve done"

$PY -u scripts/scaling_report.py > scaling_report.log 2>&1
log "scaling report done"

$PY -u scripts/run_sweeps.py --runs runs_large --out results_large \
    --budgets 16 32 64 128 256 --gpus 0 0 1 1 > large_sweep.log 2>&1
log "LARGE STAGE COMPLETE"
