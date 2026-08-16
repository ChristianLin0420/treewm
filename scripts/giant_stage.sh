#!/bin/bash
# Steps 2-4 of the post-freeze plan, chained by PID (never by process name -- a name
# pattern once matched this driver's own shell and idled the GPUs for 8 hours).
#
#   2. extended horizons h=48,64 on large  -> is h*=32 an interior optimum or a boundary?
#   3. giant layout (long+twisty)          -> decorrelates distance from straightness
#   4. pooled h* model fit across 3 layouts
set -u
cd /localhome/local-chrislin/treewm
PY=~/miniconda3/envs/treewm/bin/python
log() { echo "[giant $(date +%H:%M:%S)] $*"; }

# --- step 2: wait for the extended-horizon training already running ---------------
HPID=$(cat /tmp/hextpid)
while kill -0 "$HPID" 2>/dev/null; do sleep 60; done
log "extended-horizon (h48,h64) training done"

# re-run the large curve with the new arms folded in
$PY -u scripts/difficulty_curve.py --runs runs_large --tag pointmaze_large \
    --bins 1 3 5 7 9 12 16 21 --per-bin 12 --episodes 2 --budgets 64 128 \
    --include H8 H16 H20 H32 H48 H64 Fh20 > diffcurve_large2.log 2>&1
log "large curve (8 horizons) done"

# --- step 3: giant ----------------------------------------------------------------
$PY -u scripts/run_screen.py --recipes H8 H16 H20 H32 H48 H64 Fh20 \
    --dataset pointmaze_giant_stitch --seeds 0 1 2 --steps 20000 \
    --run-root runs_giant > giant_train.log 2>&1
log "giant training done"

# giant reaches 31 cells, so it needs two bins beyond the large schedule
$PY -u scripts/difficulty_curve.py --runs runs_giant --tag pointmaze_giant \
    --bins 1 3 5 7 9 12 16 21 26 32 --per-bin 12 --episodes 2 --budgets 64 128 \
    --include H8 H16 H20 H32 H48 H64 Fh20 > diffcurve_giant.log 2>&1
log "giant curve done"

# --- step 4: pool and fit ----------------------------------------------------------
$PY -u scripts/scaling_report.py > scaling_report3.log 2>&1
$PY -u scripts/hstar_model.py > hstar_model.log 2>&1
log "GIANT STAGE COMPLETE"
