#!/bin/bash
# Overnight driver, v2.
#
# v1 deadlocked: it polled `pgrep -f run_screen.py` to wait for screening, and the shell
# that launched the driver had "run_screen.py --list" in its own command line, so the
# pattern matched the driver's own parent and the wait never terminated (8h idle).
#
# v2 contains NO process-name polling at all. Every stage is invoked directly and waited
# on by the shell, so a stage cannot be confused by an unrelated command line.
set -u
cd /localhome/local-chrislin/treewm
PY=~/miniconda3/envs/treewm/bin/python
log() { echo "[overnight $(date +%H:%M:%S)] $*"; }
stage() { log "START $1"; shift; "$@"; log "END rc=$?"; }

log "v2 driver start (Phase-1 screening already complete: 18/18)"

$PY scripts/analyze_screen.py > analyze_screen.log 2>&1
log "screening analysed -> analyze_screen.log"

# ------------------------------------------------- Phase 2 first (needs the GPUs most)
PROMOTED=$($PY - <<'EOF'
import json, pathlib
p = pathlib.Path('results_screen/screen_summary.json')
promoted = json.loads(p.read_text()).get('promoted', []) if p.exists() else []
base = ['A0_random', 'A0_bfs']                       # mandatory controls
top = [r.rsplit('_s', 1)[0] for r in promoted][:4]   # promoted list is criteria-sorted
combos = ['P_k8_short_ms']                           # components (K=8, short) both screened well
picked = base + top + combos
seen, out = set(), []
for r in picked:
    if r not in seen:
        seen.add(r); out.append(r)
print(' '.join(out))
EOF
)
log "Phase 2 recipes: $PROMOTED"
$PY -u scripts/run_screen.py --recipes $PROMOTED --seeds 0 1 2 \
    --dataset pointmaze_medium_stitch --steps 20000 --per-gpu 3 --num-workers 4 \
    --run-root runs_phase2 > phase2.log 2>&1
log "Phase 2 training done"

$PY -u scripts/run_sweeps.py --runs runs_phase2 --out results_phase2 \
    --budgets 16 32 64 128 256 --gpus 0 0 1 1 > phase2_sweep.log 2>&1
log "Phase 2 budget sweep done"

$PY -u scripts/coverage_sweep.py --runs runs_phase2 --out results_phase2 \
    > phase2_coverage.log 2>&1
log "Phase 2 coverage sweep done"

# ------------------------------------------------------------- tracks D / E / F
$PY -u scripts/run_tracks_def.py --runs runs_novelty --dataset pointmaze-medium-stitch \
    --arm randomtreewm --budgets 32 64 128 --episodes 2 --max-tasks 8 \
    > tracks_def.log 2>&1
log "tracks D/E/F done"

# ---------------------------------------------------------------- visualisations
$PY -u scripts/render_trees.py --runs runs_phase2 --all --budget 64 --anchors 8 \
    --ground-subset 2 > render_phase2.log 2>&1
log "Phase 2 visualisations done"

$PY -u scripts/render_trees.py --runs runs_screen --all --budget 64 --anchors 8 \
    --ground-subset 1 --no-video > render_screen.log 2>&1
log "Phase 1 visualisations done (same fixed anchors, retroactive)"

# ---------------------------------------------------------- Phase 3: AntMaze smoke
$PY -u scripts/run_screen.py --recipes A0_flat A0_random P_ms_ss --seeds 0 \
    --dataset antmaze_teleport_navigate --steps 12000 --per-gpu 2 --num-workers 4 \
    --run-root runs_antmaze --extra train.max_train_anchors=100000 \
    > antmaze.log 2>&1
log "AntMaze training done"

$PY -u scripts/run_sweeps.py --runs runs_antmaze --out results_antmaze \
    --budgets 32 64 128 --gpus 0 1 > antmaze_sweep.log 2>&1
log "AntMaze sweep done"

log "OVERNIGHT PIPELINE COMPLETE"
