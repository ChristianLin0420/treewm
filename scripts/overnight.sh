#!/bin/bash
# Overnight driver: waits for Phase-1 screening, then runs the remaining pipeline
# unattended. Each stage logs to its own file so a failure in one does not lose the rest.
set -u
cd /localhome/local-chrislin/treewm
PY=~/miniconda3/envs/treewm/bin/python
log() { echo "[overnight $(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- wait for Phase 1
log "waiting for Phase-1 screening"
while pgrep -f "run_scree[n].py" >/dev/null 2>&1; do sleep 60; done
log "Phase-1 screening finished"

$PY scripts/analyze_screen.py > analyze_screen.log 2>&1
log "screening analysed -> analyze_screen.log"

# ---------------------------------------------------------------- tracks D / E / F
log "starting tracks D/E/F (no retraining)"
$PY -u scripts/run_tracks_def.py --runs runs_novelty --dataset pointmaze-medium-stitch \
    --arm randomtreewm --budgets 32 64 128 --episodes 2 --max-tasks 8 \
    > tracks_def.log 2>&1
log "tracks D/E/F done -> tracks_def.log"

# ------------------------------------------------------- Phase 2: promoted recipes
# Promotion is computed by analyze_screen.py; the shortlist below is the union of the
# mechanisms worth combining, filtered to those that actually completed screening.
PROMOTED=$($PY - <<'EOF'
import json, pathlib
p = pathlib.Path('results_screen/screen_summary.json')
promoted = json.loads(p.read_text()).get('promoted', []) if p.exists() else []
# always carry the mandatory controls plus the combination recipes worth 3 seeds
base = ['A0_random', 'A0_bfs']
combos = ['P_ms_ss', 'P_k8_short_ms', 'P_k8_ss']
picked = base + combos + [r.rsplit('_s', 1)[0] for r in promoted][:2]
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
log "Phase 2 training done -> phase2.log"

$PY -u scripts/run_sweeps.py --runs runs_phase2 --out results_phase2 \
    --budgets 16 32 64 128 256 --gpus 0 0 1 1 > phase2_sweep.log 2>&1
log "Phase 2 budget sweep done"

$PY -u scripts/coverage_sweep.py --runs runs_phase2 --out results_phase2 \
    > phase2_coverage.log 2>&1
log "Phase 2 coverage sweep done"

# ---------------------------------------------------------------- visualisations
log "rendering tree visualisations"
$PY -u scripts/render_trees.py --runs runs_phase2 --all --budget 64 --anchors 8 \
    --ground-subset 2 > render_phase2.log 2>&1
log "visualisations done"

# ---------------------------------------------------------- Phase 3: AntMaze smoke
log "starting AntMaze smoke test"
$PY -u scripts/run_screen.py --recipes A0_flat A0_random P_ms_ss --seeds 0 \
    --dataset antmaze_teleport_navigate --steps 12000 --per-gpu 2 --num-workers 4 \
    --run-root runs_antmaze --extra train.max_train_anchors=100000 \
    > antmaze.log 2>&1
log "AntMaze training done"

$PY -u scripts/run_sweeps.py --runs runs_antmaze --out results_antmaze \
    --budgets 32 64 128 --gpus 0 1 > antmaze_sweep.log 2>&1
log "AntMaze sweep done"

log "OVERNIGHT PIPELINE COMPLETE"
