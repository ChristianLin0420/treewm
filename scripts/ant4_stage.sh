#!/bin/bash
# AntMaze generalisation run: FlatKWM vs recursive TreeWM at the swept-best horizon.
#
# 300k steps is only feasible with the future-set cache. Anchors are revisited ~500x in
# a run this long and were being rebuilt from scratch every visit; caching measured 480x
# on repeat access (14.4 -> 0.03 ms/item).
set -u
cd /localhome/local-chrislin/treewm
PY=~/miniconda3/envs/treewm/bin/python
log() { echo "[ant4 $(date +%H:%M:%S)] $*"; }

log "AntMaze 300k: FlatKWM vs recursive h=20 (future-set cache ENABLED)"
$PY -u scripts/run_screen.py --recipes A4_flat A4_best_h --seeds 0 \
    --dataset antmaze_teleport_navigate --steps 300000 --gpus 0 1 --per-gpu 1 --num-workers 6 \
    --run-root runs_ant4 \
    --extra train.max_train_anchors=150000 future_sets.retrieval_pool=200000 \
            future_sets.query_multiplier=2 future_sets.cache=true \
            future_sets.horizons=[20] future_sets.h_max=20 future_sets.fixed_horizon=20 \
            train.eval_every=25000 train.ckpt_every=25000 \
            train.viz_every=2000 train.viz_every_early=2000 train.viz_early_until=0 \
            eval.episodes_per_task=2 eval.num_hard_tasks=6 \
    > ant4.log 2>&1
log "training done"

$PY -u scripts/run_sweeps.py --runs runs_ant4 --out results_ant4 --budgets 16 32 64 128 256 \
    --gpus 0 1 > ant4_sweep.log 2>&1
log "sweep done"

$PY -u scripts/locomotion_check.py --runs runs_ant4 --episodes 3 --max-tasks 6 \
    --out results_ant4 > ant4_loco.log 2>&1
log "locomotion check done"

$PY -u scripts/render_trees.py --runs runs_ant4 --all --budget 64 --anchors 4 \
    --ground-subset 0 --no-video > ant4_render.log 2>&1
log "ANTMAZE STAGE COMPLETE"
