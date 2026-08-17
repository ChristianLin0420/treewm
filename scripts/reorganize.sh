#!/bin/bash
# One-shot reorganisation of experiment artifacts into experiments/<cycle>/{runs,results,logs}.
#
# The flat layout had grown to 20 results_* dirs, 11 runs_* dirs and 53 loose logs at the
# repo root, with no way to tell which belonged together. Grouping by *cycle* rather than by
# artifact type is the useful axis: a cycle's checkpoints, metrics and driver logs are only
# interpretable next to each other.
#
#   bash scripts/reorganize.sh --dry-run   # print the plan, touch nothing
#   bash scripts/reorganize.sh             # execute
set -u
cd "$(dirname "$0")/.."
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# cycle | runs dirs | results dirs | log globs
MANIFEST=$(cat <<'EOF'
01-main-grid|runs|results|rerun_sweeps.log
02-novelty-target|runs_novelty|results_novelty|runs_novelty_grid.log runs_novelty_sweep.log
03-reliability|-|-|depth_sweep.log oracle_sel.log
04-decoded-goal|-|results_decoded results_policy results_novelty_decoded|expansion_policy.log
05-design-space|runs_screen runs_phase2|results_screen results_phase2 results_phase2_screen results_tracks|overnight.log overnight2.log screen_phase1.log analyze_screen.log render_screen.log render_phase2.log phase2.log phase2_coverage.log phase2_sweep.log tracks_def.log
06-q1-q2-q3|runs_q1 runs_q2|results_q1 results_q2 results_q2_screen results_q3|cycle3.log q1_antmaze.log q1_render.log q1_sweep.log q2_horizon.log q2_render.log q2_sweep.log q3_factorial.log
07-horizon-antmaze|runs_h runs_antmaze runs_ant4|results_h results_antmaze results_ant4|cycle4.log h_render.log h_report.log h_sweep.log hsweep.log horizon_analysis.log antmaze.log antmaze_stage.log antmaze_sweep.log ant4.log ant4_loco.log ant4_render.log ant4_stage.log ant4_sweep.log flat_h20.log render_antmaze.log
08-scaling|runs_large runs_giant|results_large results_difficulty|large_stage.log large_sweep.log large_train.log giant_stage.log giant_train.log hext_train.log diffcurve_giant.log diffcurve_large.log diffcurve_large2.log diffcurve_pm.log scaling_report.log scaling_report3.log hstar_model.log
EOF
)

run() { if [ "$DRY" = 1 ]; then echo "    $*"; else eval "$*"; fi; }

moved=0; missing=0
while IFS='|' read -r cycle runs results logs; do
    [ -z "$cycle" ] && continue
    echo "== $cycle"
    for kind in runs results logs; do
        case $kind in
            runs) items=$runs;; results) items=$results;; logs) items=$logs;;
        esac
        [ "$items" = "-" ] && continue
        for it in $items; do
            if [ ! -e "$it" ]; then echo "    MISSING $it"; missing=$((missing+1)); continue; fi
            run "mkdir -p experiments/$cycle/$kind"
            # runs/results keep their own name inside; logs land flat
            run "mv '$it' 'experiments/$cycle/$kind/'"
            moved=$((moved+1))
        done
    done
done <<< "$MANIFEST"

echo
echo "moved=$moved missing=$missing"
if [ "$DRY" = 1 ]; then
    echo
    echo "-- would be LEFT BEHIND at root (must be empty of experiment artifacts) --"
    ls -p | grep -v / | grep -E '\.log$' || echo "  (no stray logs)"
    ls -d results*/ runs*/ 2>/dev/null || echo "  (no stray results_/runs_ dirs)"
fi
