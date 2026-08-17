# TreeWM — findings of record

Frozen at tag `scaling-v1` (two layouts) and extended here to three. Every number below
comes from an artifact in `results_difficulty/` produced by the script named beside it;
nothing is quoted from memory.

## The falsifiable ladder, as it resolved

| rung | claim | verdict |
|---|---|---|
| SingleWM < FlatKWM | prediction must be multimodal | **supported** — SingleWM scores 0.000 on every run, budget and dataset |
| FlatKWM < trees | prediction must be recursive | **supported** — see the recursion advantage below |
| any-tree < q-aware tree | controllability metric `q` helps | **not supported** — negative advantage vs both `z` and a dimension-matched random projection, all 42 runs |
| fixed < learned allocation | allocation should be learned | **not supported** — nothing beat RandomTreeWM at matched budget |

The fourth rung failed against every allocation rule tried: learned coverage gain, learned
min-novelty in `q` and in `z`, goal-directed best-first, goal+α·novelty at several α,
`root_quota`, `depth_balanced`, `broad_to_focused`. Coverage anti-correlates with success
(ρ = −0.77). Learned horizon selection collapsed onto the longest horizon 87.6% of the time
(entropy 0.428 vs 1.442 uniform) and was indistinguishable from uniform random selection.

Extra prediction compute does not help either: on pointmaze-large the budget curve is flat
to declining, with the best arm peaking at 32 nodes and *falling* by 256.

## What replicates: the recursion advantage grows with required horizon

`scripts/difficulty_curve.py` → `scripts/scaling_report.py`, 3 layouts × 21 distance bins ×
6 horizons × 3 seeds × 2 budgets.

Measured as distance reduction (success-based Δ is bounded above by recursive success, so it
collapses for arithmetic reasons once both arms approach the floor):

| layout | Spearman(Δ distance-reduction, task distance) |
|---|---|
| pointmaze-medium-stitch | +1.000 |
| pointmaze-large-stitch | +0.929 |
| pointmaze-giant-stitch | +0.933 |
| **pooled, 21 bins** | **+0.913** |

## What does NOT replicate: h*(task distance)

The two-layout result — h* shifting 8 → 20 as tasks lengthen — looked like the project's most
valuable finding. `giant` was added specifically to falsify it, being long like `large` but
twisty like `medium` (turn density 0.444 vs 0.324 and 0.516), which breaks the collinearity
that makes distance and straightness indistinguishable on two layouts.

| layout | h* across increasing distance |
|---|---|
| medium | 8, 8, **20, 20, 20** |
| large | 20, 20, 16, 32, 48, 32, 64 |
| giant | **16, 16, 16, 16, 16**, 64, 64, 64 |

`giant`'s h* is flat at 16 across 6.3 → 40 world units, a 6× distance range. At matched
distance ≈40 the three layouts want h* = 20 / 16 / 48. **h* is not a function of task
distance.** Leave-one-*layout*-out R² on log2(h*) is negative for every candidate model
(`scripts/hstar_model.py`): distance −0.547, constant −0.761, per-task straightness −1.413,
layout turn density −3.405. None beats predicting the training mean.

Two corrections this exposed in the earlier two-layout reading:

- Both of medium's h*=8 bins were **floor-censored** (h=4 was never tested), so the 8→20
  shift rested on two unbracketed points. `hstar_model.py` now flags any argmax sitting at
  the end of the tested ladder and excludes it from the fit.
- large's apparent h*=32 plateau was an artifact of the ladder stopping at 32; with h=48/64
  added it becomes 32 → 48 → 64, and most bins show a genuine interior peak.

## What survives

An interior optimal temporal resolution exists and is real — h=8 fails everywhere beyond
short range, h≈16–32 is best in nearly every solvable bin on all three layouts, and the
mechanism is measured rather than inferred (per-edge execution error rises ~9× from h=8 to
h=48 while temporal reach rises). But **its location is environment-specific and not
predictable from task geometry** with three layouts.

Supporting the claim that this is about control rather than prediction quality: on large,
validation loss is minimised at h=16 while success peaks at h=48. The best-predicting model
is not the best-acting one.

## Negative result worth recording: AntMaze

Success ≈0 at 5k, 50k and 300k steps. The distance-bucketed evaluation showed failure even in
the nearest bin (11.6 world units), which falsified the earlier "goals are too far" diagnosis
— the bottleneck is locomotion competence, not planning horizon.

## Reproducing

```bash
python scripts/run_screen.py --recipes H8 H16 H20 H32 H48 H64 Fh20 \
    --dataset pointmaze_giant_stitch --seeds 0 1 2 --steps 20000 --run-root runs_giant
python scripts/difficulty_curve.py --runs runs_giant --tag pointmaze_giant \
    --bins 1 3 5 7 9 12 16 21 26 32 --per-bin 12 --episodes 2 --budgets 64 128
python scripts/scaling_report.py     # the six panels
python scripts/hstar_model.py        # the leave-one-layout-out fit
```

RNG streams for training, planner, evaluation and visualisation are isolated
(`treewm/utils/rng.py`); results are bit-identical across visualisation cadences, so logging
is observational. All artifacts are provenance-stamped and reject mixed metadata.
