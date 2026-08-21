# TreeWM — findings of record

The PointMaze record was frozen at tag `scaling-v1` (two layouts) and extended here to
three. The cross-family section below extends the record through experiment 09's formal
Wave 4. Every number comes from recorded experiment artifacts or the corrected read-only
checkpoint evaluations described below; nothing is quoted from memory. See
`experiments/08-scaling/`, `experiments/09-cross-family/`, and `experiments/README.md` for
the cycle-by-cycle map.

## The falsifiable ladder, as it resolved

| rung | claim | verdict |
|---|---|---|
| SingleWM < FlatKWM | prediction must be multimodal | **supported** — SingleWM scores 0.000 on every run, budget and dataset |
| FlatKWM < trees | prediction must be recursive | **qualified** — supported across PointMaze layouts, not as a cross-family law; experiment 09 is mixed |
| any-tree < q-aware tree | controllability metric `q` helps | **not supported** — negative advantage vs both `z` and a dimension-matched random projection, all 42 runs |
| fixed < learned allocation | allocation should be learned | **not supported** — nothing beat RandomTreeWM at matched budget |

The fourth rung failed against every allocation rule tried: learned coverage gain, learned
min-novelty in `q` and in `z`, goal-directed best-first, goal+α·novelty at several α,
`root_quota`, `depth_balanced`, `broad_to_focused`. Coverage anti-correlates with success
(ρ = −0.77). Learned horizon selection collapsed onto the longest horizon 87.6% of the time
(entropy 0.428 vs 1.442 uniform) and was indistinguishable from uniform random selection.

Extra prediction compute does not help either: on pointmaze-large the budget curve is flat
to declining, with the best arm peaking at 32 nodes and *falling* by 256.

## What replicates within PointMaze: the recursion advantage grows with required horizon

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
— the bottleneck is task success competence, not planning horizon. Experiment 09 refines
this further: the Ant policies do move and reduce distance, but never complete a task.

## Cross-family formal run: the late Flat crossover does not generalize

Experiment 09 Wave 4 completed all six scheduled jobs with zero failures or OOM retries:
scene and cube-double to 300k, and resumed cube-single Flat/recursive runs to 1M. The fleet
controller ends `6/6 complete, 0 failed`; all six checkpoints are at their exact targets and
all job logs have clean done markers. The longest job, cube-single Flat, took 41.4 hours,
inside the four-day window.

There is an evaluation bug to account for. Periodic manipulation evaluations pass the
environment domain into both the planner and evaluator, but the built-in final path in
`scripts/train.py` does not. Its target-step metrics are therefore invalid, and
`scripts/eval.py` has the same omission. The target checkpoints below were instead evaluated
read-only with the domain passed explicitly, the scheduled `eval` RNG stream, five standard
tasks, 20 episodes per task, and a 64-node budget. The reported native subgoal fraction is
the environment's own measure; the generic domain subgoal tolerance is mis-scaled for these
observations and is used only to identify domain-aware events.

| environment / target | FlatKWM | RandomTreeWM | result |
|---|---|---|---|
| scene / 300k | 0/100 success; native fraction .492; 2.9% distance reduction | 11/100; .584; 22.1% | recursive is higher by 11 pp; Fisher p=.00073, 95% Newcombe CI for Flat−recursive [−18.63, −2.55] pp |
| cube-double / 300k | 0/100; .095; 0.70% | 0/100; .055; 0.07% | both remain at the success floor; the apparent 200–250k sign flip was not a competence crossover |
| cube-single / 1M | 20/100; .200; 35.4% | 15/100; .150; 31.5% | Flat +5 pp, but inconclusive: Fisher p=.457, 95% CI [−5.63, +15.55] pp |

Scene does not reproduce the Flat crossover and, in this seed, strongly favours recursion.
Its valid success curve (Flat/recursive) is 0/5, 0/16, 0/10, 0/12, 1/8, and 0/11 from 50k
through 300k. Its native fraction starts at .520 and finishes at .492 for Flat versus .584
for recursive. Cube-double never establishes competence: its curve is 0/5, 0/4, 0/1, 1/0,
1/0, and 0/0, while both arms finish below or near their .100 native starting fraction.
Thus one cross-family task favours recursion within this seed and the other is uninformative;
neither reproduces cube-single's late Flat pattern.

### Cube-single at 1M: neither provisional plateau survives

The complete late success tail at 50k intervals, including the corrected target evaluation,
is:

```text
step (k)   550  600  650  700  750  800  850  900  950  1000
Flat        17   16   28   30   22   23   25   20   19    20
recursive   20   16   16   13   17   18   14   23   19    15
```

Flat's 30/100 plateau does not hold: 30 occurs only at 150k and 700k, and the run ends at
20. Recursive's decline does not continue either: it oscillates, reaches a run high of 23 at
900k, and ends at 15. The late paths cross repeatedly—recursive has the higher count at 550k
and 900k, they tie at 600k and 950k, and Flat has the higher count at 650–850k and 1M. Across
the 19 shared periodic looks, Flat leads 14, ties two, and trails three; only its 150k
advantage survives Holm correction (adjusted p=.0124). The late 550k–1M mean is 22.0% versus
17.1% success, descriptive only. There is no significant late linear success trend in either
arm. Recursive nevertheless has the larger distance-reduction fraction at 12 of the 19
periodic checkpoints, showing that exact success and continuous approach progress need not
rank the arms alike.

The right conclusion is narrower than the interim one: cube-single shows a higher average
late Flat success rate in this seed, not a permanent crossover or a general convergence-rate
law. The exact 1M difference is compatible with substantial effects in either direction.

### Learned support still collapses after the anneal

The recursive cube-single support head was logged every 50 optimisation steps. After
latest-wall-time de-duplication across the resume, its terminal scalar is at step 999,950:

| step | effective branching factor | branch entropy | support precision | redundancy scale |
|---:|---:|---:|---:|---:|
| 50k | 1.816 | 1.264 | .517 | .000544* |
| 300k | 1.516 | .712 | .795 | 0 |
| 500k | 1.455 | .552 | .860 | 0 |
| 750k | 1.372 | .371 | .942 | 0 |
| 999,950 | 1.348 | .300 | .960 | 0 |

\* The logged 50k value averages the preceding 50-step window; the instantaneous coefficient
is zero at step 50k, and every logged window from 50,050 onward is exactly zero.

From 50k to the end, thresholded effective branching falls 25.8%. The entropy-equivalent
soft branch count, exp(H), falls from 3.538 to 1.350 (−61.8%). All 18 successive 50k block
means decline for both effective branching and entropy; from the 750–800k block mean to the
950k–1M block mean they fall another 1.6% and 18.5%, respectively. Collapse therefore
continues but decelerates near 1.35; there is no defensible stable plateau or recovery.

The redundancy coefficient is exactly zero after 50k, so annealing did not arrest this
long-run drift. This run is not an anneal/no-anneal A/B, however, and cannot show that the
penalty was causally irrelevant or whether it reduced the collapse rate. Effective branching
here is a learned-support-head diagnostic (`KEEP > .5`, K=4), not literal inference-tree
fan-out: generation normally admits all valid children and uses `KEEP` only to rank the final
budget overflow. Entropy is computed from normalized raw `KEEP` scores, making exp(H) a soft
complement to thresholded effective branching. Support recall is matching coverage and does
not apply the `KEEP > .5` threshold, so its constancy does not establish retained support;
rising precision is mechanically compatible with keeping fewer branches. Raw Flat effective
branching is not comparable because Flat uses K=256. Each logged diagnostic is the training
metric tracker's mean over the preceding 50 optimisation steps, not a held-out snapshot.

### Locomotion: a success null, not a universal movement null

All six preserved locomotion arms have exactly zero successes at every valid evaluation.
Flat runs have six evaluations through 300k and recursive runs have eight through 400k; there
is no actual 450k evaluation despite later training scalars. At the last fixed policy, 0/200
gives a two-sided 95% Clopper–Pearson upper endpoint of 1.83% for Ant and Humanoid, while
0/100 gives 3.62% for Soccer.

At the shared 300k evaluation:

| environment | Flat: displacement / distance reduction / moving | recursive: displacement / distance reduction / moving |
|---|---|---|
| Ant | 7.658 / 20.75% / 97% | 10.599 / 24.31% / 100% |
| Humanoid | 1.594 / 0.003% / 71.5% | 2.338 / 1.73% / 86% |
| Soccer | .029 / −0.002% / 0% | .130 / .202% / 5% |

Ant clearly moves and makes directed progress without succeeding. Humanoid often moves but
mostly wanders; recursive has only slight directed progress. Soccer Flat effectively never
moves the ball, and recursive produces only marginal movement. The result is a strong
success null across three environments, not evidence that every policy is stationary.

These cross-family comparisons use one training seed and fixed evaluation tasks/seeds.
Checkpoint looks are correlated, and the episode-level paired outcomes were not retained;
Fisher tests and Newcombe intervals are therefore aggregate within-run summaries, not
evidence of model-level generalisation. In particular, experiment 09 narrows the earlier
"prediction must be recursive" verdict to the replicated PointMaze setting rather than
overturning the environment-specific evidence there.

## Reproducing experiment 08

```bash
python scripts/run_screen.py --recipes H8 H16 H20 H32 H48 H64 Fh20 \
    --dataset pointmaze_giant_stitch --seeds 0 1 2 --steps 20000 \
    --run-root experiments/08-scaling/runs/giant
python scripts/difficulty_curve.py --runs experiments/08-scaling/runs/giant \
    --tag pointmaze_giant --bins 1 3 5 7 9 12 16 21 26 32 \
    --per-bin 12 --episodes 2 --budgets 64 128
python scripts/scaling_report.py     # the six panels (defaults now resolve to 08-scaling)
python scripts/hstar_model.py        # the leave-one-layout-out fit
```

RNG streams for training, planner, evaluation and visualisation are isolated
(`treewm/utils/rng.py`); results are bit-identical across visualisation cadences, so logging
is observational. All artifacts are provenance-stamped and reject mixed metadata.
