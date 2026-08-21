# OGBench baseline matrix

This file is the evidence ledger for the canonical 85-dataset OGBench benchmark. It is
deliberately not presented as a single leaderboard: the published OGBench results and the
Reversal Q-Learning (RQL) results use different problems, training budgets, and evaluation
protocols. A number in one protocol must not be ranked against a number in the other.

The row manifest is verified against both immutable OGBench v1.2.1 manifests:
[`impls/hyperparameters.sh`](https://github.com/seohongpark/ogbench/blob/v1.2.1/impls/hyperparameters.sh)
and
[`data_gen_scripts/commands.sh`](https://github.com/seohongpark/ogbench/blob/v1.2.1/data_gen_scripts/commands.sh).
They give the same ordered 85 IDs: 47 state/vector datasets, 35 `visual-*` pixel datasets,
and three Powderworld pixel/grid datasets with discrete actions. Single-task aliases,
`oraclerep`, `-100m-`, and `cube-octuple` are not members of the canonical 85.

## How to read the matrix

- **OGBench-native columns** (`GCBC` through `HIQL`) are binary goal-success percentages,
  reported as mean ± standard deviation in the [OGBench paper](https://arxiv.org/html/2410.20092v2).
  Each dataset has five fixed goals. The paper evaluates 50 rollouts per goal, averages the
  final three evaluation epochs, and uses eight training seeds for state observations or
  four for pixel observations.
- **RQL-family columns** (`ReBRAC` through `RQL`) show the paper's plain integer point
  estimates. They are the five-task aggregate success percentages from
  [RQL Appendix Table 1](https://arxiv.org/abs/2606.17551),
  under its *single-task, reward-based offline-RL* protocol. RQL covers only 10 settings
  (50 task-specific agents). The inherited `ReBRAC`–`QAM-E` point estimates are QAM's
  1M-training-step, 12-seed results. RQL's added `BDPO`, `TFQL`, and `RQL` experiments use
  four seeds and 95% confidence intervals; the common table specifies 2M gradient steps,
  plus a 1M-step behavior-cloning warmup for BDPO. The setting uses action horizon 1 for
  locomotion and 5 for manipulation. These are not goal-conditioned OGBench scores. `TFQL`
  is the no-horizon-reduction ablation; `CGQL-M/L` and `QAM-F/E` are published variants.
- **Structural columns** add methods absent from RQL's experimental comparison:
  `ORS` is occupancy/world-model reward shaping; `GCIQL+TTGS` adds a directed graph and
  Dijkstra test-time search; `CompDiff` is a diffusion trajectory planner; and `MCTD` uses
  genuine Monte Carlo tree search over partially denoised plans. Their cells use each
  paper's own uncertainty convention. CompDiffuser and MCTD results are marked `†` because
  they modify the stock OGBench evaluation protocol.
- **TreeWM (ours)** is a readiness/evidence column, not a score column: `E` means an existing
  goal-conditioned experiment is on disk; `R` means the exact adapter exists but a formal
  result is not yet recorded; `A` means a state-domain adapter/config is required; and `X`
  means pixel or discrete-action architecture work is required. Existing `E` results use a
  different evaluation protocol and are therefore not inserted as comparable scores.
- `—` means **not reported**, never zero. A literal `0` is a measured zero. `S`, `V`, and
  `P` denote state-vector, visual-pixel, and Powderworld pixel/discrete datasets.
- RQL's `puzzle-4x4` and `cube-quadruple` row labels are marked `*`: those papers used 100M
  transition datasets, not the canonical dataset named by the row. They are placed on the
  closest semantic row only to expose the published evidence, not to claim protocol identity.
  RQL/QAM use sparse single-task rewards for scene and both puzzle rows; the other seven
  mapped settings use OGBench's default semi-sparse single-task reward.

## All 85 datasets × 30 methods

| # | dataset | obs | GCBC | GCIVL | GCIQL | QRL | CRL | HIQL | ReBRAC | FBRAC | BAM | FQL | FAWAC | CGQL | CGQL-M | CGQL-L | DAC | QSM | DSRL | FEdit | IFQL | QAM | QAM-F | QAM-E | BDPO | TFQL | RQL | ORS | GCIQL+TTGS | CompDiff (AR+replan)† | MCTD (base)† | TreeWM (ours) |
|---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `pointmaze-medium-navigate-v0` | S | 9±6 | 63±6 | 53±8 | 82±5 | 29±7 | 79±5 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 90.9±3.7 | — | 100±0 | A |
| 2 | `pointmaze-large-navigate-v0` | S | 29±6 | 45±5 | 34±3 | 86±9 | 39±7 | 58±5 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 88.1±9.4 | — | 98±6 | A |
| 3 | `pointmaze-giant-navigate-v0` | S | 1±2 | 0±0 | 0±0 | 68±7 | 27±10 | 46±9 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 91.9±3.0 | — | 100±0 | A |
| 4 | `pointmaze-teleport-navigate-v0` | S | 25±3 | 45±3 | 24±7 | 4±4 | 24±6 | 18±4 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | A |
| 5 | `pointmaze-medium-stitch-v0` | S | 23±18 | 70±14 | 21±9 | 80±12 | 0±1 | 74±6 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 44.0±8.0 | 100±0 | — | E |
| 6 | `pointmaze-large-stitch-v0` | S | 7±5 | 12±6 | 31±2 | 84±15 | 0±0 | 13±6 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 29.0±2.2 | 100±0 | — | E |
| 7 | `pointmaze-giant-stitch-v0` | S | 0±0 | 0±0 | 0±0 | 50±8 | 0±0 | 0±0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 98.0±1.4 | 68±3 | — | E |
| 8 | `pointmaze-teleport-stitch-v0` | S | 31±9 | 44±2 | 25±3 | 9±5 | 4±3 | 34±4 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | A |
| 9 | `antmaze-medium-navigate-v0` | S | 29±4 | 72±8 | 71±4 | 88±3 | 95±1 | 96±1 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 81.1±4.7 | — | 100±0 | A |
| 10 | `antmaze-large-navigate-v0` | S | 24±2 | 16±5 | 34±4 | 75±6 | 83±4 | 91±2 | 94 | 2 | 84 | 76 | 17 | 76 | 71 | 65 | 88 | 90 | 61 | 58 | 36 | 81 | 83 | 83 | 83 | 0 | 83 | 88±7 | 57.2±3.8 | — | 98±6 | E |
| 11 | `antmaze-giant-navigate-v0` | S | 0±0 | 0±0 | 0±0 | 14±3 | 16±3 | 65±5 | 57 | 0 | 1 | 0 | 0 | 0 | 4 | 3 | 16 | 24 | 3 | 2 | 1 | 18 | 12 | 1 | 0 | 0 | 37 | 56±9 | 5.4±2.4 | — | 94±9 | A |
| 12 | `antmaze-teleport-navigate-v0` | S | 26±3 | 39±3 | 35±5 | 35±5 | 53±2 | 42±3 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | A |
| 13 | `antmaze-medium-stitch-v0` | S | 45±11 | 44±6 | 29±6 | 59±7 | 53±6 | 94±1 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 53.0±8.7 | 96±2 | — | A |
| 14 | `antmaze-large-stitch-v0` | S | 3±3 | 18±2 | 7±2 | 18±2 | 11±2 | 67±5 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 30.6±4.6 | 86±2 | — | A |
| 15 | `antmaze-giant-stitch-v0` | S | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | 2±2 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 32.7±6.6 | 65±3 | — | A |
| 16 | `antmaze-teleport-stitch-v0` | S | 31±6 | 39±3 | 17±2 | 24±5 | 31±4 | 36±2 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | A |
| 17 | `antmaze-medium-explore-v0` | S | 2±1 | 19±3 | 13±2 | 1±1 | 3±2 | 37±10 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 81±2 | — | A |
| 18 | `antmaze-large-explore-v0` | S | 0±0 | 10±3 | 0±0 | 0±0 | 0±0 | 4±5 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 22±7 | 66.7±8.3 | 27±1 | — | A |
| 19 | `antmaze-teleport-explore-v0` | S | 2±1 | 32±2 | 7±3 | 2±2 | 20±2 | 34±15 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | A |
| 20 | `humanoidmaze-medium-navigate-v0` | S | 8±2 | 24±2 | 27±2 | 21±8 | 60±4 | 89±2 | 69 | 39 | 60 | 68 | 24 | 60 | 42 | 62 | 83 | 82 | 53 | 22 | 86 | 67 | 65 | 59 | 34 | 44 | 93 | — | 50.5±5.5 | — | — | E |
| 21 | `humanoidmaze-large-navigate-v0` | S | 1±0 | 2±1 | 2±1 | 5±1 | 24±4 | 49±4 | 17 | 0 | 5 | 9 | 0 | 5 | 6 | 6 | 0 | 6 | 3 | 3 | 24 | 11 | 12 | 2 | 1 | 1 | 39 | — | 8.4±3.6 | — | — | A |
| 22 | `humanoidmaze-giant-navigate-v0` | S | 0±0 | 0±0 | 0±0 | 1±0 | 3±2 | 12±4 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.5±2.0 | — | — | A |
| 23 | `humanoidmaze-medium-stitch-v0` | S | 29±5 | 12±2 | 12±3 | 18±2 | 36±2 | 88±2 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 20.2±6.8 | 91±1 | — | A |
| 24 | `humanoidmaze-large-stitch-v0` | S | 6±3 | 1±1 | 0±0 | 3±1 | 4±1 | 28±3 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2.0±0.7 | 72±3 | — | A |
| 25 | `humanoidmaze-giant-stitch-v0` | S | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | 3±2 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.2±0.5 | 67±4 | — | A |
| 26 | `antsoccer-arena-navigate-v0` | S | 5±1 | 47±3 | 50±2 | 8±2 | 23±2 | 58±2 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | A |
| 27 | `antsoccer-medium-navigate-v0` | S | 2±0 | 4±1 | 7±1 | 2±2 | 3±1 | 13±2 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | E |
| 28 | `antsoccer-arena-stitch-v0` | S | 24±8 | 21±3 | 2±0 | 1±1 | 1±0 | 15±1 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 69±3 | — | A |
| 29 | `antsoccer-medium-stitch-v0` | S | 2±1 | 1±0 | 0±0 | 0±0 | 0±0 | 4±1 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 17±3 | — | A |
| 30 | `visual-antmaze-medium-navigate-v0` | V | 11±2 | 22±2 | 11±1 | 0±0 | 94±1 | 93±4 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 31 | `visual-antmaze-large-navigate-v0` | V | 4±0 | 5±1 | 4±1 | 0±0 | 84±1 | 53±9 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 3.6±0.6 | — | — | X |
| 32 | `visual-antmaze-giant-navigate-v0` | V | 0±0 | 1±1 | 0±0 | 0±0 | 47±2 | 6±4 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.2±0.3 | — | — | X |
| 33 | `visual-antmaze-teleport-navigate-v0` | V | 5±1 | 8±1 | 6±1 | 6±3 | 48±2 | 37±2 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 34 | `visual-antmaze-medium-stitch-v0` | V | 67±4 | 6±2 | 2±0 | 0±0 | 69±2 | 87±2 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 35 | `visual-antmaze-large-stitch-v0` | V | 24±3 | 1±1 | 0±0 | 1±1 | 11±3 | 28±2 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.0±0.0 | — | — | X |
| 36 | `visual-antmaze-giant-stitch-v0` | V | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.0±0.0 | — | — | X |
| 37 | `visual-antmaze-teleport-stitch-v0` | V | 32±3 | 1±1 | 1±0 | 1±2 | 32±6 | 37±4 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 38 | `visual-antmaze-medium-explore-v0` | V | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.0±0.0 | — | — | X |
| 39 | `visual-antmaze-large-explore-v0` | V | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.0±0.0 | — | — | X |
| 40 | `visual-antmaze-teleport-explore-v0` | V | 0±0 | 0±0 | 0±0 | 0±0 | 1±0 | 19±8 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 41 | `visual-humanoidmaze-medium-navigate-v0` | V | 0±0 | 0±0 | 0±0 | 0±0 | 1±0 | 0±0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.0±0.0 | — | — | X |
| 42 | `visual-humanoidmaze-large-navigate-v0` | V | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 43 | `visual-humanoidmaze-giant-navigate-v0` | V | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 44 | `visual-humanoidmaze-medium-stitch-v0` | V | 1±0 | 0±0 | 0±0 | 0±0 | 1±0 | 0±0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.0±0.0 | — | — | X |
| 45 | `visual-humanoidmaze-large-stitch-v0` | V | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 46 | `visual-humanoidmaze-giant-stitch-v0` | V | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 47 | `cube-single-play-v0` | S | 6±2 | 53±4 | 68±6 | 5±1 | 19±2 | 15±3 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 98±6 | E |
| 48 | `cube-double-play-v0` | S | 1±1 | 36±3 | 40±5 | 1±0 | 10±2 | 6±2 | 9 | 0 | 47 | 46 | 2 | 38 | 41 | 45 | 35 | 33 | 74 | 40 | 11 | 64 | 65 | 65 | 32 | 48 | 23 | 45±7 | — | — | 22±11 | E |
| 49 | `cube-triple-play-v0` | S | 1±1 | 1±0 | 3±1 | 0±0 | 4±1 | 3±1 | 1 | 0 | 3 | 3 | 0 | 8 | 8 | 8 | 5 | 6 | 1 | 2 | 0 | 3 | 3 | 5 | 2 | 8 | 4 | 37±8 | 4±2 | — | 0±0 | A |
| 50 | `cube-quadruple-play-v0`<sup>*</sup> | S | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | 9 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 3 | 19 | 2 | 5 | 2 | 3 | 14 | 6 | 0 | 0 | 51 | — | — | — | 0±0 | A |
| 51 | `cube-single-noisy-v0` | S | 8±3 | 71±9 | 99±1 | 25±6 | 38±2 | 41±6 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | A |
| 52 | `cube-double-noisy-v0` | S | 1±1 | 14±3 | 23±3 | 3±1 | 2±1 | 2±1 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | A |
| 53 | `cube-triple-noisy-v0` | S | 1±1 | 9±1 | 2±1 | 1±0 | 3±1 | 2±1 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 22±7 | — | — | — | A |
| 54 | `cube-quadruple-noisy-v0` | S | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | A |
| 55 | `scene-play-v0` | S | 5±1 | 42±4 | 51±4 | 5±1 | 19±2 | 38±3 | 65 | 50 | 98 | 78 | 38 | 38 | 74 | 88 | 68 | 78 | 99 | 62 | 84 | 97 | 95 | 97 | 94 | 61 | 89 | 80±4 | 52±4 | — | — | E |
| 56 | `scene-noisy-v0` | S | 1±1 | 26±5 | 26±2 | 9±2 | 1±1 | 25±4 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 40±5 | — | — | — | A |
| 57 | `puzzle-3x3-play-v0` | S | 2±0 | 6±1 | 95±1 | 1±0 | 3±1 | 12±2 | 79 | 0 | 56 | 70 | 3 | 48 | 100 | 90 | 68 | 57 | 87 | 99 | 100 | 100 | 99 | 100 | 82 | 100 | 100 | — | — | — | — | R |
| 58 | `puzzle-4x4-play-v0`<sup>*</sup> | S | 0±0 | 13±2 | 26±3 | 0±0 | 0±0 | 7±2 | 0 | 15 | 0 | 5 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | 34 | 0 | 0 | 6 | 39 | 0 | 36 | 37 | 70±5 | — | — | — | A |
| 59 | `puzzle-4x5-play-v0` | S | 0±0 | 7±1 | 14±1 | 0±0 | 1±0 | 4±1 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 20±0 | 13±3 | — | — | A |
| 60 | `puzzle-4x6-play-v0` | S | 0±0 | 10±2 | 12±1 | 0±0 | 4±1 | 3±1 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 20±2 | 10±0 | — | — | A |
| 61 | `puzzle-3x3-noisy-v0` | S | 1±0 | 42±19 | 94±3 | 0±0 | 30±6 | 51±11 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | A |
| 62 | `puzzle-4x4-noisy-v0` | S | 0±0 | 20±3 | 29±7 | 0±0 | 0±0 | 16±4 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 56±7 | — | — | — | A |
| 63 | `puzzle-4x5-noisy-v0` | S | 0±0 | 19±0 | 19±0 | 0±0 | 3±2 | 5±1 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | A |
| 64 | `puzzle-4x6-noisy-v0` | S | 0±0 | 17±2 | 18±2 | 0±0 | 6±3 | 2±1 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 19±1 | — | — | — | A |
| 65 | `visual-cube-single-play-v0` | V | 5±1 | 60±5 | 30±5 | 41±15 | 31±15 | 89±0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 66 | `visual-cube-double-play-v0` | V | 1±1 | 10±2 | 1±1 | 5±0 | 2±1 | 39±2 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 67 | `visual-cube-triple-play-v0` | V | 15±2 | 14±2 | 15±1 | 16±1 | 17±2 | 21±0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 68 | `visual-cube-quadruple-play-v0` | V | 8±1 | 0±0 | 7±1 | 5±1 | 4±1 | 14±1 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 69 | `visual-cube-single-noisy-v0` | V | 14±3 | 75±3 | 48±3 | 10±5 | 39±30 | 99±0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 70 | `visual-cube-double-noisy-v0` | V | 5±1 | 17±4 | 22±2 | 6±2 | 6±3 | 59±3 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 71 | `visual-cube-triple-noisy-v0` | V | 16±1 | 18±1 | 12±1 | 9±4 | 16±1 | 23±2 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 72 | `visual-cube-quadruple-noisy-v0` | V | 9±0 | 0±0 | 2±2 | 0±0 | 8±2 | 12±8 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 73 | `visual-scene-play-v0` | V | 12±2 | 25±3 | 12±2 | 10±1 | 11±2 | 49±4 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 74 | `visual-scene-noisy-v0` | V | 13±2 | 23±2 | 12±4 | 2±0 | 15±2 | 50±1 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 75 | `visual-puzzle-3x3-play-v0` | V | 0±0 | 21±1 | 1±2 | 1±1 | 0±0 | 73±8 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 76 | `visual-puzzle-4x4-play-v0` | V | 10±1 | 60±5 | 16±4 | 0±0 | 10±6 | 60±41 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 77 | `visual-puzzle-4x5-play-v0` | V | 5±2 | 17±1 | 7±2 | 0±0 | 6±1 | 13±9 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 78 | `visual-puzzle-4x6-play-v0` | V | 2±1 | 15±1 | 2±1 | 0±0 | 3±1 | 9±6 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 79 | `visual-puzzle-3x3-noisy-v0` | V | 1±1 | 20±0 | 26±4 | 0±0 | 1±1 | 70±6 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 80 | `visual-puzzle-4x4-noisy-v0` | V | 7±3 | 47±3 | 49±7 | 0±0 | 6±2 | 84±4 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 81 | `visual-puzzle-4x5-noisy-v0` | V | 6±1 | 14±10 | 19±0 | 0±0 | 7±1 | 14±10 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 82 | `visual-puzzle-4x6-noisy-v0` | V | 2±1 | 12±8 | 17±1 | 0±0 | 2±1 | 14±2 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 83 | `powderworld-easy-play-v0` | P | 0±0 | 99±1 | 93±5 | 12±2 | 22±5 | 33±9 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 84 | `powderworld-medium-play-v0` | P | 1±1 | 50±4 | 16±5 | 3±1 | 1±1 | 22±14 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |
| 85 | `powderworld-hard-play-v0` | P | 0±0 | 4±3 | 0±0 | 0±0 | 0±0 | 1±1 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | X |

## What the structural-method search found

RQL's 19 experimental columns contain no learned environment world model and no tree/search
planner. RQL and BDPO expand the MDP over iterative *action-generation* steps; that is not an
environment dynamics model or a planning tree. The four extra structural columns were
chosen because they have direct OGBench-named evidence:

- [Occupancy Reward Shaping (ORS)](https://arxiv.org/abs/2604.20627) learns a flow occupancy
  model and uses it to shape rewards for a GCIQL backbone. It reports 13 canonical datasets.
  Values are mean ± 95% bootstrap-CI half-width over eight seeds, not standard deviations.
  Training runs to task-dependent convergence (1M–6M policy iterations), with per-dataset
  tuning.
- [Test-Time Graph Search (TTGS)](https://arxiv.org/abs/2510.07257) samples offline states,
  builds a value-derived directed graph, and uses Dijkstra search for adaptive subgoals. The
  table fixes the backbone to GCIQL and reports 31 datasets. Values are mean ± SD over eight
  seeds; planner hyperparameters `(τ, T)` are tuned by dataset. ORS and TTGS use the direct
  GCRL success metric, but their reruns and tuning differ from the original OGBench table;
  compare each against the paired baselines in its own paper, not by cross-paper boldfacing.
- [CompDiffuser](https://arxiv.org/abs/2503.05153) is a diffusion trajectory planner with an
  inverse-dynamics controller. It reports 13 OGBench-named settings, mean ± SD over five
  seeds and five fixed evaluation tasks per dataset × 20 episodes per task, using
  autoregressive sampling with replanning.
  Its evaluation changes episode limits on hard environments and uses supplied noisy
  start/goal files; the two AntSoccer cells use the paper's 17-D variant. Hence `†`.
- [Monte Carlo Tree Diffusion (MCTD)](https://arxiv.org/abs/2502.07202) performs genuine
  MCTS—selection, expansion, denoising simulation, and backup—over diffusion subplans. It
  reports ten canonical-named rows for the consistent no-replanning base variant, mean ± SD
  over ten seeds. The paper also reports `MCTD-Replanning` on two PointMaze stitch rows; those
  values are intentionally not mixed into this column. The base experiments remove start/goal
  noise, alter state/features and some horizons, select the best of three AntMaze execution
  trials, and use task-specific executors/guidance, so they are `†` and must not be ranked
  against stock OGBench values.

RQL also cites well-known trajectory/world-model work—
[Diffuser](https://arxiv.org/abs/2205.09991),
[Decision Diffuser](https://arxiv.org/abs/2211.15657),
[Guided Flows for Generative Modeling and Decision Making](https://arxiv.org/abs/2311.13443),
[Simple Hierarchical Planning with Diffusion](https://arxiv.org/abs/2401.02644),
[HDMI](https://proceedings.mlr.press/v202/li23ad.html),
[SynthER](https://arxiv.org/abs/2303.06614),
[Diffusion World Model](https://arxiv.org/abs/2402.03570),
[Policy-Guided Diffusion](https://arxiv.org/abs/2404.06356), and
[DIAMOND](https://arxiv.org/abs/2405.12399)—but their original papers report 0/85 native
OGBench datasets. They are documented here rather than padded into eight all-`—` columns.
MCTD is the canonical true tree-search representative selected here; its modified protocol
means it provides structural context, not a stock leaderboard entry. Later family variants,
[Fast-MCTD](https://arxiv.org/abs/2506.09498) and
[C-MCTD](https://arxiv.org/abs/2510.21361), were not added as near-duplicate columns.

## TreeWM evidence and launch gate

The current `E` rows are `pointmaze-{medium,large,giant}-stitch`,
`antmaze-large-navigate`, `humanoidmaze-medium-navigate`,
`antsoccer-medium-navigate`, `cube-{single,double}-play`, and `scene-play`.
Their audited results are in [RESULTS.md](./RESULTS.md). For the completed cross-family runs,
the best recorded arm reaches 0% on AntMaze, HumanoidMaze, AntSoccer, and cube-double;
11% on scene at 300k; and 20% on cube-single at 1M. These are one-training-seed,
goal-conditioned five-task evaluations and must not be compared directly with the RQL
single-task cells. `puzzle-3x3-play` is `R`: its adapter exists, but there is no completed
formal result of record.

No new 85-environment training fleet was launched while constructing this literature table.
That would currently produce invalid coverage: only ten exact domain adapters exist, the
35 visual datasets need an observation encoder, and Powderworld needs a discrete-action/grid
path. The standalone/final evaluator must also be made domain-aware before any formal sweep.
After those gates, the safe first launch is the 47 state datasets, with family probes and
RAM-aware scheduling before filling all A100 slots. Based on the measured Wave-4 jobs, a
47-dataset × two-arm × one-seed state sweep to 300k is roughly 2,150 slot-hours (about 11
days at eight continuously occupied slots) before family-specific variance; visual and
Powderworld cost cannot be estimated from the present implementation.

## Method and source index

### OGBench-native goal-conditioned baselines

- `GCBC`: goal-conditioned behavioral cloning; implemented from
  [Learning to Reach Goals via Iterated Supervised Learning](https://arxiv.org/abs/1912.06088)
  and [Learning Latent Plans from Play](https://arxiv.org/abs/1903.01973).
- `GCIVL`, `GCIQL`: goal-conditioned adaptations of
  [Implicit Q-Learning](https://arxiv.org/abs/2110.06169), as specified by OGBench.
- `QRL`: [Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning](https://arxiv.org/abs/2304.01203).
- `CRL`: [Contrastive Learning as Goal-Conditioned Reinforcement Learning](https://arxiv.org/abs/2206.07568).
- `HIQL`: [Offline Goal-Conditioned RL with Latent States as Actions](https://arxiv.org/abs/2307.11949).
- Aggregate source and runnable reference code:
  [OGBench paper](https://arxiv.org/abs/2410.20092),
  [v1.2.1 release](https://github.com/seohongpark/ogbench/releases/tag/v1.2.1), and
  [official repository](https://github.com/seohongpark/ogbench).

### RQL comparison methods

- `ReBRAC`: [Revisiting the Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/abs/2305.09836).
- `FQL`, `FBRAC`, and the action-chunked `FAWAC`/`IFQL` implementations:
  [Flow Q-Learning](https://arxiv.org/abs/2502.02538). Their underlying objectives trace to
  [AWAC](https://arxiv.org/abs/2006.09359) and
  [IDQL](https://arxiv.org/abs/2304.10573), respectively.
- `BAM`, `CGQL`, `CGQL-M`, `CGQL-L`, `FEdit`, `QAM`, `QAM-F`, and `QAM-E`:
  [Q-Learning with Adjoint Matching](https://arxiv.org/abs/2601.14234).
- `DAC`: [Diffusion Actor-Critic](https://arxiv.org/abs/2405.20555).
- `QSM`: [Q-Score Matching](https://arxiv.org/abs/2312.11752).
- `DSRL`: [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799).
- `BDPO`: [Behavior-Regularized Diffusion Policy Optimization](https://arxiv.org/abs/2502.04778).
- `TFQL` and `RQL`: [Reversal Q-Learning](https://arxiv.org/abs/2606.17551), with
  [official code](https://github.com/aoberai/rql). `ReBRAC`–`QAM-E` were inherited from the
  [QAM experiment archive](https://github.com/ColinQiyangLi/qam/tree/main/exp_data);
  `BDPO`, `TFQL`, and `RQL` were added/evaluated in RQL. The RQL appendix table is the
  authority for the aggregate values copied here.

### Structural additions

- `ORS`: [paper](https://arxiv.org/abs/2604.20627),
  [code](https://github.com/aravindvenu7/occupancy_reward_shaping).
- `GCIQL+TTGS`: [paper](https://arxiv.org/abs/2510.07257),
  [project](https://ktolnos.github.io/ttgs/), [code](https://github.com/ktolnos/TTGS).
- `CompDiff`: [paper](https://arxiv.org/abs/2503.05153),
  [project](https://comp-diffuser.github.io/),
  [code](https://github.com/devinluo27/comp_diffuser_release).
- `MCTD`: [paper](https://arxiv.org/abs/2502.07202),
  [code](https://github.com/ahn-ml/mctd).
- `TreeWM`: this repository; see [RESULTS.md](./RESULTS.md) for the evidence currently on disk.

## Reproducibility notes

- The 85 OGBench values were transcribed from paper Table 2 plus the 18 noisy-manipulation
  rows in Appendix Table 5. OGBench does not publish an aggregate CSV; its runner writes
  per-run `eval.csv`, but those official run files are not bundled.
- The RQL cells are the aggregate point estimates in its official arXiv TeX source, not
  values inferred from plots. `ReBRAC`–`QAM-E` were inherited from QAM; `BDPO`, `TFQL`, and
  `RQL` were added/evaluated in RQL.
- ORS, TTGS, CompDiffuser, and MCTD values were taken from their primary paper tables. Their
  uncertainty definitions differ, as documented above.
- This file uses `—` for absent evidence. Never convert it to `0` in downstream parsing.
