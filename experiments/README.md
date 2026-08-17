# Experiment cycles

Each directory is one cycle of work, holding everything needed to interpret it:

```
NN-name/
  runs/      training runs, one dir per grid (checkpoints + TensorBoard)  [not versioned, ~29 GB]
  results/   metrics JSON and figures produced by the analysis scripts    [versioned]
  logs/      driver and analysis stdout -- the numeric tables             [versioned]
  REPORT.md  the written report, where one exists
```

Cycles are numbered in the order they were run, because each was designed to test what the
previous one turned up. Read them in order and the project is a narrative; read `results/`
alone and it is a pile of JSON.

`../RESULTS.md` holds the findings of record. This file is the map from question to artifact.

| # | cycle | question | outcome |
|---|---|---|---|
| 01 | `main-grid` | Does allocated tree prediction beat single / flat / fixed-tree at matched node budget? | Multimodality and recursion **supported**; controllability-aware `q` and learned allocation **not** |
| 02 | `novelty-target` | Does a min-novelty target `G*(n\|T) = min_j d_q(q_n,q_j)` fix the degenerate gain head? | No. Learned q-novelty and z-novelty both reduce to a fixed rule; neither beats random |
| 03 | `reliability` | Is the null a depth artifact? Depth-limited novelty sweep + oracle-grounded leaf selection | No. The null survives; coverage anti-correlates with success (ρ = −0.77) |
| 04 | `decoded-goal` | Does scoring leaves by decoded physical goal distance beat latent distance? | **Yes** — roughly doubles success. Became the default. Expansion policy still no better than random |
| 05 | `design-space` | Broad controlled search over 8 architectural axes (A–H), screen then promote | No axis rescues learned allocation. See `REPORT.md` |
| 06 | `q1-q2-q3` | AntMaze at 50k; learned vs random vs fixed horizon; ancestor × root_quota factorial | AntMaze null. Learned horizon collapses onto the longest option 87.6% of the time |
| 07 | `horizon-antmaze` | Fixed-horizon sweep h ∈ {8,12,16,20,24,32}; AntMaze at 300k | **Interior optimum** in h with a measured reach/error tradeoff. AntMaze null again at 300k |
| 08 | `scaling` | Does the recursion advantage, and h*, scale with required horizon across layouts? | Advantage **replicates** (pooled ρ = +0.913, 3 layouts). h*(d) **does not** — see below |

## Where the current claim rests

Cycle 08 is the live one. It added `pointmaze-large` and then `pointmaze-giant`, the latter
chosen specifically to decorrelate task distance from corridor straightness — `medium` is
short+twisty and `large` is long+straight, so the two are collinear and cannot separate the
competing explanations. `giant` is long+twisty.

That test **falsified** the h*(distance) scaling law that cycles 07–08 had made look like the
project's best result. Artifacts:

- `08-scaling/results/difficulty/scaling_report.png` — the six panels
- `08-scaling/logs/hstar_model.log` — leave-one-layout-out fit; every model scores negative
- `08-scaling/logs/diffcurve_{pm,large2,giant}.log` — the per-bin success tables

## Reproducing a cycle

Runs are not versioned, so a cycle must be retrained before its analysis can be re-run.
Cycle 08, end to end:

```bash
python scripts/run_screen.py --recipes H8 H16 H20 H32 H48 H64 Fh20 \
    --dataset pointmaze_giant_stitch --seeds 0 1 2 --steps 20000 \
    --run-root experiments/08-scaling/runs/giant
python scripts/difficulty_curve.py --runs experiments/08-scaling/runs/giant \
    --tag pointmaze_giant --bins 1 3 5 7 9 12 16 21 26 32 \
    --per-bin 12 --episodes 2 --budgets 64 128 --out experiments/08-scaling/results/difficulty
python scripts/scaling_report.py --out experiments/08-scaling/results/difficulty
python scripts/hstar_model.py    --out experiments/08-scaling/results/difficulty
```

## A note on the older driver scripts

`scripts/*_stage.sh` are one-shot drivers written during their cycle and reference the old
flat paths (`runs_large`, `results_difficulty`). They are kept as a record of exactly what was
executed, not as reusable entry points — pass explicit `--runs` / `--out` instead.
