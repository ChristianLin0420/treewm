# 09 — Cross-family OGBench screen

Eight PointMaze cycles established two surviving phenomena and a long list of rejected
components. This cycle asks whether the two survivors are properties of *recursive
executable-future prediction* or properties of *PointMaze*:

1. Recursive multimodal prediction beats matched Flat-K prediction.
2. Recursive control has an interior temporal-edge optimum.

Deliberately **not** revisited: controllability `q`, learned frontier allocation, learned
horizon selection, novelty search. All were falsified in cycles 01–08 and none returns here.

## Waves

| wave | question | design |
|---|---|---|
| 1 | Which dynamics families are discriminative at all? | 7 envs × {FlatKWM, RecursiveWM}, K=4, fixed h=16, 1 seed, staged 20k/50k/100k |
| 2 | Does an interior h* appear outside PointMaze? | promoted envs × h ∈ {8,16,32,64}, ladder extended if the optimum lands on an endpoint |
| 3 | Does it survive seeds and budget? | best 2–3 envs × {Flat, best h, h±1 rung} × ≥3 seeds × budgets [16,32,64,128,256] |

Wave 1's purpose is **not** SOTA. It is to find a second and ideally third family where
the thesis can be tested cleanly.

## The matched comparison

Both arms share encoder size, hidden size, optimizer, steps, dataset, branch/action head,
residual latent dynamics, decoded goal scoring, evaluation episodes, seed and node budget.
The only intended conceptual difference is recursion. Resource settings
(`retrieval_pool`, `max_train_anchors`, `num_workers`) are applied identically to both
arms so they cannot become a confound.

## Why a domain adapter layer exists

PointMaze let us treat `obs[:2]` as the quantity the goal constrains. That is wrong for
five of the seven environments, and wrong in the worst possible way — silently. Scoring
antsoccer by the *agent's* position produces a complete, plausible set of numbers that
measure the wrong thing, because the task is to move the **ball**
(`norm(ball_xy - goal_xy) <= tol`, `ball_xy = qpos[-7:-5]` → `obs[15:17]`).

`treewm/evaluation/domains.py` records, per environment, which observation dims the goal
constrains and what partial progress means. Every index is verified against the
environment's own `privileged/*` info in `tests/test_domains.py` rather than trusted from
a reading of OGBench's source. Success is always taken from the environment, never
reimplemented.

Partial progress matters as much as success: three AntMaze cycles produced nothing
interpretable because a zero success rate was the only signal available. Here a floor-bound
environment still reports subgoal fractions, object displacement and Hamming distance.

## Difficulty measures are domain-specific by design

No shared geometric "horizon" is imposed across families — that would be a category error
for a puzzle. Instead:

| domain | difficulty |
|---|---|
| cube | number of cubes that must actually move |
| scene | atomic subgoals required by the task |
| puzzle | Hamming distance to the target board (lower bound on button presses) |
| locomotion | geodesic task distance, but only where success is non-saturated |

## Environments

| dataset | obs | act | goal constrains | family |
|---|---|---|---|---|
| `antmaze-large-navigate-v0` | 29 | 8 | agent xy `obs[0:2]` | locomotion |
| `humanoidmaze-medium-navigate-v0` | 69 | 21 | agent xy `obs[0:2]` | locomotion |
| `antsoccer-medium-navigate-v0` | 42 | 8 | **ball** xy `obs[15:17]` | locomotion |
| `cube-single-play-v0` | 28 | 5 | cube pos `obs[19:22]` | manipulation |
| `cube-double-play-v0` | 37 | 5 | 2 cube positions | manipulation |
| `scene-play-v0` | 40 | 5 | cube + 2 buttons + drawer + window | manipulation |
| `puzzle-3x3-play-v0` | 55 | 5 | 9 binary buttons | puzzle |

### powderworld is excluded from Wave 1, and why

`powderworld-easy-play-v0` is a different **model class**, not a goal-metric variation:
observations are `Box(0,255,(32,32,6),uint8)` grids, actions are `Discrete(8)`, and the
task is specified as an *action sequence* plus tolerance rather than a goal state. Every
other environment here is flat-float-observation with continuous `Box` actions, and the
whole screen rests on the two arms being capacity-matched. Supporting powderworld needs a
discrete action head with cross-entropy where the rest use continuous chunk MSE, which is
a separate change that should not ride along inside a comparison whose validity depends on
matching. It is deferred rather than dropped.

## Running

```bash
python scripts/fleet.py --probe-only            # calibrate VRAM/RSS/throughput per (env, arm)
python scripts/fleet.py --wave 1 --steps 20000  # probe, bin-pack, run, back off
python scripts/wave_report.py                   # per-environment verdicts + shortlist
```

Live status: `fleet_status.json` (machine-readable) and the periodic terminal table.
