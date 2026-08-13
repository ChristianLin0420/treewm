# TreeWM

**Budgeted controllability-aware reachability trees as world models.**

The hypothesis under test:

> A useful world model should allocate finite prediction compute over
> controllability-distinct executable futures, rather than predicting a single
> trajectory or a flat set of successor states.

This codebase exists to *test* that claim, not to make TreeWM win. It is built so the
causal chain can be falsified one link at a time:

```
multimodal prediction -> recursive prediction -> controllability-aware support -> adaptive compute
   FlatKWM > SingleWM     FixedTree > FlatKWM      q-coverage > z-coverage      TreeWM > FixedTree
```

---

## Quickstart

```bash
conda create -n treewm -c conda-forge --override-channels python=3.11 -y
conda activate treewm
pip install -r requirements.txt

python -c "import ogbench; ogbench.download_datasets([
    'pointmaze-medium-navigate-v0','pointmaze-medium-stitch-v0','pointmaze-large-navigate-v0'])"

pytest tests/ -q
```

Train, evaluate, sweep:

```bash
torchrun --nproc_per_node=2 scripts/train.py \
    experiment=pointmaze_treewm \
    seed=0

python scripts/eval.py \
    checkpoint=runs/pointmaze-medium/treewm/20260812_seed0/checkpoints/latest.pt \
    tree.node_budget=64

python scripts/sweep_budget.py \
    checkpoint=runs/pointmaze-medium/treewm/20260812_seed0/checkpoints/latest.pt \
    budgets='[16,32,64,128,256]'

python scripts/visualize_tree.py \
    checkpoint=runs/pointmaze-medium/treewm/20260812_seed0/checkpoints/latest.pt

tensorboard --logdir runs/
```

The full arm x seed x dataset grid:

```bash
python scripts/run_grid.py --arms all --seeds 0 1 2 --datasets navigate stitch --steps 20000
```

### A note on DDP

`torchrun --nproc_per_node=2` works and is tested. But PointMaze models are ~3.7M
parameters and the bottleneck is future-set retrieval in the dataloader, not matmul
throughput, so **two independent single-GPU jobs beat one 2-GPU DDP job** at this scale.
`scripts/run_grid.py` therefore runs one job per GPU. DDP is kept working for AntMaze and
for a future pixel encoder.

---

## The seven arms

Every arm is the *same* network with the same components and matched capacity (3.67M
params; FlatKWM 3.70M for its extra branch tokens). They differ in three settings:

| arm | K | max depth | frontier scorer | what it isolates |
|---|---|---|---|---|
| `singlewm` | 1 | unbounded | — | single-future prediction |
| `flatkwm` | 256 | 1 | — | multimodality without recursion |
| `fixedtreewm` | 4 | unbounded | `bfs` | recursion without allocation |
| `randomtreewm` | 4 | unbounded | `random` | allocation without a signal |
| `uncertaintytreewm` | 4 | unbounded | `uncertainty` | allocation on a *different* signal |
| `heuristictreewm` | 4 | unbounded | `heuristic` | **adaptive but unlearned** |
| `treewm` | 4 | unbounded | `learned` | learned coverage allocation |

### Budget parity is enforced, not assumed

Every arm emits **exactly** `node_budget` chunk-nodes; only the *shape* of the spend
differs. At N=64:

```
SingleWM    o-o-o-...-o        depth 63, breadth 1
FlatKWM     o<63 branches>     depth 1,  breadth 63
FixedTree   o -> 4 -> 16 -> …  BFS, uniform
TreeWM      o -> 4 -> adaptive learned allocation
```

Without this, "TreeWM beats FlatKWM at N=64" would just mean "16x more nodes wins".
`tests/test_tree.py::test_node_budget_enforced_for_every_arm` asserts it for all seven
arms at three budgets, and `sweep_budgets` re-asserts it at evaluation time.

**`heuristictreewm` is not in the original spec** and was added deliberately. BFS and
random are non-adaptive; uncertainty keys off a different signal. None of them separates
*adaptive* from *learned*. If TreeWM does not beat a greedy q-novelty rule with no
trained parameters, the learned gain head is not the mechanism — which is a real result,
and a cheap one to know.

---

## What a "mode" is

Everything about support, coverage, redundancy and rare-mode recall presupposes that the
local future set decomposes into discrete modes. PointMaze is deterministic with a
continuous action space, so the reachable set is a *continuum* — the partition has to be
defined, not assumed.

For an anchor state, `treewm/data/future_sets.py`:

1. retrieves nearby dataset states (normalised raw state, k-d tree),
2. borrows their continuations, each with its own duration,
3. clusters the resulting **endpoints** into modes (average linkage, distance threshold).

That yields two separate, non-interchangeable targets:

```
support (kappa)   1 per cluster, regardless of size    -> mode-balanced loss
mass    (rho)     cluster frequency                    -> frequency-aware loss
```

Given retrieved futures splitting `left 80% / straight 15% / right 5%`, the targets are
`support = [1,1,1]` and `mass = [.80,.15,.05]`. Truncation to `max_modes` is **random,
never mass-ordered** — dropping the smallest clusters would delete exactly the rare-but-
valid modes the project exists to preserve.

Tuned on pointmaze-medium: ~3 modes per anchor (max 5), predicted horizons spread across
all five candidates. Maze topology is used **only** to validate clusters at evaluation
and to colour plots; no loss ever reads it.

---

## The expansion-gain target is data-grounded

The spec suggests supervising `G` with "a more exhaustive teacher expansion". That
bootstraps a self-graded signal off noise: at initialisation the teacher *is* the model,
and coverage would be measured in the model's own q-space. Instead:

```
G(n | T) = | { endpoint_cell[c] : c in dataset_neighbours(z_n) } \ cells(T) |
```

"How many new regions of the world become reachable if this node is expanded", where
reachability comes from the offline data and regions from a global quantiser
(`treewm/evaluation/coverage.py`). Both are model-independent, and the target exists at
step 1. The same quantiser defines the coverage metric, so a model cannot look good by
redefining the space it is scored in.

---

## Diagnostics that decide whether the story is alive

Read these first; they are logged from step 1.

**1. Is the split discriminative?** `eval/success_rate` for `singlewm` at `node_budget=16`
on the hard split. If it is already ≥0.9, the ladder is unmeasurable and the split must be
re-cut before any comparison means anything.

**2. Does q beat z — dimension-matched?**

```
control/retrieval_precision_q
control/retrieval_precision_z
control/retrieval_precision_random_proj     <- the control
control/q_advantage_over_z
control/q_advantage_over_random_proj
```

`q = C(z)` is a deterministic function of `z` and therefore **cannot** hold more
information about the future than `z` does. Any q-advantage is a claim about metric
geometry, so it is compared against a frozen random projection of `z` down to `q_dim`. If
q beats z but not the random projection, the apparent win is dimensionality.

**3. Is *learned* the mechanism?** `treewm` vs `heuristictreewm` at matched budget.

Plus the major scientific diagnostic of section 19-H,
`control/branching_future_diversity_corr`: does the effective branching factor track
*empirical* local future diversity? The reference is data-derived. A geometric version
(`control/branching_junction_degree_corr`) is reported as a secondary check — geometry
only partly determines future diversity, measured at ρ≈0.32 on pointmaze-medium before
training, so a weak value there is expected and is not evidence against the hypothesis.

---

## Results (20k steps, 3 seeds, PointMaze)

Full grid: 7 arms x 3 seeds x {navigate, stitch}, 42 runs, 0 failures, 471 min on 2xA100.
Tables in `results/summary.md`; curves in `results/*.png`.

### Success rate vs matched node budget (mean of 3 seeds x 50 episodes)

`pointmaze-medium-stitch` (the hard split), AUC = mean over budgets 16-256:

| arm | 16 | 32 | 64 | 128 | 256 | AUC |
|---|---|---|---|---|---|---|
| SingleWM | .000 | .000 | .000 | .000 | .000 | **.000** |
| FlatKWM | .040 | .020 | .040 | .040 | .080 | .044 |
| FixedTreeWM | .167 | .127 | .133 | .093 | .113 | .127 |
| RandomTreeWM | .133 | .140 | .187 | .233 | .253 | .189 |
| UncertaintyTreeWM | .153 | .153 | .160 | .120 | .073 | .132 |
| **HeuristicTreeWM** | .200 | .320 | .280 | .333 | .300 | **.287** |
| TreeWM | .147 | .200 | .193 | .193 | .140 | .175 |

### Coverage vs node budget (distinct regions reached, budget 256)

| arm | navigate | stitch |
|---|---|---|
| FlatKWM | 27.8 | 12.4 |
| FixedTreeWM | 41.2 | 28.6 |
| SingleWM | 39.5 | 37.7 |
| TreeWM | 50.1 | 53.6 |
| RandomTreeWM | 63.7 | 53.7 |
| UncertaintyTreeWM | 67.3 | 37.3 |
| **HeuristicTreeWM** | **90.8** | **75.4** |

### What the causal chain actually shows

```
multimodal prediction   -> SUPPORTED   SingleWM scores 0.000 on all 6 runs; FlatKWM > SingleWM
recursive prediction    -> SUPPORTED   tree arms >> FlatKWM (d = -1.97 on stitch)
controllability-aware q -> NOT SUPPORTED
adaptive compute        -> NOT SUPPORTED (learned allocation specifically)
```

**Links 1-2 hold decisively.** SingleWM never reaches a hard goal at any budget. FlatKWM
is barely better and has the *worst* coverage of any arm -- spending the whole budget at
depth 1 cannot reach beyond one chunk length. Recursion is what buys reach.

**Link 4 fails, and the control arm is what shows it.** TreeWM is beaten by random
frontier expansion on navigate and by the parameter-free q-novelty heuristic on stitch
(.175 vs .287 AUC). Coverage says the same thing more sharply: **TreeWM is trained to
predict marginal coverage gain and expand by it, yet an unlearned greedy novelty rule
achieves 1.8x (navigate) and 1.4x (stitch) more coverage per node.** It loses at the exact
objective it optimises.

Without `HeuristicTreeWM` the table would read "TreeWM > FixedTreeWM on both datasets ->
learned allocation matters". That conclusion is an artifact of a missing control.

**TreeWM anti-scales with budget.** Success falls from .267 (64 nodes) to .080 (256) on
navigate while RandomTreeWM climbs monotonically .127 -> .267. The learned head
concentrates expansion and doubles down as budget grows; random and novelty-driven
expansion keep spreading. `gain_rank_correlation` is a healthy +0.43..+0.53, so the head
*does* predict its target -- the target (retrieval-based marginal new-cell count) appears
to be the weak link, not the regression.

**Link 3 fails independently.** `q_advantage_over_z` and `q_advantage_over_random_proj`
are negative in **all 42 runs** (-0.06 to -0.28): q never beats z, nor a random projection
of z at matched dimension, at future-set retrieval. Since `q = C(z)` is deterministic this
was always a claim about metric geometry, and the geometry is not there. The
dimension-matched control is what makes this conclusion stick rather than being
dismissible as "fewer dimensions retrieve better".

**What did work:** `branching_future_diversity_corr` = **+0.42 (navigate) / +0.82
(stitch)** -- effective branching factor genuinely tracks empirical local future
diversity. But it is identical across all seven arms, so it is a property of the shared
branch network, not of any allocation policy.

### Caveats

- n=3 seeds. Only the SingleWM/FlatKWM gaps are comfortably outside noise; the
  TreeWM-vs-controls differences are suggestive (Cohen's d 0.9-1.7), not established.
- 20k steps, not trained to convergence.
- All tree arms share identical training and differ only in frontier scoring, so this
  isolates *frontier scoring*. It does not indict the tree representation itself --
  recursion clearly helps.
- Absolute success is low (<=0.35) for every arm; the hard split is hard.
- Coverage uses decoded latents, comparable across arms because encoder/decoder training
  is identical, but it is not a simulator ground-truth reachable set.


---

## Experiment 2 — novelty gain target (diagnostic rerun)

36 runs (6 arms x 3 seeds x {navigate, stitch}), 0 failures, 292 min. Gain target replaced
with `G*(n|T) = min_j d(q_n, q_j)`; architecture, planner, budgets, schedule and evaluation
unchanged. Tables in `results_novelty/`.

### The target change fixed coverage

Distinct regions at budget 256:

| arm | navigate | stitch |
|---|---|---|
| FixedTree | 43.5 | 31.4 |
| Random | 63.1 | 45.8 |
| **noveltyq (direct)** | **100.0** | **84.4** |
| learnedq | 79.1 | 67.9 |
| noveltyz (direct) | 87.4 | 82.8 |
| learnedz | 63.3 | 56.8 |
| *exp-1 TreeWM (retrieval target)* | *50.1* | *53.6* |

Learned allocation gained **+58% / +27%** over the retrieval target, and **no arm
anti-scales** -- coverage is monotone in budget for all six. Min-novelty also beats the
old mean-pool heuristic (100.0 vs 90.8). The gain target really was a large part of the
earlier failure.

### But coverage does not buy success

Success AUC over budgets 16-256, and its correlation with coverage across arms:

| arm | navigate AUC | stitch AUC |
|---|---|---|
| FixedTree | 0.189 | 0.137 |
| **Random** | **0.199** | **0.271** |
| noveltyq | 0.155 | 0.171 |
| learnedq | 0.172 | 0.140 |
| noveltyz | 0.167 | 0.123 |
| learnedz | 0.165 | 0.149 |

```
spearman(coverage@256, success AUC) = -0.77 (navigate),  -0.03 (stitch)
```

Random wins on both datasets with 40-50% *less* coverage than noveltyq, and is the only
arm that scales cleanly with budget on stitch (.107 -> .367).

### Why: novelty buys depth, and depth is where the model breaks

Executing every sampled tree node's root-to-node chunks in the simulator and comparing
against the model's decoded prediction (`scripts/grounded_coverage.py`, stitch, budget 256):

| arm | mean depth | mean error | predicted/actual regions |
|---|---|---|---|
| FixedTree | 3.58 | 1.07 | 1.03 |
| Random | 4.66 | 1.76 | 1.08 |
| noveltyq | 8.05 | 5.00 | 1.06 |
| learnedq | 9.06 | 3.25 | 1.10 |
| noveltyz | 8.08 | 3.42 | 1.07 |
| learnedz | 8.75 | 4.24 | 1.05 |

Open-loop error compounds with depth and is essentially arm-independent (all arms share
one world model): **d1 ~0.1, d4 ~1.3, d8 ~3.9, d12+ 4-13** world units, against a maze
corridor width of 4.0. Arms differ only in *how deep they choose to go*.

The region *count* is honest -- predicted/actual region ratio is 1.03-1.10 for every arm,
so the tree really does reach about as many distinct regions as it claims. What breaks is
the **node -> position mapping**: at depth 8 a node's predicted position is off by more
than a corridor width. The planner selects a node by latent distance to the goal and then
executes that node's path, so a deep node with a wrong mapping produces a wrong plan.

**Coverage and planning utility are in tension under an imperfect chunk-level model.**
Novelty-driven allocation converts budget into depth, depth into compounding error, and
the planner needs an accurate mapping more than it needs breadth of coverage. Shallow,
accurate allocation (random/BFS) wins despite covering far less.

### Answers to the pre-registered questions

1. **Does learned novelty close the gap to direct?** For coverage, partly -- learnedq
   reaches 79% (navigate) / 80% (stitch) of noveltyq with Spearman 0.80-0.91 on the
   target. For success, both are noise-dominated and neither beats random.
2. **Does success stop anti-scaling?** Coverage: yes, monotone everywhere. Success: no arm
   except random scales cleanly; the rest are flat-to-declining past budget 64.
3. **Does the head preserve ranking?** Yes -- Spearman 0.80-0.91, Pearson 0.78-0.89.
4. **Is q-novelty better than z-novelty?** For coverage, modestly (100.0 vs 87.4 navigate;
   84.4 vs 82.8 stitch -- near parity). For success, no. **Do not attribute the gain to
   controllability-aware q.**
5. **Coverage at equal budget:** table above; ordering is the near-inverse of success.

### Caveat retracted from the training logs

`expansion/*` diagnostics logged *during training* are uninformative: the gain-training
tree uses budget 16 with `expansion_batch_size=4`, so iteration 1 has only the root on the
frontier and iteration 2 expands all four children -- the scorer never chooses, and all
arms log identical values. Evaluation at budget >= 64 is unaffected. Raise
`train.gain_tree_budget` before relying on those columns.

### What this implies for the next step

The bottleneck is no longer the gain target or the representation -- it is that expansion
maximises novelty without regard to whether the prediction at that depth is still valid.
The uncertainty head already exists and is trained; a novelty/reliability-traded score
(or a depth-discounted planner objective) is the change the evidence actually points to.


---

## Experiment 3 — reliability diagnostics (no retraining)

Both diagnostics run on the Experiment-2 checkpoints. `train.gain_tree_budget` raised
16 -> 64 for future runs (at 16 with `expansion_batch_size=4` the scorer never ranks
competing nodes, which is why all q-space arms logged an identical Spearman of 0.796).

### D1. Depth-limited novelty — helps, never catches Random

Success on stitch, 3 seeds, with selected-leaf depth matched between arms:

| depth cap | noveltyq (leaf depth) | Random (leaf depth) | delta |
|---|---|---|---|
| 3 | 0.178 (2.53) | 0.222 (2.53) | -0.044 |
| 5 | 0.189 (3.77) | 0.300 (3.54) | -0.111 |
| 8 | 0.233 (3.78) | 0.300 (3.47) | -0.067 |
| none | 0.122 (3.82) | 0.289 (3.43) | -0.167 |

(budget 64; budget 256 is the same story, worst at -0.267.) Depth penalty
`S = novelty_q - lambda*d` peaks at 0.233 (lambda=0.05), also below Random.

Capping depth **does** help novelty (0.122 -> 0.233 at budget 64), so the
coverage-vs-reliability tradeoff is real. But **Random wins at every matched depth**,
including depth 3 where selected-leaf depths are identical to two decimals. The two arms
respond to depth in *opposite* directions -- Random improves with depth, novelty degrades
-- even though they share one world model. Random simply never goes deep on its own
(leaf depth 5.5 uncapped vs 9.9 for novelty). Depth is not the explanation.

### D2. Oracle-grounded leaf selection — the tree is the problem

Every node's root-to-node chunks replayed in MuJoCo (DFS with exact state save/restore),
then leaves selected three ways on the identical tree. 10 episodes/cell, seed 0:

| arm / budget | latent | predicted | **oracle** |
|---|---|---|---|
| noveltyq B64 | 0.000 | 0.200 | **0.200** |
| noveltyq B256 | 0.000 | 0.100 | **0.100** |
| randomtreewm B64 | 0.200 | 0.300 | **0.400** |
| randomtreewm B256 | 0.100 | 0.100 | **0.300** |

| arm / budget | disagree | min predicted d | min actual d | regret |
|---|---|---|---|---|
| noveltyq B64 | 0.415 | 15.17 | 15.91 | 0.33 |
| noveltyq B256 | 0.780 | 9.93 | 14.68 | 0.49 |
| randomtreewm B64 | 0.455 | 12.94 | 13.06 | 0.20 |
| randomtreewm B256 | 0.867 | 9.60 | 11.69 | 0.79 |

**Oracle selection does not rescue novelty.** For noveltyq, oracle == predicted (0.200 /
0.100): grounding adds nothing beyond decoding, so long-horizon endpoint error is *not*
its bottleneck. Random gains from grounding (0.300 -> 0.400) -- its trees contain good
futures that selection misses -- and **Random under oracle selection still beats noveltyq
under oracle selection, 0.400 vs 0.200.**

The best *actually reachable* node in a novelty tree ends 15.9 units from the goal
(budget 64) versus 13.1 for Random, in a maze ~20 units across. Novelty trees simply
contain worse futures. Regret is small everywhere (0.2-0.8), i.e. selection picks a
node nearly as good as the best one available -- the best one available just is not good.

Novelty trees are also more over-optimistic: predicted-vs-actual gap 4.8 units at budget
256 versus 2.1 for Random, consistent with their greater depth.

### D3. Latent goal matching is a separate, large bottleneck

Confirmed on 90 episodes/cell (3 seeds x 10 tasks x 3 episodes), budget 64:

| arm | latent (deployed) | decoded position | gain |
|---|---|---|---|
| noveltyq | 0.144 +/- 0.079 | **0.367 +/- 0.047** | +155% |
| randomtreewm | 0.256 +/- 0.016 | **0.489 +/- 0.042** | +91% |

Scoring leaves by `d_z(z_n, z_g)` is far worse than decoding both to position and
comparing there. `z` is trained for dynamics and future prediction, never for metric goal
matching, so distances in it do not track spatial proximity. The decoder is part of the
model, so this is a deployable change, not privileged information -- and it is free.

### Verdict against the pre-registered criteria

```
oracle q-novelty >> Random ?           NO  -> endpoint error is NOT the bottleneck
depth-limited novelty improves ?       PARTLY -> real but secondary; never catches Random
oracle still does not rescue novelty ? YES -> coverage maximisation is misaligned
                                              with goal control
```

**Do not build uncertainty/reliability-weighted novelty yet.** The evidence says the
expansion objective is pointed at the wrong thing: maximising q-novelty drives expansion
toward states that are maximally *distinct*, which in a maze means the far corners, while
goal-reaching needs dense coverage of the corridor the goal sits in. A reliability weight
would make novelty trees shallower and safer without making them more useful.

Two changes the data actually supports, in order:
1. **Score leaves by decoded position, not latent distance** (+0.22 absolute, free).
2. **Replace the expansion objective.** Novelty/coverage is not the right target; the
   evidence points toward goal-agnostic *reachability spread* that stays within the
   model's reliable horizon, or accepting that goal-conditioned expansion is required.


---

## Layout

```
configs/           base.yaml + env/ + model/ + experiment/ (7 arms x 2 datasets + ablations)
treewm/data/       OGBench loading, chunk sampling, future-set modes, latent retrieval
treewm/models/     encoder, controllability encoder, branch transformer, dynamics, arms
treewm/tree/       batched tensor trees, frontier scorers, budgeted expansion, matching
treewm/losses/     world / support / controllability losses + the total-loss assembly
treewm/planning/   goal enters only at leaf selection
treewm/evaluation/ rollout, coverage, diagnostics, task splits
treewm/logging/    TensorBoard wrapper (rank-0 only), DDP-safe metric reduction
scripts/           train, eval, sweep_budget, visualize_tree, run_grid
tests/             31 tests
```

## Loss warmup (and why)

Auxiliary terms ramp in linearly (`losses.warmup`). This is not cosmetic: before the
branch heads differentiate, every sibling pair has `d_q ≈ 0`, so `exp(-d_q/tau) ≈ 1` and
the redundancy penalty `Σ_{i<j} κ_i κ_j` is minimised by driving **every** KEEP score to
zero. Observed directly — effective branching factor pinned at 0.00 for a whole run — and
fixed by the ramp (0 → 1.84 over 600 steps).

## Evaluation protocol

- Primary metric: `eval/success_rate` vs node budget on the **hard split**.
- Hard split = start/goal pairs in the top geodesic-distance quartile. On
  `pointmaze-medium-stitch` (5000 trajectories of 200 steps) no single dataset trajectory
  spans a distant pair, so success requires composing futures across trajectories — where
  single-trajectory prediction fails structurally rather than by tuning.
- Node count is the compute-normalised axis; wall-clock is recorded but secondary.
- Episode seeds derive from the task index, so all arms see identical start states.
- Report ≥3 seeds.

## Known limitations

- `q_advantage_over_z` was **negative** in a 600-step smoke run. Far too early to mean
  anything, but it is the premise diagnostic and it is not yet positive.
- The expansion-gain head trains on a stride of 4 under a 2000-step warmup, so it needs a
  substantial run before `expansion/gain_rank_correlation` is meaningful.
- Only PointMaze is validated. AntMaze configs exist but are untested, per the spec's
  instruction not to proceed until PointMaze diagnostics are sensible.
- The retrieved future set transfers a neighbour's *displacement* onto the anchor (a local
  chart assumption). PointMaze observations are position-only `(x, y)`, so velocity
  mismatch between anchor and neighbour is not observable and is absorbed as noise.
