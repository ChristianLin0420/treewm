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
