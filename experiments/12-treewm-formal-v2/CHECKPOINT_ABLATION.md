# Paused-v2 checkpoint inference ablation

This diagnostic is read-only: it loads the paused v2 `latest.pt` files, changes only
inference settings in memory, and writes content-addressed JSON under
`outputs/treewm-v2-checkpoint-ablation`. It does not resume training, alter optimizer
state, or write in any checkpoint directory.

## Stage 1: representative screen (default)

The four preregistered settings cover the observed failure families:

- `antmaze-large` — locomotion with measurable goal-directed movement;
- `cube-double` — manipulation with near-immobile objects;
- `puzzle-3x3` — categorical progress followed by undoing; and
- `scene` — mixed manipulation state.

The compact grid has nine arms and is paired on checkpoint, task ID, episode index,
environment seed, and 64-node budget. The Slurm wrapper therefore launches 36
independent work items by default. Each item requests one GPU and 64 GiB of memory so
multiple diagnostics can share a multi-GPU node:

```bash
sbatch experiments/12-treewm-formal-v2/checkpoint_ablation.slurm
```

Do not promote based on a single task or the best-looking arm. The screen passes only
when all 36 results are complete and supported, and the fixed corrected reference
`domain_raw-d3-e4-learned-hlearned` versus the historical
`normalized_l2-d16-e16-learned-hlearned` satisfies both preregistered conditions:

1. `eval/distance_reduction_frac` improves in at least three of the four settings and
   its median paired setting-level delta is positive.
2. `eval/success_rate` does not decrease in any setting.

The fixed-16 horizon and alternative frontier scorers are diagnostic contrasts; they
do not replace the corrected reference after looking at screen outcomes.

## Stage 2: all ten settings

Only after the screen criterion passes, expand the unchanged nine-arm grid to all ten
seed-0 checkpoints (90 work items):

```bash
sbatch --array=0-89%40 experiments/12-treewm-formal-v2/checkpoint_ablation.slurm --stage full
```

Use `python scripts/checkpoint_ablation.py --dry-run` (or `--stage full --dry-run`) to
print the exact immutable work map before submission. `--resume` only verifies and
skips an identical result; there is no overwrite mode.

## Grounded-repair diagnostic grid

`--grid grounded-repair` is a separate, post-failure diagnostic for the stopped
grounded-v1 25k checkpoints. It locks eight inference-only arms spanning the current
recipe, KEEP thresholds 0.50/0.42, first-edge guard on/off, learned/BFS/exact-q frontier
allocation, and one fixed-16 horizon control. The 0.42 threshold was fixed from the
midpoint of the two puzzle settings' observed target support priors before evaluating
any rollout outcomes. It is a diagnostic of score calibration and tree admission; the
replacement training recipe retains the formal 0.50 decision threshold and instead
trains KEEP with mode-balanced BCE.

For five settings, four model seeds, and eight arms the exact work map contains 160
items:

```bash
python scripts/checkpoint_ablation.py \
  --checkpoint-glob 'outputs/treewm-grounded-formal-v1/*/treewm/*/checkpoints/latest.pt' \
  --settings antmaze-large,scene,puzzle-3x3,puzzle-4x4-100m,cube-quadruple-100m \
  --seeds 0,1,2,3 --grid grounded-repair --dry-run
```

As with the original study, run all work from one immutable source snapshot and use a
separate content-addressed output root. No arm is promotable when every success count is
zero; progress and guard telemetry are causal diagnostics only.
