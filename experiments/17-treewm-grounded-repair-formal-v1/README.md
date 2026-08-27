# TreeWM grounded repair formal v1

This is the fresh formal campaign for the repaired trajectory-grounded TreeWM method.
It never resumes or reinterprets any v1, formal-v2, diagnostic, pilot, or bridge
checkpoint. Launch is blocked until `bind_prerequisites.py` validates and seals the
exact accepted exp15 report plus exp16's deterministic global F/H recipe selection.
Formal training consumes that selection; it performs no outcome-based recipe choice.
The binder independently pins exp15's package/source/runtime identities and recomputes
its integrity, per-run scientific, quorum, outcome, and noninferiority claims from the
complete 40-run report, pins exp16's final manifest/protocol/source/runtime identities, recomputes
its 18/20 + 8/10 + every-setting scientific quorum and outcome gates from all 40 raw
records, and requires exp16's embedded exp15 binding to match the same exact files.

## Fixed scientific design

- 10 settings × 4 fresh seeds 200–203 = 40 independent one-GPU jobs from scratch.
- Every job retains a 1M scientific/scheduler horizon through exact-resume lifecycle
  boundaries 2k → 25k → 100k → 1M. A stage starts only after the preceding immutable
  all-40 gate accepts.
- The method uses world LR 3e-5, balanced KEEP, sequence-level sampling,
  `grounded_execution_v2`, detached self-fed parents, published-union recipes,
  learned+guard-on inference, domain-raw scoring, clipped e4 execution, and exactly
  the full or half grounded-loss weights selected by exp16.
- The fixed representative validation sample uses seed 1701. Validation, diagnostics,
  checkpoints, and monitoring never select a formal recipe.
- The registered tree-search budget is 64 nodes. The first-edge guard adds four
  explicit root-successor predictions, reported as 68 effective world predictions
  per full-budget replan.
- Every one-GPU task requests 12 CPUs and 64 GiB. Requeue/cancellation state is scoped
  to the exact array element. A cancellation signal latches intent without killing the
  local `srun` client; the remote worker must checkpoint/verify and publish its durable
  cancellation receipt before the batch may exit.

Every stage requires target-appropriate telemetry: train scalars must reach the exact
target, while validation/diagnostic scalars must reach the exact registered 2k
boundary (24k for the 25k checkpoint). The recent 5k train-scalar axis must be
complete, both module gradient norms must stay finite and nonzero, both clip
coefficients must stay finite in (0,1], and at most 25% may fall below 0.05. The 2k
gate additionally requires exact provenance/checkpoint integrity and fixed finite
validation for all 40 jobs. At exactly 25k, every setting must
pass the preregistered horizon, q, gain rank/pair/coverage, support, validation, and
self-fed nonregression gates in at least 3/4 seeds. The 100k gate requires the common
five-episode prospective monitor on all jobs, at least one success, positive
fleet-mean distance reduction, and at least one positive-progress run. After 100k,
the 1M continuation depends only on all-40 integrity and numerical health.

## Separate final evaluation

The accepted exact-1M gate unlocks a separate 200-element array:
`training_index*5+(task_id-1)`. Each cell evaluates learned and BFS rails on the same
protocol-bound 50 raw episode seeds. All 200 immutable results (10,000 episodes per
rail) are required. Raw episode records retain task ID, episode index, exact seed, and
success, with resumable planner RNG state.

Aggregation treats the four training seeds as the primary inference units. It reports
paired learned-minus-BFS seed macros, sample SD, df=3 95% Student-t confidence
intervals, and paired episode discordance. Pooled episode rates are descriptive only.

## Blocked/test-only verification

While exp15 or exp16 is pending, this validates structure and reports the campaign as
blocked without snapshotting or contacting Slurm:

```bash
/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python experiments/17-treewm-grounded-repair-formal-v1/submit.py --test-only
```

After both reports accept, first dry-run and then explicitly publish their immutable
binding. Publishing refreshes `protocol.sha256`.

```bash
/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python experiments/17-treewm-grounded-repair-formal-v1/bind_prerequisites.py
/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python experiments/17-treewm-grounded-repair-formal-v1/bind_prerequisites.py --publish
```

Then run full external recipe-byte verification:

```bash
/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python experiments/17-treewm-grounded-repair-formal-v1/submit.py --verify-files
```

`--submit` is the only launch path. It creates a read-only source snapshot and one
fail-closed `afterok` DAG:

```text
train2k[40] -> gate2k -> train25k[40] -> gate25k
 -> train100k[40] -> gate100k -> train1m[40] -> gate1m
 -> final_eval[200] -> aggregate
```

Never bypass a failed gate or reuse an old namespace. Protocol, prerequisite,
snapshot, source/runtime, data/recipe, checkpoint, and seed-table hashes must agree.
