# TreeWM grounded formal v1

This is the fresh formal campaign for the corrected, regularized, trajectory-grounded
TreeWM candidate. It does not resume or reinterpret any v1, formal-v2, diagnostic, or
pilot checkpoint. The operator's explicit pilot-gate bypass is recorded in
`manifest.json`; it does not weaken any subsequent lifecycle or outcome gate.

## Fixed scientific design

- 10 settings × 4 independent training seeds = 40 independent one-GPU jobs.
- Each job keeps `train.steps=1_000_000` and
  `train.scheduler_total_steps=1_000_000` at every lifecycle stage.
- The lifecycle-only `TREEWM_STOP_AFTER_UPDATE` boundary stops at exactly 2k, 25k,
  100k, and 1M updates after an exact-resume checkpoint. A later stage cannot start
  until the previous all-run gate is accepted.
- The method is the corrected grounded recipe: regularized world model, dedicated
  gain optimizer, mixed learned/novelty-q gain behavior, grounded recursive depth 3,
  raw-domain/e4 planning, and the first-edge non-regression guard.
- That guard recomputes the four root-prefix successors at each replan. The registered
  tree-search budget remains 64 nodes, while telemetry reports 68 effective world-model
  predictions per full-budget replan (64 search nodes + 4 guard checks).
- Every setting consumes the complete four-seed union already present in its published,
  read-only v2 future recipe. Package preflight chooses its audit anchors from those
  recipe records themselves. It also proves distinct train/validation manifest
  identities and zero source-file path/hash overlap.
- Every one-GPU task requests 12 CPUs and 64 GiB. Scheduler requeue and cancellation
  state are scoped to the exact array element `${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}`.

The 2k gate requires provenance, exact checkpoints, finite telemetry (including both
teacher-forced and self-fed grounded multistep validation losses), a fixed validation
sample, and live world/gain gradients for all 40 runs. Starting at 25k, each setting
must pass the preregistered horizon, q, gain rank/pair/eligibility-coverage, support,
total-validation, and independently checked self-fed multistep nonregression criteria
in at least 3 of its 4 seeds. The 100k gate additionally requires
the paired five-episode monitor bank on all 40 runs and at least one success across the
200 monitor episodes. Thus an all-zero replicated campaign cannot spend the remaining
900k updates per run.

## Separate final evaluation

The 1M lifecycle stage stops before the trainer's built-in terminal evaluation. After
the accepted 1M gate, a separate 200-element array maps exactly
`training_index*5+(task_id-1)`. Each element evaluates both learned and BFS allocation
on the same locked 50 episode seeds, so there are 10,000 episodes per rail. Progress is
resumable at episode boundaries and persists the planner RNG state.

Aggregation accepts exactly 200 immutable results. Pooled episode rates are descriptive;
the primary inferential units are the four training seeds. It reports per-seed macros,
sample SDs, df=3 95% t intervals (`t=.975=3.182446`), the paired learned-minus-BFS
interval, and paired episode discordance. No setting, task, seed, checkpoint, scorer, or
episode can be selected adaptively.

## Launch contract

First verify without creating a snapshot or contacting Slurm:

```bash
/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python experiments/14-treewm-grounded-formal-v1/submit.py
```

`--submit` is the only mutation path. It performs full recipe-byte verification,
atomically creates a read-only source snapshot, obtains a kernel-exclusive single-submit
claim, and submits one durable DAG:

```text
train2k[40] -> gate2k -> train25k[40] -> gate25k
 -> train100k[40] -> gate100k -> train1m[40] -> gate1m
 -> final_eval[200] -> aggregate
```

Every edge is `afterok`, and submission first proves the cluster's
`kill_invalid_depend` policy, so a failed gate cancels all descendants rather than
allowing them to consume compute. Every submitted
job ID and exact command is captured in the single durable receipt. The package never
submits unless explicitly invoked with:

```bash
/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python experiments/14-treewm-grounded-formal-v1/submit.py --submit
```

Do not invoke that command from a stale copy, and do not manually bypass a failed stage
gate. `protocol.sha256`, the snapshot marker, launch hashes, data contracts, recipe
manifests, checkpoint identities, and locked seed-table hashes must all agree.
