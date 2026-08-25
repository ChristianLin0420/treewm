# Formal TreeWM 50-task campaign

This campaign trains the named TreeWM method (`arm=treewm`, model class `TreeWM`,
learned frontier scorer) for exactly 1,000,000 optimizer updates with activation
rematerialization. It is a new formal TreeWM experiment; it does not execute or merge
the earlier task-specific baseline campaign.

## Scientific unit

TreeWM learns a goal-agnostic model of each dataset, so the correct matrix is ten
dataset settings by four training seeds: 40 models. Every model is evaluated on the
environment's five built-in tasks for 50 episodes/task. The report therefore contains
10 x 5 x 4 = 200 task/seed cells derived from 10,000 raw final-evaluation episodes.
This differs from a protocol that trains a separate policy for every task, and the
report labels that distinction explicitly.

The immutable manifest pins the learned scorer, a 64-node inference tree, four-way
branching, standard task IDs 1..5, the full official releases for both 100M settings,
and the exact per-domain episode limits. Training selects 300,000 anchors uniformly
without replacement from each full valid transition universe. The shared cache,
global normalization, and source universe are built from the complete underlying
release, while the future-retrieval pool is a fixed uniform 50,000 items. This is a
disclosed task-aligned TreeWM anchor-cap protocol, not matched-data or full-anchor
training semantics. `protocol.sha256` covers the canonical
manifest and every executable campaign/launch source; a change requires a new lock and
cannot silently reuse old completions.

Endpoint policy follows the selected planning state: scene and both puzzles use
absolute endpoints because their state includes categorical one-hots, while cube and
locomotion settings use relative endpoints.

## Fixed infrastructure

The formal paths are deliberately fail-closed:

- Python: `/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python`
- OGBench source: `/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/datasets/ogbench-rql-50task`
- full shared cache: `/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/datasets/treewm-50task-full-cache-v1`
- runs: `outputs/treewm-50task-1m-v1`
- W&B project: `treewm-50task-formal`

Formal jobs authenticate to W&B through a mode-600 `~/.netrc`. The submitter strips
ambient variables whose names contain `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, or
`CREDENTIAL` before every `sbatch`, so unrelated shell credentials are not exported to
compute jobs. No command accepts a credential argument, and no token is written to this
directory. Hugging Face authentication is not used because the formal source data is
already local.

## Validate and launch

Run from the repository root. Dry-run performs the source inventory, protocol, runtime,
topology/command, Slurm, and secret checks without submitting:

```bash
python experiments/11-treewm-formal/submit.py \
  --dry-run --stage-data --scheduler-test
```

Formal submission creates one dependency chain:

```text
10-element cache array
        | afterok
3-element training array (16 independent one-GPU TreeWM models/element)
        | afterok after all array elements
strict 40-model / 200-cell report
```

Submit it with:

```bash
python experiments/11-treewm-formal/submit.py \
  --submit --stage-data --scheduler-test
```

To reuse an already submitted cache-stage array, retain an explicit dependency:

```bash
python experiments/11-treewm-formal/submit.py \
  --submit --data-job-id DATA_ARRAY_JOB_ID --scheduler-test
```

The launcher prints the three master job IDs. Training does not begin unless the entire
cache array completes successfully, and aggregation does not begin unless all three
training elements complete successfully.

## Four-hour requeue lifecycle

Each training element keeps the literal two-node/eight-ranks-per-node/eight-GPUs-per-
node allocation. Its 16 ranks are 16 independent single-GPU models (`world_size=1`),
not one DDP model. Array ownership is fixed by
`global_index = 16 * array_task + rank`, giving shard sizes 16, 16, and 8; the last
eight idle ranks remain alive for the allocation barrier.

Every generation reruns a current-node probe proving two hosts x eight ranks and one
working CUDA device per rank. A protocol/source/runtime-keyed full gate additionally
exercises the named TreeWM model, gradient rematerialization and backward, EGL rendering,
atomic checkpoint restore, and read-only W&B authentication.

At 3h40m the shared allocation deadline asks all active trainers to checkpoint. The
literal scheduler `USR1@420` is a second stop path. Each trainer writes and fsyncs
`checkpoints/latest.pt`, marks the stable W&B run as preempting, and exits 75. Normally
all 16 dispatchers publish durable ready markers before an elected survivor authorizes
requeue; the batch shell then explicitly requeues only `ArrayJobID_ArrayTaskID`. If a
rank is dead or hung, the survivor barrier times out, records the missing ranks, and
explicitly requeues from every run's latest periodic checkpoint instead of letting the
four-hour limit terminate the campaign. At most 2,000 uncommitted updates are
deterministically recomputed; no durable update is skipped. A generation-scoped flock
promotes another ready rank if the first coordinator dies. The successor uses the same
immutable run directory and W&B ID and restores model/optimizer, loader cursor, and RNG
state from the selected exact checkpoint.

SIGTERM is cancellation, not preemption. Both the dispatcher and batch shell publish a
generation-independent cancellation latch. It overrides an earlier USR1/deadline
request, is checked immediately before `scontrol requeue`, and causes a racing successor
to exit without requeue. Immediately before an intentional scheduler call, a durable
restart-generation marker suppresses only the TERM emitted by that call; a failed call
removes the marker. Thus `scancel TRAIN_ARRAY_ID` cannot create a requeue loop, and an
intentional requeue cannot poison its successor with a false cancellation latch. A fresh
intentional launch receives a new array ID and isolated coordination namespace.

The cache array follows the same explicit composite-element requeue and cancellation
rule. Its two 100M elements consume every one of the 100 train and 100 validation shards;
there is no sampling or shard rotation mode.

## Durable state and W&B continuity

Each run is isolated at:

```text
outputs/treewm-50task-1m-v1/<setting>/treewm/treewm-<setting>-seed<seed>/
```

A nonblocking lease prevents duplicate allocations from mutating the same directory.
The stable W&B ID is a deterministic hash of campaign, setting, and seed, so every
requeue resumes one run rather than creating duplicates. Tracking outages do not
invalidate a locally durable scientific update; logs remain in the run directory and
W&B can synchronize after connectivity returns.

Do not delete an incomplete run to resume it. `resume=auto` loads `checkpoints/latest.pt`.
A `COMPLETED.json` is skippable only when its full run identity, current trainer/runtime,
protocol, data cache identity, exact 1M step, and resumable 250-episode final artifact
all validate.

## Report and monitoring

Use `squeue -j JOB_ID` and `sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed` for
scheduler state. Slurm appends generation logs under `logs/treewm50-1m_*.out`; per-run
attempt logs and coordination records are under the durable run root.

The afterok reporter writes:

- `report/report.json`: strict provenance and seed-level statistics;
- `report/task_seed.csv`: all 200 setting/task/seed cells;
- `report/per_task.csv`: 50 four-seed summaries with Student-t intervals;
- `report/summary.md`: compact per-setting and 50-task macro summary;
- `report/REPORT_COMPLETED.json`: hashes of every report artifact.

Aggregation rereads all ordered raw episodes and independently recomputes each task
success rate. Missing runs, reordered/duplicate episodes, stale source/cache/code,
metric/raw disagreement, or any non-TreeWM identity makes the report job fail rather
than publish a partial result.
