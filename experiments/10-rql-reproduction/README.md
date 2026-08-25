# Formal Reversal Q-Learning reproduction

This directory is a fail-closed campaign runner for the released RQL training
protocol: 10 OGBench settings, five numbered single-task goals, four seeds,
exactly 1,000,000 offline updates, and a mandatory final evaluation of 50
episodes.  The manifest expands to exactly 200 independent one-GPU runs.  It
does not train TreeWM or use distributed data parallelism.  A 13-element Slurm
array requests two nodes per element and can run all 200 agents concurrently:
12 elements own 16 runs and the final element owns eight.

No credential is stored here.  W&B authentication is read from
`WANDB_API_KEY` or the existing `api.wandb.ai` entry created by `wandb login`
in `~/.netrc`.  OGBench data are public, so `HF_TOKEN` is not consumed by this
pipeline.  Do not put either secret on a command line or in a Slurm file.

## Protocol inventory

| setting | reward | dataset | alpha | expectile | rho | h | discount |
|---|---|---|---:|---:|---:|---:|---:|
| scene | sparse | standard | 3.0 | 0.7 | 0.5 | 5 | 0.99 |
| puzzle-3x3 | sparse | standard | 1.0 | 0.7 | 0.5 | 5 | 0.99 |
| puzzle-4x4 | sparse | 100M shards | 1.0 | 0.9 | 0.5 | 5 | 0.99 |
| cube-double | semi-sparse | standard | 10.0 | 0.9 | 0.5 | 5 | 0.99 |
| cube-triple | semi-sparse | standard | 1.0 | 0.9 | 0.5 | 5 | 0.99 |
| cube-quadruple | semi-sparse | 100M shards | 1.0 | 0.7 | 0.5 | 5 | 0.99 |
| antmaze-large | semi-sparse | standard | 0.1 | 0.5 | 0.5 | 1 | 0.99 |
| antmaze-giant | semi-sparse | standard | 0.1 | 0.5 | 0.5 | 1 | 0.995 |
| humanoidmaze-medium | semi-sparse | standard | 0.3 | 0.5 | 0.0 | 1 | 0.995 |
| humanoidmaze-large | semi-sparse | standard | 0.3 | 0.5 | 0.0 | 1 | 0.995 |

Every row uses ensemble count 10 and batch size 256.  Task IDs are explicitly
`task1` through `task5`; the unnumbered OGBench default aliases are tuning
commands, not five distinct evaluations.  The two 100M settings rotate across
100 train/validation shard pairs every 1,000 absolute updates and explicitly
pass a 100M buffer-size compatibility flag.  The parsed semantics of
[`manifest.json`](./manifest.json) are locked by [`protocol.sha256`](./protocol.sha256)
and passed into every trainer checkpoint.

## 1. Environment and durable paths

Install the vendored requirements in a Python environment with the
cluster-appropriate CUDA JAX wheel.  The formal scripts never mutate the
environment:

```bash
python -m pip install -r experiments/10-rql-reproduction/upstream_rql/requirements.txt
wandb login
```

Choose durable, high-capacity storage.  The two sharded datasets plus standard
pairs require roughly 66 GB; leave additional headroom for `.part` files and
200 checkpoints.

```bash
export RQL_PYTHON=/absolute/path/to/python
export RQL_DATA_ROOT=/absolute/scratch/path/rql-data
export RQL_RUN_ROOT=/absolute/scratch/path/rql-runs
```

## 2. Stage and verify data

On a login or transfer node:

```bash
"$RQL_PYTHON" experiments/10-rql-reproduction/prepare_data.py \
  --download --data-root "$RQL_DATA_ROOT"
```

This expects 416 NPZ files: 16 standard train/validation files and 400 files
for the two 100M datasets.  Per-dataset `flock` locks prevent races between
jobs.  Each transfer uses an fsynced `.part` file, HTTP Range resume, bounded
retry/backoff for transient network failures, structural ZIP/NPZ validation,
and atomic rename.  Re-running the same command skips valid files and resumes
partial ones.

If login-node transfers are inappropriate, submit the resumable single-rank
staging job and make training depend on it in one command.  The available
partitions require a GPU request, so this I/O job reserves one GPU even though
the downloader itself does not use it:

```bash
"$RQL_PYTHON" experiments/10-rql-reproduction/submit.py \
  --submit --stage-data \
  --data-root "$RQL_DATA_ROOT" --run-root "$RQL_RUN_ROOT"
```

`stage_data.slurm` also has a four-hour limit.  On USR1, its downloader fsyncs
the current partial file and exits 75; the batch wrapper explicitly calls
`scontrol requeue`.  `submit.py` uses an `afterok` dependency for training.

For a read-only check:

```bash
"$RQL_PYTHON" experiments/10-rql-reproduction/prepare_data.py \
  --check --data-root "$RQL_DATA_ROOT"
```

## 3. Validate without submitting

```bash
"$RQL_PYTHON" experiments/10-rql-reproduction/submit.py \
  --dry-run --require-data \
  --data-root "$RQL_DATA_ROOT" --run-root "$RQL_RUN_ROOT"
```

The validator locks the 200-run expansion, exact 13-allocation ownership,
1M/50-episode/rematerialization flags, protocol hash, trainer interface,
literal Slurm directives, 16 one-GPU `srun` mapping, completion status, data,
and absence of embedded credential-like strings.  Dry-run never calls
`sbatch`.

## 4. Submit

Run the same entry point with `--submit` after data preflight:

```bash
"$RQL_PYTHON" experiments/10-rql-reproduction/submit.py \
  --submit \
  --data-root "$RQL_DATA_ROOT" --run-root "$RQL_RUN_ROOT"
```

The submitter pre-creates repository-level `logs/`, because Slurm opens
`logs/%x_%j.out` before the batch shell runs.  It submits array elements
`0-12`, followed by a strict aggregation job with an `afterok` dependency on
the entire array.  It accepts a W&B key from the environment or `~/.netrc`
without printing it.  Every allocation generation
performs a cheap 16-rank gate: it verifies two current hosts with local ranks
0--7 and a single working JAX GPU per rank.  The first generation for a given
protocol/code/environment fingerprint also creates and renders an OGBench
MuJoCo environment through EGL, compiles and blocks one full
batch-256/horizon-5 RQL update with Flax rematerialization enabled, and
round-trips an atomic agent checkpoint on every rank.  Rank 0 performs a
read-only authenticated W&B API call.  A durable success sentinel is keyed by
the protocol, trainer/preflight source, Python build, and the trainer's full
curated runtime package set, so only this costly gate is reused on requeues;
the current-allocation binding/JAX/topology check always reruns.

## Requeue and resume semantics

The formal allocation stops itself at a shared 3h40m deadline, before Slurm's
literal `USR1@420` fallback.  A signal or deadline follows this path:

```text
dispatcher rank -> trainer USR1 -> safe update boundary -> atomic checkpoint -> exit 75
        |                                                               |
        +-- state/allocations/shardN/requeue/$SLURM_RESTART_COUNT/ready.rank --+
                                      |
                         rank 0 waits up to 240 s
                                      |
        scontrol requeue ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
```

The cluster has no `RequeueExit` policy, so exit 75 and `#SBATCH --requeue`
alone are insufficient.  Rank 0 explicitly requeues only its composite array
element ID, never the array master, even if one readiness marker times out; in
that case the missing rank resumes from its latest periodic atomic checkpoint.
A genuine trainer error records `action=abort` and suppresses requeue.

Runs have deterministic global-slot ownership: manifest index `i` belongs to
array element `i // 16` and worker rank `i % 16`.  No run can be claimed by
two elements.  Paths, run names, and W&B IDs are stable.  A restarted run uses
`resume="allow"`, appends local/W&B history,
and restores model, optimizer/targets, JAX/NumPy/Python RNG state, absolute
step, shard index, and pending evaluation.  A rank skips only a
`COMPLETED.json` whose full immutable identity hash and mapping agree with
`run_metadata.json`, the current trainer source manifest and runtime software,
the protocol/upstream/rematerialization/run identity, the 1M final step, and
the 50-episode final evaluation.  Idle ranks remain in the job step so every
generation can still rendezvous all 16 ranks.

## Results

After all 13 array elements complete, the dependent report job invokes:

```bash
"$RQL_PYTHON" experiments/10-rql-reproduction/aggregate.py \
  --run-root "$RQL_RUN_ROOT" --output-dir "$RQL_RUN_ROOT/report"
```

Aggregation fails rather than emitting a partial result if any completion or
final evaluation is missing or mismatched.  It produces `report.json`,
`per_task.csv`, and `summary.md`: four-seed success mean and two-sided 95%
Student-t CI for each of 50 tasks, seed-paired five-task setting aggregates,
and the seed-paired 50-task aggregate.

See [`PROVENANCE.md`](./PROVENANCE.md) and
[`upstream_rql/UPSTREAM_PROVENANCE.md`](./upstream_rql/UPSTREAM_PROVENANCE.md)
for source and local-patch provenance.

## Offline tests

```bash
pytest -q experiments/10-rql-reproduction/tests/test_campaign.py
```

These tests require neither a network connection nor a GPU.
