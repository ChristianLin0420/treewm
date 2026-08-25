# TreeWM v2 corrected factorial pilot

This is a clean, bounded diagnostic pilot. It cannot be described as formal validation,
cannot resume the stopped 1M checkpoints, and cannot directly promote another 1M run.
Its only allowed outcome is selecting (or rejecting) a corrected recipe for a later,
separately authorized experiment.

## Design

The array contains exactly 32 independent one-GPU runs:

`4 settings × 2 regularization levels × 2 grounded-multistep levels × 2 seeds`.

The deterministic array mapping is
`((setting_index * 4) + arm_index) * 2 + seed_index`. Settings are antmaze-large,
cube-double, puzzle-3x3, and scene; arms are `r0-g0`, `r0-g1`, `r1-g0`, and `r1-g1`;
seeds are 0 and 1. `r1` means lr `1e-4`, weight decay `1e-3`, dropout `0.1` (versus
the v2 values `3e-4`, `1e-4`, `0.0`). `g1` means grounded multistep weight `1.0` and
scheduled sampling `0.25` after a 5k warmup (versus disabled).

Every arm uses corrected fixed validation and task-metric q diagnostics, raw-domain
planner scoring, maximum model/tree depth 3, and clipped execution of 4 actions. Runs
train for 12k optimizer updates, validate/checkpoint every 2k, run one rollout per each
of the five built-in tasks at 6k and 12k, then run a separate final one-episode/task
evaluation.

The compatible input path is explicit: `future_sets.shared_cache=true` maps the
immutable normalised arrays and attaches the approved compact recipes, while
`future_sets.cache=false` prevents a second mutable future-target cache.

## Provenance boundary

The calibrated v2 future recipes are immutable compatible inputs, not products of the
current code. The package verifies their original campaign protocol, producer code,
producer runtime, data-contract, calibration, and recipe hashes. At the same time it
injects the current trainer `TREEWM_CODE_SHA256` and `TREEWM_RUNTIME_SHA256`. The two
producer hashes use the separate `TREEWM_RECIPE_CODE_SHA256` and
`TREEWM_RECIPE_RUNTIME_SHA256` fields and are recorded in checkpoint/completion identity.

Each run additionally binds:

- the current complete trainer/config source hash;
- this package's locked protocol hash;
- its deterministic Hydra-override config hash;
- the compatible input-contract, calibration, and recipe hashes.

A changed source between submission and execution, or between requeues, fails closed.

## Lifecycle

Validate without scheduling anything:

```bash
PY=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
$PY experiments/13-treewm-corrected-pilot/submit.py
```

Only an explicit `--submit` calls `sbatch`:

```bash
$PY experiments/13-treewm-corrected-pilot/submit.py --submit
```

The Slurm array is `0-31%32`, four hours maximum, one GPU/task, and exactly `64G` of
memory/task. The explicit memory request prevents Slurm's site default from assigning
an entire node's memory to every one-GPU element, so independent elements can share a
node. USR1 is forwarded to the trainer; exit 75 is requeued only after `latest.pt` is
loaded and its optimizer, scheduler, RNG streams, config, and provenance are verified.
The script uses absolute `/cm/shared/apps/slurm/current/bin/srun` and `scontrol` paths
and requeues only the exact `${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}` element.

This launch uses the isolated run root
`outputs/treewm-v2-corrected-factorial-pilot-v1-launch2`; the first launch root and its
cancelled-before-start submission receipt remain immutable audit records.

Cancellation has permanent precedence over requeue. To stop a live element safely,
create its persistent latch before signalling it:

```text
<run-root>/state/arrays/array-job-<array_job_id>_<task_id>/task-<task_id-2digits>/CANCEL_REQUESTED
```

The worker observes the latch, forwards TERM, requires the trainer's atomic checkpoint,
and never requeues that element. Do not delete a latch to reuse an old Slurm identity;
submit a new array instead.

## Acceptance

`report.py` is strict: all 32 identity-complete runs must exist. For candidate `r1-g1`,
both seeds in at least three settings must pass fixed-validation stability; horizon CE
below both `ln(5)` and the empirical constant-prior baseline; positive q advantage over
z and a dimension-matched random projection; gain rank/pairwise thresholds; support
recall/precision; and finite, non-pathological gradient/clipping telemetry. Paired final
rollout progress must beat `r1-g0` on average, progress must not regress from 6k to 12k,
and success must be non-inferior.

Even a passing report says `accepted_for_next_bounded_pilot`, with
`formal_validation=false`.
