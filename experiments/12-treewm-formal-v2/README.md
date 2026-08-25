# Formal TreeWM-v2 50-task campaign

This is the fresh, scale-coherent TreeWM campaign. It trains only `arm=treewm` with
objective `treewm_v2_rms_rank_v1`; it neither launches nor merges RQL or any TreeWM
baseline. The ten dataset settings each expose five built-in tasks. Four training
seeds therefore mean 40 dataset-level models, 200 task/seed evaluations, and 10,000
final episodes—not 200 task-specific training jobs.

## Objective correction and preregistered gate

The v2 configuration puts controllability and every other loss into explicit,
dimension-stable units. It uses task-metric RMS endpoints with a bounded control
target and rank loss, detached RMS-normalized world targets, action-swap binding
negatives, balanced uncertainty, one-to-one support coverage, detached matched-only
redundancy, calibrated KEEP admission, normalized matching costs, and a set-aware
listwise gain scorer. Depth embeddings are disabled because formal training does not
supervise the depth rows used by inference. The inference-unused mass head and branch
gain prior are disabled and frozen, and latent retrieval is disabled because novelty
gain does not consume its index.

Retrieval radius, horizon displacement thresholds, and cluster thresholds are
calibrated independently for every setting using only a fixed train split sample.
The stage also publishes an exact compact future-recipe cache. Validation data,
rewards, goals, and evaluation results are forbidden inputs to calibration. Each
calibration must pass the immutable distribution gates in `manifest.json` before its
data contract is published.

Before the million-step campaign, every setting runs seed zero for 5,000 updates in
an isolated pilot namespace while retaining the formal one-million-update learning-rate
horizon. The pilot then restores the exact 5k checkpoint and, on
three deterministic, representative recorded batches of real train anchors at active
expansion-gain steps, computes
per-effective-loss gradients for the encoder, branch transformer, dynamics,
controllability, and contextual gain head. On each of the three batches, no one term
may account for more than 0.80 of the summed term norms on any of the four shared
modules. Contextual-gain gradients must be finite and positive but are not subjected
to a cross-term dominance test. Live contextual-gain health uses the interval-averaged
separately clipped gain-group norm because stride-4 gain updates never coincide with
the stride-50 module-snapshot step; the branch-prior head is frozen, so that group is
contextual-only. This gate also requires finite nonzero
module gradients, exact raw/effective objective accounting, calibrated future-set
telemetry, a real EGL render, activation rematerialization, and an exact atomic
model/optimizer/loader/RNG checkpoint restore. The bound is fail-closed: orchestration
does not silently alter a loss weight when it fails.

Every audited batch, the audit artifact union, and recent raw/effective telemetry must
contain exactly `state`, `action`, `horizon`, `bind`, `coverage`, `redundancy`, `keep`,
`uncertainty`, `recursive`, `reconstruction`, `control`, and active `expand`. The gate
rejects missing or additional terms, including disabled `mass` and `multistep`.

The same post-5k gate rejects representational collapse: recent mean controllability
pair distance must exceed 0.05, near-collapse fraction must stay below 0.95, predicted
and target gain standard deviations must each exceed `1e-4`, and effective branching
factor must be above 0.5 and at most 4.0. KEEP-aware support recall must exceed 0.1,
and expansion must generate at least 32 of the 64 budgeted nodes; generated-node and
budget-shortfall telemetry must also reconcile exactly to that budget.

The compact recipe makes future-set lookup bounded, while each rank's twelve allocated
CPUs permit ten loader workers and leave two CPUs for the trainer/runtime. BLAS/OpenMP
threads are pinned to one, preventing oversubscription across the sixteen ranks. The
pilot benchmarks `train/data_wait_frac` over the full 5k run. It warns above 0.35 and
fails above 0.50, so the million-step jobs cannot start if this throughput-oriented
loader setting still leaves input work dominant.

## Immutable namespaces

- Python: `/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python`
- source data: `/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/datasets/ogbench-rql-50task`
- approved read-only raw cache: `/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/datasets/treewm-50task-full-cache-v1`
- v2 contracts/recipes: `/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/datasets/treewm-50task-formal-v2-contracts-v1`
- pilot runs: `outputs/treewm-50task-v2-pilot`
- formal runs: `outputs/treewm-50task-1m-v2`
- formal W&B project/group: `treewm-50task-formal-v2` / `treewm-50task-1m-v2`

The old raw cache is reused by content identity only and is never modified. All v2
calibrations, recipes, contracts, scheduler state, checkpoints, W&B IDs, and reports
are fresh and namespaced separately. Formal launch guards reject a v1 run,
checkpoint, or W&B namespace.

Submission atomically copies only the executable TreeWM Python, Hydra configuration,
and exp12 protocol files into
`outputs/treewm-50task-1m-v2/state/source-snapshots/PROTOCOL_SHA/repo`. Source files
are read-only, outputs/data/caches/git metadata/credentials are excluded, and every
stage and every requeue verifies the complete snapshot hash before doing work. All
dependencies run from this one snapshot, so later edits to the shared worktree cannot
change or interrupt the multi-week campaign.

W&B authentication comes only from a mode-600 `~/.netrc` entry for `api.wandb.ai`.
The submitter strips variables containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, or
`CREDENTIAL` before every `sbatch`; no credential is stored in this experiment.

## Validate and launch

From the repository root, validate source inventory, config/objective invariants,
runtime, protocol lock, scripts, namespaces, and the real scheduler parser:

```bash
python experiments/12-treewm-formal-v2/submit.py \
  --dry-run --stage-data --scheduler-test
```

Submit the complete fail-closed chain with:

```bash
python experiments/12-treewm-formal-v2/submit.py \
  --submit --stage-data --scheduler-test
```

The submitted dependency graph is:

```text
10-setting train-only calibration array
        | afterok
all-setting calibration gate
        | afterok
10-setting compact future-recipe array
        | afterok
10-setting seed-zero 5k pilot array
        | afterok
all-setting gradient/health acceptance gate
        | afterok
3-element formal array: 16 + 16 + 8 independent 1-GPU runs
        | afterok (the complete array)
strict 40-model / 200-cell / 10,000-episode report
```

The pilot and formal jobs use stable, namespace-specific W&B IDs. A requeue resumes
the same run with `resume=auto`; it does not create another W&B run.

## Four-hour requeue and cancellation semantics

Every formal allocation uses the literal requested two-node layout: eight ranks and
eight GPUs per node, one GPU per independent model. At 3h40m the dispatcher requests
an allocation-wide checkpoint; Slurm's `USR1@420` provides a second stop path. A
current-generation barrier records durable checkpoints before an elected survivor
authorizes `scontrol requeue ArrayJobID_ArrayTaskID`. If one rank dies or hangs, a
survivor requeues the same composite element from the remaining periodic checkpoints.

SIGTERM is cancellation, not preemption. The batch shell and dispatchers write a
generation-independent cancellation latch. An intentional `scontrol requeue` writes
and fsyncs a generation-scoped intent marker first, so only the TERM caused by that
specific scheduler call is suppressed. A racing `scancel` still wins, and a successor
observing the persistent latch exits rather than entering a requeue loop. Run locks
prevent two allocations from mutating the same checkpoint directory.

Every restart reruns the two-host/eight-rank/one-GPU topology check. The heavier gate
is cached only under a hash of the live trainer/runtime, executable launch protocol,
all v2 contracts and calibrations, and the accepted pilot artifact.

## Monitoring and output

Use `squeue -j JOB_ID` and
`sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed` for scheduler state. Slurm
generation logs append under `logs/`; per-run trainer attempts and coordination
markers live under the corresponding pilot/formal root.

The reporter rereads all ordered raw episode records and independently recomputes
every success rate. It refuses missing, duplicate, reordered, stale-data,
stale-calibration, stale-objective, or non-TreeWM artifacts. Successful output is in
`outputs/treewm-50task-1m-v2/report/`.
