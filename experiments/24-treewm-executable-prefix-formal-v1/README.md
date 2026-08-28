# Exp24: fresh executable-prefix formal v1

Exp24 is the distinct 1M formal campaign that may follow a future accepted Exp23
engineering pilot. Launch7 terminated negative and can never provide positive
authority. This
directory freezes the matrix geometry, global recipe, decision operators,
fail-closed dependency, and the bounded M2A orders 0–3 runtime-authority scaffold.
Per-setting contracts and the executable scientific protocol remain explicitly unsealed;
it is intentionally **not executable**. Running `submit.py --submit` fails before
any output, snapshot, scheduler, or W&B mutation.

## Frozen geometry and global recipe

The method is selected GSEP only, with executable-prefix weights fixed before any
formal outcome is observed:

- action `0.033368419`
- latent `0.027645085`
- endpoint `0.011350645`

The training matrix is the Exp22-style ten settings crossed with the new Exp24-only
preregistered seeds 240, 241, 242, and 243: forty scratch models. A pre-assignment
worktree and all-history source census found no earlier training assignment for that
bank. The package authenticates the independently committed Launch7 terminal-negative
provenance record; it does not read the live Launch7 output tree. Exp24 never resumes or otherwise consumes
an Exp23 checkpoint, optimizer state, run telemetry, W&B run, evaluation progress,
mutable report, or other output state. Its run and held-out namespaces are new.

The guarded lifecycle is:

`40×2k → all-40 gate → 40×25k → all-40 gate → 40×100k → all-40 gate → 40×1M → all-40 gate → 200 held-out cells → aggregate → report`

Every edge is `afterok`. After 100k, only integrity and numerical health may stop a
run; outcomes cannot revise or select the formal recipe.

All four stage gates share the same exact 50-update gradient/clipping axis, bounded
5k recent gradient window, 1k gauge window, and 2k validation axis. Every required
gradient norm is strictly above `1e-8`; every clip coefficient is in `(0,1]`; and,
for each branch-transformer, world-rest, and gain clip tag separately, at most 25%
of recent values may be strictly below `0.05`. The early 2k gate uses the available
`min(stage, window)` axis. Prefix component gradients are bounded by the outcome-
blind prelaunch safety audit; runtime health uses the real world/gain/world-rest/
branch-transformer group tags rather than inventing a component-gradient tag.

Exp24 deliberately adopts the Exp22 formal visualization cadence: every 25k, with
an early 2k cadence through update 25k. Its future recipe uses the actual runtime
keys `future_sets.cache=false` and `future_sets.shared_cache=true`.

The held-out array is forty models times five tasks. Each cell loads the same exact
1M checkpoint and runs learned and BFS on the same tasks and ordered fifty episode
seeds with the same node budget, action bounds, maximum environment steps, and
resolved config. The only allowed rail difference is
`tree_config.scorer=learned|bfs`; rail tuning and order-dependent RNG are forbidden.
Pooled episodes are descriptive. The
primary inference first computes one learned-minus-BFS macro success-rate delta per
training seed as the unweighted mean of exactly fifty cell success-rate deltas, then
reports the two-sided 95% paired t
interval over four seed replicates (`df=3`, `t*=3.182446`).

## Independent negative and future-positive prerequisites

`launch7_negative.binding.json` authenticates the committed Launch7 terminal-negative
record at Git commit `b354b12b9446ee88aebbf58ccc67683194ad0fe1`, including its
raw/canonical hashes, exact 16-failed/4-completed plus canceled-reporter terminal
census, zero active jobs, lifecycle markers, deterministic shared failure, and the
no-report/no-resume/no-reuse conclusion. Launch7 checkpoints, W&B identities,
namespace, telemetry, and partial results remain permanently non-reusable.

Positive authority is a separate future leaf. It requires campaign
`treewm-executable-prefix-repair-pilot-v1-launch8`, status
`accepted_engineering_pilot`, a frozen version-specific reporter/gate adapter, and
an independently authenticated immutable quartet. The checked-in
`accepted_engineering_pilot.binding.json` is an exact unbound placeholder, and
`engineering_pilot_binder.py` is deliberately fail-closed until that protocol
freezes.

Independently of future acceptance, input-contract, future-recipe, prefix-target,
resolved-config, causal-parity, and fixed-weight-safety audits must cover all ten
settings. The safety audit verifies
the already-frozen tuple—without optimizer steps, outcomes, rollout, or retuning—at
the Exp23 3% per-component median and 10% every-row/group aggregate gradient caps.
In particular, cube-double, cube-triple,
antmaze-giant, humanoidmaze-medium, and humanoidmaze-large need new outcome-blind
formal audit rows. Pilot evidence selects the method and weights; it is not a
substitute for per-setting launch provenance.

## M2A orders 0–3 runtime-authority scaffold

Exp24 contains its own M1 control-plane foundation plus the bounded M2A authority
layer. It defines the exact 11-node,
10-edge DAG; a private atomic transaction claim; no-replace immutable records;
append-only journals; an exact source snapshot; isolated queued bootstrap; scheduler
control-plane capture; zero-job `sbatch --test-only` validation; parent-aware Slurm
array reconciliation; accepted-job normalization; all-node precommit re-observation;
exact-ID reverse-DAG rollback/cancellation; crash recovery; and a receipt barrier.
The 600-second pre-READY transaction budget and 30-second scheduler-client timeout
remain fail-closed operational bounds. Worker liveness does not rely on bounding
post-READY filesystem latency: the root remains held until receipt and authorization
are durable, and the outer transaction lock is released before the root is released.

M2A adds an exact acyclic artifact/schema graph in `m2a_schema.json`. Every
submission contract binds that schema, the authenticated Launch7 negative leaf,
the exact unbound future-positive leaf, and an interpreter record covering the
exact Python 3.11.15 venv symlink, resolved
executable bytes, base executable, `pyvenv.cfg`, and venv/base package roots. Queued
bootstrap re-captures and exact-matches that provenance before execution. This is
binary/pyvenv/path identity only: authenticated environment-content closure remains
an explicit execution-readiness blocker.

`train_2000` is submitted with `--hold`. The complete eleven-job receipt commits
while that root is still `PENDING` for `JobHeldUser`. Under the report/cancel lock,
the runtime re-observes that held state and durably publishes
`ROOT_RELEASE_AUTHORIZATION.json`; it then releases the outer claim-to-receipt lock
before taking the report/cancel lock again for exact-parent-ID `scontrol release`.
`ROOT_ACTIVATION_RESULT.json` records the released-state observation. Recovery
handles receipt-before-authorization, authorization-before-release, release-before-
result, publication residues, and retries without releasing a second time. Workers
authenticate the durable pre-release authorization; the result is audit/recovery
evidence rather than a new liveness prerequisite, and no released worker waits on
the outer transaction lock. A separate activation lock serializes result publishers;
the emergency-cancel lock covers the exact release side effect and observation but
is released before activation-result fsync, so delayed audit publication cannot
strand cancellation.

`engineering_pilot_binder.py` is only a versioning interface. It rejects every
positive verification request before opening the candidate path. Its canonical
requirements preserve the eventual raw-bundle telemetry-to-decision derivation,
per-cell bundle/decision/provenance joins, exact gate schemas, terminal artifact
joins, outcome-blind provenance hashes, and recomputed acceptance predicates. The
semantic adapter cannot be implemented or audited until Launch8 reporter/gate
source freezes. Its current implementation closure binds both the facade and
`runtime.py`; no synthetic fixture is treated as positive evidence.

This is still not an execution-ready runtime. The formal trainer objective is only
documented in `formal_objective_delta.json`, scientific adapters and durable
training/held-out resume handling are absent, the strict scalar/report evidence
schemas are not sealed, and report publication is deliberately disabled. Future
Launch8 acceptance, all-ten contracts/audits, clean committed snapshot provenance,
interpreter environment contents, and training/evaluation/requeue feasibility remain
hard blockers. Same-stage requeue scheduler mutation is unconditionally disabled
before any `REQUEUE_CALLING` record or scheduler call until exact run/W&B/checkpoint identity and an
idempotent, cancel-serialized lost-response transaction are sealed. `--submit` checks
those blockers before creating a claim, snapshot, output, or scheduler process.
The bootstrap requires the exact configured executable, Python 3.11.15, real
`-I -S -B` flags, and the contract-bound interpreter provenance described above.

Emergency cancel/recovery uses a stable v1 contract envelope and a minimal sealed
snapshot capsule (`cancel.py`, `runtime.py`, `campaign.py`, and `manifest.json`). It
does not depend on an unrelated training source file surviving an incident. A
committed cancellation still authenticates the exact receipt, canonical DAG, full
committed journal ledger, and exact parent job IDs before `scancel --quiet`.

Exp22's submit transaction and scalar parser are not reusable as-is. Exp24 must not
strand accepted ancestors after a partial `sbatch` failure, and it must not choose a
latest-wall-time value for duplicate `(tag, update)` records. Conflicting duplicates
are fatal; identical duplicates are inventoried or suppressed by the writer; dense
gain/support telemetry is checked on its 50-update axis.

A new `treewm_v2_grounded_executable_prefix_formal_v1` config/objective must also be
registered in the trainer's V2, latent-gauge, executable-prefix, formal, staged, and
authorization sets. Reusing the pilot objective is forbidden because that identity
is hard-capped at 25k.

## Current checks and emergency commands

From the repository root:

```bash
FORMAL_PYTHON=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
PYTHONDONTWRITEBYTECODE=1 "$FORMAL_PYTHON" -B -m pytest -q -p no:cacheprovider experiments/24-treewm-executable-prefix-formal-v1/tests
PYTHONDONTWRITEBYTECODE=1 "$FORMAL_PYTHON" -I -S -B experiments/24-treewm-executable-prefix-formal-v1/submit.py --test-only
PYTHONDONTWRITEBYTECODE=1 "$FORMAL_PYTHON" -I -S -B experiments/24-treewm-executable-prefix-formal-v1/submit.py --snapshot-test
PYTHONDONTWRITEBYTECODE=1 "$FORMAL_PYTHON" -I -S -B experiments/24-treewm-executable-prefix-formal-v1/engineering_pilot_binder.py --describe
```

The `--test-only` command prints the exact 40-run/200-cell DAG and current blockers. It is
required to report `persistent_writes_performed: false`, an empty scheduler-call
list, and `snapshot_created: false`.

Mutation and recovery reject every interpreter other than that exact Python with
`-I -S -B`. For a real transaction root, the live command authenticates the stable
dispatch envelope and `execve`s the sealed snapshot implementation before importing
the mutable worktree package:

```bash
PYTHONDONTWRITEBYTECODE=1 "$FORMAL_PYTHON" -I -S -B experiments/24-treewm-executable-prefix-formal-v1/cancel.py --cancel --submission-root /absolute/submission-root
PYTHONDONTWRITEBYTECODE=1 "$FORMAL_PYTHON" -I -S -B experiments/24-treewm-executable-prefix-formal-v1/cancel.py --recover --submission-root /absolute/submission-root
```

The equivalent direct snapshot-resident form, useful when the later live worktree
is unavailable, is:

```bash
PYTHONDONTWRITEBYTECODE=1 "$FORMAL_PYTHON" -I -S -B /absolute/submission-root/source-snapshot/repo/experiments/24-treewm-executable-prefix-formal-v1/cancel.py --snapshot-resident --cancel --submission-root /absolute/submission-root
```

No command above authorizes Exp24 execution in M2A. `--snapshot-test` writes only a
private temporary tree and removes it; it does not create the Exp24 output namespace
or contact Slurm. A live scheduler preclaim, when separately invoked for audit,
uses only controller reads and `sbatch --test-only` and must prove zero matching jobs.

## Remaining milestones

1. In parallel and before any Launch8 outcome is needed: bind interpreter environment
   contents; apply the shared objective/config authorization; seal all-ten
   outcome-blind audits/contracts; finish the held-out seed census and feasibility;
   and implement the scientific adapters.
2. After Launch8 reporter/gate source freezes, implement and independently audit the
   exact versioned semantic adapter. Only after an immutable quartet is accepted may
   it publish the real future-positive binding.
3. Seal all-ten input/future/prefix/resolved/causal/fixed-weight-safety audits,
   complete per-setting contracts, and build a held-out seed table disjoint from
   formal monitors, Launch7, and authenticated prior consumed evaluation seeds.
4. Finish the scientific adapters, strict telemetry/report layer, and distinct 1M
   objective on top of the present M2A runtime-authority scaffold.
5. Prove all-ten training throughput/requeue feasibility and every held-out
   setting/task/rail episode-runtime, durable-resume, paired-order, array, aggregate,
   and report walltime bound.
6. Join both prerequisite leaves and every independent pre-acceptance branch into
   the final protocol/readiness seal; run focused/full tests plus zero-write snapshot
   and scheduler preflights, and obtain an independent audit.
7. Only from an exact committed, pushed, clean tree: explicitly submit the fresh DAG.
8. Monitor all forty models through every gate, then run all 200 held-out cells and
   publish the seed-level paired report.
