# Exp23 executable-prefix repair pilot v1 launch6

This is the sealed, launch-capable but unsubmitted launch6 package for a bounded 20-cell
engineering pilot. It compares the corrected Exp20 gauge + separate-branch-clipping
recipe (`GS`) with the same recipe plus executable-prefix grounding (`GSEP`).
The active campaign identity is `treewm-executable-prefix-repair-pilot-v1-launch6`,
with a fresh `outputs/treewm-executable-prefix-repair-pilot-v1-launch6` namespace,
`outputs/.exp23-34d79ab13d65ef27.transaction.lock`, and `exp23-launch6-*` run names.
No launch6 submission, persistent submission snapshot, scheduler job, W&B run,
checkpoint, optimizer update, or scientific output was created while building or
verifying this package; the required private snapshot-test copy was removed on success.

Launch1 through launch5 are permanently ordered negative provenance. Each
preserved read-only source snapshot was independently byte-matched 137/137 to its
recorded commit. Launch1 aborted before the submission contract because its isolated
audit inputs were unavailable. Launch2 also aborted before the contract: its causal
replay incorrectly included the newly created controlled `state/submission` tree in an
output fingerprint. Neither attempt produced a submission SHA, contract, job ID,
receipt, scientific run, checkpoint, W&B run, or optimizer update.

Launch3 sealed its contract but failed safely in the first sanitized train-name
`squeue` reconciliation, before either `sbatch` call. Its preserved tree has exactly
163 regular files and no symlinks: 137 snapshot files, 20 launch records, one contract,
and five journal records. The contract SHA is
`0cd594c8a49499b5e3d10a09ddbf3b89f981264be67bb603dc64836568a1b4c2`.
Both abort journals record empty train/report job-ID sets, no receipt exists, the
committed source order places both absence checks before the first submission call,
and independent exact-name `squeue` plus `sacct` observations found no matching jobs.
Launch3 produced no scheduler job, run, checkpoint, W&B run, scientific result, or
optimizer update.

Launch4 sealed contract
`0aa63e5787fbdb06331265f03dd5e1aa32c32c32bb9b74728cd3060be7200336`
and the scheduler accepted train ID `33211846` and report ID `33211848`, but the
post-submit canonical array-dependency validator rejected the report dependency. The
fail-closed abort cancelled both allocations before runtime: current `sacct` history
records both `CANCELLED`, elapsed zero, zero allocated nodes, and no array-task rows;
exact-ID `squeue` has no active rows. No receipt, READY marker, report-submitted journal,
log, task directory, checkpoint, W&B run, scientific result, or optimizer update exists.
The preserved launch4 root contains exactly 164 regular nonsymlink files: 137 snapshot
files, 20 launch records, one contract, and six journals. Its canonical path/content
inventory hash is
`d67768c00795e209a0b1998058cd475360b98cd3aab331b416d1b6934142adb5`.
The exception and scheduler submit/cancel commands are durable journal evidence. The
canonical `afterok:33211846_*(unfulfilled)` dependency and
`KillOInInvalidDependent=Yes` were instead an independent, unsealed live observation
immediately after abort and before slurmctld purged the records; the package does not
claim that value was preserved in journal stdout.

Launch5 sealed submission contract
`8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe`,
and the scheduler accepted train array `33217168` plus dependent reporter `33217171`.
Twelve tasks entered the real sealed `train_entry.py` bridge and failed with one
byte-identical log because Hydra could not import the relative `configs` package. This
occurred before config composition, model construction, or any optimizer update. The
exact cancel latch, call, and result preserve a successful exact-ID `scancel`. A later
independent terminal observation showed eight other tasks and the reporter cancelled.
That 21-row `sacct` ledger and empty `squeue` observation are explicitly unsealed
Launch6 attestation, not bytes preserved by Launch5. By contrast, the accepted
array-wide dependency and `KillOInInvalidDependent=Yes` are durable in Launch5 journal
`0004`.

The preserved Launch5 run root has 235 regular nonsymlink files and canonical
run-root-relative inventory SHA-256
`c07dce9aa58352f790af94bff8c719a3e9c8639bdd268d5b7d33824db8b7a874`;
its task subtree has 53 files and SHA-256
`31d41fd81ee8092c0ebdfc68e9cd5199e81698fb62a7a2a88e5f2c2a6f5666f5`.
Twelve empty run directories are bootstrap residue. There are no model, optimizer,
checkpoint, result, W&B, scientific READY, or scientific report-bundle files, and
Launch5 must never be retried or used as a resume/recovery source. The repair adds the
empty `configs/__init__.py` package marker to both trainer and snapshot identity and
exercises the actual sealed trainer bridge through resolved Hydra composition before
submission.

Exact protocols, manifests, inventories, claims, contracts,
fingerprints, scheduler evidence, and journals for all five attempts are sealed in order
in `manifest.superseded_launches`; no superseded namespace, identity, snapshot, or
state may be reused or resumed by launch6. Launch1–5 must never be retried, recovered
into submission, or used as a recovery source.

## Scientific design

The five settings are antmaze-large, scene, puzzle-3x3, puzzle-4x4-100m, and
cube-quadruple-100m. Each has both arms at fresh seeds 110 and 111. Array identity is
`((setting_index * 2) + arm_index) * 2 + seed_index`, with settings, arms, and seeds
in manifest order.

All 20 independent GPU cells start from scratch and continue under one trainer identity
to update 25,000. Update 5,000 is a retrospective analysis boundary only: it is not a
stop, promotion, stage, or selection point. Genuine USR1 requeues may exact-resume the
same owned checkpoint. Natural 25k completion then runs the terminal evaluation bank of
five tasks times five episodes. The gate consumes those exact 25 terminal rows, not the
descriptive five-episode periodic monitor.

Both arms enable the same executable-prefix graph. `GS` uses weights `(0, 0, 0)`;
`GSEP` uses the outcome-blind audited tuple:

- action: `0.033368419`
- latent successor: `0.027645085`
- normalized task endpoint: `0.011350645`

The weight audit used 40 fixed CPU-FP32 gradient observations: all ten immutable Exp20
GS exact-5k checkpoints and scratch seeds 230/231, with two counter-hash-stratified
published-union batches per setting/regime. Each component was capped at 3% of the
median base-objective gradient norm across both clipping groups and regimes, then a
common scale capped the summed component-norm upper bound at 10% on every row/group.
All intended gradients were finite and nonzero; the frozen maximum ratio is
`0.09999999714797946`.

Before either weight or prefix-target replay can deserialize a historical checkpoint,
the adjacent weight lock must match the exact file hash frozen by snapshot preflight.
That lock raw-byte-binds the Exp20 campaign manifest and all ten consumed
`GAUGE_PILOT_V2_LAUNCH.json` control files. Each checkpoint is opened through
component-safe `O_NOFOLLOW` descriptors, copied while hashing into a private temporary
file, and deserialized only from that authenticated copy. Dataset/cache/recipe payloads
are deliberately not added to this raw-byte control-file map: their non-pickle NumPy
memmaps and canonical/self-hashed manifests remain content-addressed by the existing
source, cache, and future-recipe identities, while exact consumed batch and prefix-target
hashes remain frozen in the scientific locks.

Action application reuses the planner's canonical projection. Loss and planner bounds
are explicit, equal, and tied to each environment's hash-checked Box. The executable
validator receives the effective depth-resolved tree config and fails closed on absent
or mismatched depth/bounds.

The fixed-validation prefix-target audit independently derives all 5,120 anchors per
setting. It hashes per-anchor branch counts, every logged selected horizon (only
`{4,8,16,32,64}`), each `p=min(4,h)` length, and each full 64-wide mask. At run time
the trainer publishes only its fixed-validation sampler summary and aggregate prefix
telemetry; the reporter binds those actual values to the frozen target lock. It does
not relabel a static recomputation as an observed loss artifact.

The causal-parity audit is also deliberately narrow and outcome-blind. For every
setting/seed pair it independently composes both arm launches, loads both arm data and
samplers, constructs a controlled CPU scratch model and fixed-validation batch, and
compares parameters, RNG, raw targets/artifacts, and telemetry. This is deterministic
reconstruction evidence—not instrumentation of the eventual GPU process before its
first forward, not an outcome, and not a claim over a provisional package protocol.
Actual workers separately bind the final immutable snapshot, resolved config, source,
runtime, data contracts, and all four audit artifacts.

`GSEP` must pass every unchanged Exp20 candidate gauge, separate-clip, validation,
self-fed, horizon, q, gain, support, finite, and complete-axis gate in all ten cells at
both 5k and 25k, plus the registered action-amplitude and predicted-vs-realized h4
calibration gates. At 25k, paired calibration must improve without higher action
distortion/clipping, and the preregistered absolute and paired terminal-outcome gates
must pass. `GS` must pass structural, finite, target, parity, and zero-weight checks;
its candidate-quality values remain causal observations rather than vetoes.

## Immutable launch lifecycle

The only launch topology is Slurm batch shell → `worker.py` → `train_entry.py` →
in-process `scripts.train`; there is no `srun` and no midpoint process. The array is
exactly `0-19%20`. A CPU report/gate job is submitted with `afterok` on the whole
array. The worker rejects inherited `TREEWM_STOP_AFTER_UPDATE`, validates scratch
generation zero or the exact current requeue lineage, and accepts success only after
the 25k checkpoint, complete 25-row final-evaluation artifact, and `COMPLETED.json`
form the terminal triplet. USR1 requeue and cancellation use create-exclusive,
fsynced state transitions and fail closed.

Any launch6 snapshot is read-only, nonsymlinked, byte-verified, and binds the final protocol,
manifest, source/runtime, all resolved configs, all four audits, lifecycle scripts,
reporter, and tests. `scripts/__init__.py` and `configs/__init__.py` are exact
empty-file import markers recorded in `campaign.SNAPSHOT_IMPORT_FILES`; both also enter
the trainer source fingerprint because their presence controls the sealed module-import
semantics.
Static preflight freezes that exact inventory once; copying and the isolated snapshot
bootstrap reopen and verify only the same mapping, so a later mutation or change to the
campaign file list cannot redefine the sealed execution tree.

Scheduler access is fail-closed. The package binds the canonical root-admin Slurm
configuration and Lua policy bytes across preclaim, exact-name absence checks,
`sbatch --test-only`, submission, reconciliation, cancellation, and requeue. The
submission preclaim makes seven read-only scheduler calls, requires zero matching jobs
before and after, and tests the exact whole-array `afterok` report dependency without
creating a job. A sealed copy of the originally authenticated Slurm configuration may
be used only for exact-ID reconciliation, cancellation, or requeue after a job is
accepted; it is never authorized for submission.

## Commands

Use only the manifest-pinned Python. Control-plane commands use `-I -S -B`:

```bash
PY=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
PKG=experiments/23-treewm-executable-prefix-repair-pilot-v1

"$PY" -I -S -B "$PKG/campaign.py" --verify
"$PY" -I -S -B "$PKG/submit.py" --scheduler-test
"$PY" -I -S -B "$PKG/submit.py" --test-only
```

All commands above are read-only. `--scheduler-test` performs only the authenticated
Slurm control-plane observation, exact-name zero-job checks, and the two
`sbatch --test-only` probes; it requires zero scheduler mutation calls and zero
persistent writes. `submit.py` defaults to `--test-only`, replays
the frozen audits through its hash-bound isolated runtime, recomposes all 20 direct
Hydra configs, checks Bash/Slurm test-only surfaces, and creates no snapshot, output,
or job. The worker remains isolated with `-I -S -B`; only the trainer entry uses
`-P -S -B`, intentionally, so the per-cell `PYTHONHASHSEED` is honored while unsafe
implicit import paths and site startup remain disabled.

Before any launch attempt, the copied-tree path must also pass its explicit regression:

```bash
"$PY" -I -S -B "$PKG/submit.py" --snapshot-test
```

`--snapshot-test` runs the production snapshot inventory, seals a real read-only copy
under a private task-specific `/tmp` tree, and executes the full isolated copied-tree
preflight: all four real audit replays plus all 20 direct Hydra compositions. It never
checks or contacts Slurm clients and never creates submission/run state. It also runs
one representative sealed launch through the real `train_entry.py` bridge under
`-P -S -B`, hides CUDA, calls the Hydra-decorated `scripts.train.main` only with
`--cfg job --resolve`, proves that `configs` came from the sealed package marker, and
requires the parsed config to equal the frozen resolved-config row. Temporary
snapshot and library-cache files are permission-restored and removed before success is
reported. A successful `--test-only` does not substitute for this check; this command
must pass for the final launch namespace and protocol before `--submit` is authorized.

Submission is intentionally explicit and was not run during package construction:

```bash
"$PY" -I -S -B "$PKG/submit.py" --submit
```

For an existing sealed submission, cancellation is separately explicit:

```bash
"$PY" -I -S -B "$PKG/cancel.py" \
  --test-only --submission-root /absolute/path/to/state/submission
"$PY" -I -S -B "$PKG/cancel.py" \
  --cancel --submission-root /absolute/path/to/state/submission
```

The afterok report job invokes `report.py --publish`. A read-only assembly check uses
the exact snapshot/submission identities:

```bash
"$PY" -I -S -B "$PKG/report.py" --test-only \
  --snapshot-root /absolute/path/to/snapshot \
  --submission-root /absolute/path/to/state/submission \
  --submission-sha256 <64-hex-submission-id>
```

To gate an already assembled immutable report directly:

```bash
"$PY" -I -S -B "$PKG/gate.py" \
  --report /absolute/path/to/immutable-report.json
```

The report root binds the exact package protocol and manifest, trainer fingerprint,
and weight, prefix-target, resolved-config, and causal-parity audit artifacts. The gate
does not discover live run directories.

This pilot supports bounded engineering evidence only. It is not formal validation and
cannot support a 1M-training claim.
