# Exp23 executable-prefix repair pilot v1 launch3

This is the sealed, launch-capable but unsubmitted launch3 package for a bounded 20-cell
engineering pilot. It compares the corrected Exp20 gauge + separate-branch-clipping
recipe (`GS`) with the same recipe plus executable-prefix grounding (`GSEP`).
The active campaign identity is `treewm-executable-prefix-repair-pilot-v1-launch3`,
with a fresh `outputs/treewm-executable-prefix-repair-pilot-v1-launch3` namespace,
`outputs/.exp23-6e55bb3083712144.transaction.lock`, and `exp23-launch3-*` run names.
No launch3 submission, snapshot, scheduler job, W&B run,
checkpoint, optimizer update, or scientific output was created while building or
verifying this package.

Launch1 and launch2 are permanently ordered negative provenance. Each preserved
read-only source snapshot was independently byte-matched 137/137 to its recorded
commit; neither journal itself claims git provenance. Launch1 aborted before the
submission contract because its isolated audit inputs were unavailable. Launch2 also
aborted before the contract: its causal replay incorrectly included the newly created
controlled `state/submission` tree in an output fingerprint. Neither attempt produced a
submission SHA, contract, job ID, receipt, scientific run, checkpoint, W&B run, or
optimizer update. Their exact protocols, manifests, inventories, claims, fingerprints,
and journals are sealed in order in `manifest.superseded_launches`; neither namespace,
identity, snapshot, nor state may be reused or resumed by launch3.

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

Any launch3 snapshot is read-only, nonsymlinked, byte-verified, and binds the final protocol,
manifest, source/runtime, all resolved configs, all four audits, lifecycle scripts,
reporter, and tests. `scripts/__init__.py` is included as an exact supplemental import
file with the empty-file SHA-256 recorded in `campaign.SNAPSHOT_IMPORT_FILES`.
Static preflight freezes that exact inventory once; copying and the isolated snapshot
bootstrap reopen and verify only the same mapping, so a later mutation or change to the
campaign file list cannot redefine the sealed execution tree.

## Commands

Use only the manifest-pinned Python. Control-plane commands use `-I -S -B`:

```bash
PY=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
PKG=experiments/23-treewm-executable-prefix-repair-pilot-v1

"$PY" -I -S -B "$PKG/campaign.py" --verify
"$PY" -I -S -B "$PKG/submit.py" --test-only
```

Both commands above are read-only. `submit.py` defaults to `--test-only`, replays
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
checks or contacts Slurm clients and never creates submission/run state. Temporary
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
