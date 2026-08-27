# TreeWM grounded repair all-ten bridge v1

This package defines a fresh, bounded 25,000-update bridge across every formal
setting. It is not formal validation, never resumes campaigns 14 or 15, and cannot
be extended beyond 25k under the `treewm_v2_grounded_repair_pilot_v1` objective.

## Locked design

One Slurm array expands to exactly 40 independent one-GPU runs:

`10 settings × 2 global grounded-loss scales × 2 fresh seeds (102, 103)`.

The deterministic array mapping is
`((setting_index * 2) + arm_index) * 2 + seed_index`. Settings retain the formal
order: scene, puzzle-3x3, puzzle-4x4-100m, cube-double, cube-triple,
cube-quadruple-100m, antmaze-large, antmaze-giant, humanoidmaze-medium, and
humanoidmaze-large.

- **F** exactly reproduces exp15-C: world LR `3e-5`; selector weights
  `(action=1, endpoint=1, horizon=0.25)`; grounded loss weights
  `(latent=0.25, action=0.5, horizon=0.25, endpoint=0.5)`.
- **H** changes only the four grounded loss weights, globally halving them to
  `(latent=0.125, action=0.25, horizon=0.125, endpoint=0.25)`. Its selectors and
  world LR are identical to F.

There is no domain-specific scale. Both arms use grounded execution v2, balanced
KEEP supervision at decision threshold `0.5`, sequence-level scheduled sampling at
`p=0.25` after a 5k warmup, a detached self-fed parent, depth three, published-union
recipe anchors, `domain_raw` decoded scoring, clipped four-action execution, and
canonical `learned_guard_on` inference.

Every run validates and checkpoints every 1k, evaluates at 12.5k and 25k, and keeps
the 1M scheduler horizon while optimization remains hard-capped at 25k. The common
validation sample is pinned by `train.validation_sample_seed=1701`. Terminal
evaluation uses five episodes for each of task IDs 1--5. `eval.seed=2718` produces
the exact same 25 terminal episode rows in every arm, setting, and training seed.

## Why this bridge exists

The conservative full scale passed the meaningful shared-module gradient screen in
the five exp15 settings, while raw all-ten gradient diagnostics warned about scale in
cube-triple and the humanoid settings. This bridge tests one global half-loss scale
against the exact full exp15-C scale. It records every weight in the manifest and
forbids per-domain tuning.

## Fail-closed selection

`report.py` applies one preregistered deterministic rule:

1. Select F only when all 40 runs have exact integrity; F passes at least 18/20
   scientific runs, both seeds in at least 8/10 settings, and at least one seed in
   every setting; and F has at least one terminal success, positive mean distance
   reduction, and at least 12/20 positive-progress runs.
2. Otherwise apply the identical replicated scientific and outcome quorums to H,
   then select H only when paired mean terminal success and distance reduction for H
   are each noninferior to F on the identical episode bank.
3. Otherwise select no recipe. An all-zero-success arm can never be promoted.

The order is always F then H. H cannot replace an eligible F, and no setting can
choose its own arm.

On `--publish`, the report writes `reports/acceptance.json`,
`reports/acceptance.md`, and `reports/SELECTED_RECIPE.json`. The last artifact is
written for both acceptance and rejection; formal consumers must require
`selected=true`, then verify its self-hash, acceptance hash, protocol, trainer
source/runtime, manifest, selected recipe hash, and all 20 selected-run config hashes.
`acceptance_sha256` binds the exact pretty-printed `acceptance.json` bytes, while
`bridge_acceptance_sha256` binds that report's canonical `report_sha256` self-hash.

## Provenance and lifecycle

Exp16 has a hard exp15 prerequisite. Every dry-run, launch command, worker requeue,
and report revalidates
`outputs/treewm-grounded-repair-pilot-v1/report/acceptance.json` and the exp15 sealed
launch plan. The report must have the exact campaign/status, `accepted=true`, arm C,
40/40 integrity, the exact 40-cell matrix, and every preregistered aggregate gate
true. The launch plan must self-hash and match the pinned exp15 protocol, trainer
source, runtime, and evaluation bank. Exp16 binds both the canonical and exact-byte
acceptance hashes, the launch-plan hashes, and their combined prerequisite hash into
every config, launch protocol, checkpoint/completion identity, acceptance report,
and selected-recipe artifact. Direct `sbatch` lacks the guarded prerequisite export
and fails before execution. If exp15 rejects or remains incomplete, exp16 cannot be
dry-run as submission-ready or launched.

The guarded submit path binds the active trainer source, pinned py311 runtime,
resolved override list, package protocol, compatible data contracts, raw data
manifests, normalizers, train/validation manifests, calibration, published future
recipes, evaluation banks, and recipe-producer source/runtime. Submission re-hashes
all external recipe records, creates a content-addressed source snapshot, seals it
read-only, and launches only from that snapshot.

Each array element requests one GPU, 12 CPUs, and 64G. USR1 is forwarded seven
minutes before the four-hour limit. Exit 75 is requeued only after the worker loads
`latest.pt` and verifies optimizer, scheduler, RNG streams, loader cursor,
checkpoint-manager state, and all launch identities. Cancellation latches take
precedence. The strict CPU report is submitted `afterok` on the complete array.

After exp15 has published an accepted report, dry-run with the pinned interpreter
revalidates and binds it while creating neither a snapshot nor jobs. Before that
report exists and accepts, this command intentionally fails closed:

```bash
PY=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
$PY experiments/16-treewm-grounded-repair-all-ten-bridge-v1/submit.py
```

Only explicit `--submit` may snapshot and submit:

```bash
$PY experiments/16-treewm-grounded-repair-all-ten-bridge-v1/submit.py --submit
```

This prepared package is intentionally not submitted by its construction task.
