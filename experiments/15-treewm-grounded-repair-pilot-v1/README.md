# TreeWM grounded repair pilot v1

This package defines a fresh, bounded 25,000-update repair screen. It is not formal
validation, never resumes campaign 14, and cannot be extended beyond 25k under the
`treewm_v2_grounded_repair_pilot_v1` objective.

## Locked design

The single Slurm array expands to exactly 40 independent one-GPU runs:

`5 settings × 4 arms × 2 fresh seeds (100, 101)`.

The deterministic array mapping is
`((setting_index * 4) + arm_index) * 2 + seed_index`. Settings are ordered
antmaze-large, scene, puzzle-3x3, puzzle-4x4-100m, and cube-quadruple-100m.

- **A** is the preregistered teacher-action control at world LR `3e-5`.
- **B** uses conservative grounded selection/loss weights at world LR `1e-4`.
- **C** uses the same conservative grounded weights at world LR `3e-5` and is the
  only preregistered promotable candidate.
- **D** uses the stronger endpoint/action control weights at world LR `3e-5`.

Arms B and D are mechanistic sensitivities. `report.py` has no path that substitutes
either for C after outcomes are visible.

All cells use balanced KEEP supervision with decision threshold `0.5`, sequence-level
scheduled sampling at `p=0.25` after a 5k warmup, depth three, published-union recipe
anchors, `domain_raw` decoded scoring, clipped four-action execution, gain LR `3e-4`,
zero gain weight decay, and a gain update every optimizer step. Training and validation
use the full published recipe unions, validate/checkpoint every 1k, and retain the 1M
scheduler horizon while the objective hard-caps optimization at 25k.

The validation sample is common and pinned by `train.validation_sample_seed=1701`.
Periodic evaluation runs at 12.5k and 25k; terminal evaluation is five episodes for
each built-in task. `eval.seed=2718` makes the actual terminal episode rows identical
across every arm and training seed. The worker verifies those episode rows directly.

## Inference choice

This pilot is prospectively locked to `inference_choice.profile=learned_guard_on`.
Frozen-checkpoint scorer/guard arms are descriptive diagnostics and cannot select or
change this fresh pilot after its protocol is locked. The profile table remains in the
manifest to make the scorer/guard mapping explicit, but validation rejects any profile
other than canonical learned scoring with the first-edge guard enabled.

## Provenance and lifecycle

The explicit submit path verifies and binds the active trainer source, pinned py311
runtime, resolved override list, package protocol, compatible data contract, raw data
manifest, normalizer, train/validation manifests, calibration, published future
recipe, and recipe-producer source/runtime. Submission re-hashes every external recipe
record, copies only executable training/config/package inputs into a content-addressed
snapshot, seals all snapshot files read-only, and launches from that snapshot.

The GPU array requests one GPU, 12 CPUs, and exactly 64G per element. USR1 is forwarded
to the trainer seven minutes before the four-hour limit. Exit 75 is requeued only after
the worker loads `latest.pt` and verifies optimizer, scheduler, RNG streams, loader
cursor, checkpoint-manager state, and all launch identities. A persistent cancellation
latch always takes precedence over requeue. Submission uses an exclusive claim plus
atomic plan/progress/receipt artifacts, so an ambiguous partial submission cannot be
retried into a duplicate array.

Dry-run with the pinned interpreter (no snapshot and no jobs are created):

```bash
PY=/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python
$PY experiments/15-treewm-grounded-repair-pilot-v1/submit.py
```

Only an explicit `--submit` creates the snapshot and submits the 40-element array plus
its after-ok CPU report:

```bash
$PY experiments/15-treewm-grounded-repair-pilot-v1/submit.py --submit
```

To cancel a live element permanently, create
`<run-root>/state/arrays/array-job-<array_job_id>_<task_id>/task-<task_id-2digits>/CANCEL_REQUESTED`
before signalling the element. Never remove a cancellation latch to reuse an old job
identity.

## Preregistered report

The strict report requires identity and structural integrity for all 40 runs, including
the common fixed validation sample, exact 1k validation axis, midpoint and terminal
rollouts, and exact common episode bank. For arm C, both seeds must pass in at least
four of five settings. Each passing run must satisfy:

- final validation and self-fed multistep loss no more than `1.10 ×` its observed
  minimum;
- horizon CE below both `ln(5)` and the empirical constant-prior entropy;
- positive q advantages over z and a dimension-matched random projection;
- gain rank `>=0.10`, pairwise accuracy `>=0.52`, eligibility `>=0.20`, and the locked
  ordered-pair/coverage minima;
- support recall `>=0.50` and precision `>=0.25`;
- finite nonzero aggregate world/gain shared-module gradients with the locked
  clipping-tail constraint. This is deliberately not a single-term KEEP-head gradient
  share criterion.

Across the ten C runs, the report additionally requires at least one terminal success,
positive mean distance reduction, positive distance reduction in at least six runs,
and paired mean success and distance-reduction noninferiority to A on the identical
episode bank. Therefore an all-zero C/A tie is rejected. Even acceptance is labeled
`accepted_for_fresh_formal_campaign_design`, with `formal_validation=false`.
