# Exp21: fresh all-ten gauge bridge

This package is a bounded, non-formal successor to Exp20. It is intentionally
unsealed until Exp20 publishes both immutable 5k and accepted 25k gates. Nothing in
this package authorizes a 1M run.

The fixed design is 10 settings x fresh seeds 106/107 = 20 independent one-GPU runs
from scratch. `bind_exp20.py` rereads Exp20's raw TensorBoard records, recomputes every
integrity, unchanged-method, gauge, clipping, selection, and outcome claim, and then
derives G-before-GS. It also binds the exact event and launch bytes. It rejects direct
Exp15/Exp16/Exp18 identities and never consumes their reports, checkpoints, or recipes.

Exp21 applies the selected recipe unchanged across all ten settings: published-union
anchors, balanced KEEP, sequence scheduled sampling, `grounded_execution_v2`, learned
guard-on planning, and `domain_raw` clipped e4. The selected clipping switch is the
only G/GS difference. The isolated `train_entry.py` registers the new bounded objective
at runtime and accepts only an argv/environment tuple reproduced from a sealed launch;
direct Hydra composition cannot train.

Exact requeue also preserves the sealed 50-update telemetry cadence. If a graceful
checkpoint lands after an optimizer update but before that boundary's scalar flush,
the worker transactionally publishes the saved complete MetricTracker window and
resets only that tracker window before exact resume. A checkpoint at the exact 25k
boundary is resumable so the trainer can publish its immutable boundary marker.

At exactly 25,000 updates, every one of 20 cells must clear the unchanged method,
gauge, gradient, finite-telemetry, and bounded-clipping gates. The prospective monitor
bank is exactly five episodes per run (one fixed fallback seed for each of tasks 1..5).
Before any later 1M package may be designed, both seeds must each have at least one
success and positive 10-setting macro distance-reduction fraction, at least one setting
must have nonzero success in both seeds, and at least 6/10 settings must have positive
progress in both seeds.

Safe preparation checks (no snapshot or job):

```bash
/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python \
  experiments/21-treewm-grounded-gauge-all-ten-bridge-v2/submit.py --test-only
```

While Exp20 is incomplete this returns a successful static verification with
`launch_allowed=false`. After acceptance, sealing requires an explicit
`bind_exp20.py --publish`; actual submission separately requires explicit
`submit.py --submit`. Neither command is run during package preparation.
