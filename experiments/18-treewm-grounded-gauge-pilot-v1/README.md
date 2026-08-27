# Exp18: grounded latent-gauge causal pilot

This package is a bounded engineering pilot. It is not formal validation and cannot support a 1M claim.

The pilot starts every run from scratch with fresh seeds 104 and 105 on the five settings that exposed the exp15 gauge failure: antmaze-large, scene, puzzle-3x3, puzzle-4x4-100m, and cube-quadruple-100m. Exp15 is bound only as an engineering-aborted diagnosis; none of its checkpoints or acceptance state is eligible.

## Prospective design

The 5k stage is an exact 30-element matrix: five settings × three arms × two seeds.

- `N`: the historical exp15-C recipe with latent-gauge loss disabled and weight zero. The new objective still emits the sealed-reference gauge telemetry, so N is a nonpromotable causal control.
- `G`: the same recipe with latent gauge enabled at weight 1 and the existing shared world clipping.
- `GS`: G plus separate branch-transformer clipping at norm 1.

All three arms otherwise preserve the same optimizer, data, future recipe, grounded selector/loss weights, scheduled sampling, evaluation bank, and learned/guard-on inference contract. N never extends past 5k.

At 5k, every G or GS setting/seed cell must independently retain a recent and terminal minimum latent scale ratio of at least 0.8, bind a positive update-0 sealed reference, and pass the unchanged validation, self-fed, horizon, q, gain, support, gradient, and clipping gates. GS additionally requires complete finite world-rest and branch-transformer norm/coefficient axes. A promotable arm must also have a positive paired mean scale-retention delta versus N. Selection is immutable and ordered G, then GS; failure of both rejects the pilot.

Gauge tags are 50-update tracker means. The gate therefore checks the valid aggregate relation `mean(min(root,future)) <= min(mean(root),mean(future))` (within tolerance), not an invalid equality of those quantities, and applies the 0.8 rail directly to the logged minimum-ratio series.

The 25k array maps all 20 G/GS cells. Only the selected arm's ten cells resume their byte-bound exp18 5k checkpoints. The ten nonselected cells launch no trainer and publish hash-bound `SKIPPED_BY_SELECTION` artifacts. The terminal gate again requires every selected cell to pass the absolute gauge and unchanged method gates. Outcome acceptance requires nonzero successes and positive mean progress independently for both fresh seeds, at least one setting with nonzero success in both seeds, and at least three settings with positive progress in both seeds.

## Fail-closed DAG

`train_5000[30] -> gate_5000 -> train_25000[20] -> gate_25000`

Every dependency is `afterok`. Submission verifies the cluster's `kill_invalid_depend` policy, sealed source/runtime/package identities, exact launch mappings, an empty output namespace, and a read-only source snapshot.

## Cancellation and requeue

The batch process never sends TERM or USR1 to its local `srun` client. Signal traps only create persistent request latches in the task state directory. The remote worker polls those latches, signals its trainer directly, waits for it, verifies the exact-resume checkpoint, and publishes `CANCELLED.json` or `READY_FOR_REQUEUE.json`. The batch exits for cancellation only after `CANCELLED.json` exists. A post-child cancellation race is finalized by a second remote worker invocation, still without signaling `srun`.

Gauge-pilot checkpoints also preserve each rank's exact in-flight 50-update metric-tracker sums, counts, and histogram chunks. Resume and the exp18 worker both reject a post-update gauge checkpoint if that state is absent or malformed, so a requeue between logging boundaries cannot silently turn the next gauge or clipping point into a partial-window mean.

Safe fleet cancellation is therefore:

1. Create `CANCEL_REQUESTED` in every intended task-state directory.
2. Send `scancel --batch --signal=TERM <array-job-id>`; do not cancel the full allocation or step.
3. Wait for one durable `CANCELLED.json` per active element and verify the checkpoint metadata before considering shutdown complete.

## Commands

Run package tests and dry verification with the pinned Python. `submit.py` is dry-run by default. Only an explicit `--submit` creates a snapshot and submits the four-node DAG. This package must not be submitted, snapshotted, committed, or pushed before review.
