# Exp22: fresh grounded-gauge formal v1

Exp22 is a preregistered fresh 1M formal campaign. It is not a continuation of a
pilot checkpoint. Its forty models use ten settings crossed with training seeds
220, 221, 222, and 223; those seeds are outside the 0–3 and 100–111 banks used by
earlier formal or method-selection work. Every run starts in the new
`treewm-grounded-gauge-formal-v1` namespace.

## Promotion authority

Formal launch remains fail-closed until `bind_prerequisites.py` can reproduce both
accepted decisions from raw evidence:

1. Exp20 launch2: replay all thirty 5k N/G/GS event streams, apply structural-only
   treatment to nonpromotable N, retain every absolute G/GS gauge, clipping,
   integrity, method, causal, and G-first selection gate, then replay the ten
   selected 25k outcomes and ten exact skips.
2. Exp21: replay all twenty all-setting 25k event/checkpoint rows, including 20/20
   method health, per-seed success and positive macro progress, at least one setting
   with nonzero success in both seeds, and the fixed 6/10 both-seed positive-progress
   quorum.

The binder compares each independently regenerated object byte-for-object with its
published artifact, rejects any Exp14–18 ancestry token/field, inventories the exact
launch/event/checkpoint bytes, and derives the complete G or GS recipe from the
Exp21 acceptance. Exp22 performs no outcome-based recipe selection. Sealing the
binding changes the Exp22 protocol lock; until then `submit.py --test-only` must
report a blocked status and `--submit` must fail.

`train_entry.py` first checks the exact wrapper arguments and environment. After it
execs `scripts/train.py`, the shared trainer independently rederives the canonical
run, arguments, environment, package protocol, and prerequisite receipt before any
dataset or model construction. Direct Hydra composition therefore has no launch
authority, even when copied prerequisite hashes are internally consistent.

## Training and gates

The guarded DAG is:

`40×2k → gate → 40×25k → gate → 40×100k → gate → 40×1M → gate → 200 final cells → aggregate`

- 2k requires all forty exact checkpoints and target-current finite numerical,
  gradient, clipping, validation-sample, and latent-gauge telemetry.
- 25k again requires all forty integrity rows and at least 3/4 complete method-gate
  passes in every setting.
- 100k requires current numerical/gauge health plus a prospective, protocol-bound
  five-episode monitor rail with nonzero fleet success and positive progress.
- After 100k, continuation and the exact-1M gate depend only on integrity and
  numerical health. Later outcomes cannot stop or select the formal recipe.

Graceful preemption preserves model, optimizer, scheduler, loader position, all CPU
and CUDA RNG streams, planner/evaluation/visualization generators, horizon state,
and MetricTracker state. Every Exp22 checkpoint carries the shared trainer's strict
`post_update_cadence` state. A normal graceful stop occurs only after all logging,
diagnostics, validation, periodic checkpoint/evaluation, and visualization work due
for the committed update is complete. An interrupted evaluation or visualization is
durable only with explicit replay intent and is completed before the next optimizer
update; no package-local checkpoint repair is performed. Stage completion wins over
a racing cancellation only when the durable completion marker is already valid.

## Heldout final evaluation

The final array contains 40 models × 5 tasks. Each cell evaluates learned and BFS
rails on the same ordered fifty environment seeds from the locked table. The table
is common across all four model seeds, disjoint from every prospective monitor seed,
and contains no duplicate across settings. Each episode is durably appended with its
exact seed and planner-generator state, so requeue resumes only a validated prefix.

The aggregate requires all 200 cells and all raw episodes. Pooled episode summaries
are descriptive only. The primary paired learned-minus-BFS inference unit is the
training seed (four replicates), using the preregistered two-sided 95% t interval with
`df=3` and critical value `3.182446`.

## Root workflow

After Exp20 and Exp21 acceptance exists, run the binder without `--publish`, inspect
the full replay, then publish the binding and re-run package tests, full byte
verification, and `submit.py --test-only`. Submission is permitted only from a
committed and pushed tree after independent prelaunch verification. The submitter
then creates one read-only source snapshot and one guarded afterok DAG; this package
itself never authorizes an ad-hoc partial launch.
