# Exp20: corrected grounded latent-gauge causal pilot v2 launch2

This is a bounded 5k-to-25k engineering pilot, not formal validation and not evidence for a 1M claim. It supersedes Exp18's gate implementation but consumes no Exp18 result, checkpoint, optimizer, RNG, loader, or metric state.

## Prospective design

All 30 first-stage cells start from scratch in the fresh Exp20 launch2 namespace: five settings × N/G/GS × seeds 108/109. Launch1 job 33147842 failed during Hydra startup before any optimizer update; its output, checkpoint, launch, and W&B namespaces are explicitly ineligible here.

- `N` is the nonpromotable no-gauge causal control.
- `G` enables the update-zero sealed latent gauge at weight 1 with shared world clipping.
- `GS` is G with separate branch-transformer clipping at norm 1.

Every arm otherwise shares the frozen data, optimizer, grounded_execution_v2, published-union recipe, KEEP balance, sequence sampling, evaluation bank, and learned/guard-on inference choices.

N must have an exact checkpoint and finite, complete, boundary-fresh telemetry; nonzero gradients; clip coefficients in `(0,1]`; fixed validation sampling; sealed update-zero references; consistent positive gauge ratios; and complete recent gradient/gauge axes. A high finite low-clip saturation fraction, absolute scale collapse, or finite below-threshold method scores in N are causal measurements and cannot reject the pilot.

Every G/GS cell still must satisfy every unchanged Exp18 absolute gauge, clipping, integrity, validation, self-fed, horizon, q, gain, support, and gradient threshold. A candidate must pass all ten setting/seed cells and have positive paired mean scale retention versus N. Selection remains deterministic: G first, then GS, otherwise reject.

The 25k array maps all 20 G/GS cells. Only the selected arm's ten cells resume their byte-bound Exp20 5k checkpoints; the other ten publish immutable selection-skip artifacts without launching the trainer. All selected terminal cells again satisfy the unchanged absolute/method gates. Outcome acceptance retains the frozen per-seed nonzero-success and positive-mean-progress requirements, one setting with success in both seeds, and three settings with positive progress in both seeds.

## Fail-closed lifecycle

`train_5000[30] -> gate_5000 -> train_25000[20] -> gate_25000`

Every edge is `afterok`; submission also verifies `kill_invalid_depend`, sealed package/source/runtime identities, exact mappings, the fresh namespace, and a read-only source snapshot. The objective is registered in the shared trainer, and every sealed launch invokes the historical working `scripts/train.py` Hydra entrypoint directly. Static and sealed-snapshot launch planning execute that exact command with `--cfg job --resolve` before submission.

Cancellation and requeue signals are durably latched and forwarded to the trainer, never the local `srun` client. A successful worker plus durable completion wins a late cancellation race. A signal arriving before child creation is forwarded immediately after `Popen`.

The worker also closes the exact MetricTracker boundary race: when a graceful requeue occurs after a 50-update boundary was committed but before that boundary's scalar flush, it publishes the saved complete-window metrics first and then atomically resets only the checkpointed tracker window. Model, optimizer, scheduler, loader, RNG, and horizon-generator state remain unchanged, and before/after checkpoint plus event identities are recorded.

## Verification

`submit.py --test-only --verify-files` performs static Slurm, protocol, 30/20 mapping, configuration, and full recipe-byte verification without consulting the scheduler, creating a snapshot, or submitting jobs. Only explicit `--submit` can create the sealed snapshot and four-node DAG. Root may submit only after the reviewed package is committed and pushed and an independent prelaunch verification is green.
