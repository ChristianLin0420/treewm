# TreeWM effective-gradient audit v1

This package performs a read-only scale audit of the ten seed-zero checkpoints paused
at update 25,000 in `treewm-grounded-formal-v1`. It is diagnostic evidence about loss
and gradient scales, not efficacy evidence and not authority to resume training.

Every job verifies the exact launch, checkpoint, stage marker, sealed experiment-14
source snapshot, cache, normalizer, and complete train/validation recipes. It selects
three fixed 16-example batches from each split using disjoint counter-hashed rank
strata. Each individual batch spans the full recipe rank range.

The same six batches and RNG seeds are audited under three locked recipes:

- `baseline-exact`: the checkpoint's unbalanced KEEP and historical teacher-action,
  step-sampled multistep objective.
- `candidate-conservative`: balanced KEEP plus grounded predicted-action execution,
  sequence sampling, and conservative component weights.
- `candidate-control`: the same candidate with doubled endpoint/control emphasis.

Candidate branch selection uses action RMS, decoded task-coordinate endpoint RMS, and
the bounded horizon error `1 - p(target horizon)`. The supervised horizon component
remains cross-entropy; both selector horizon error and horizon cross-entropy are logged.

For each effective branch term plus the combined multistep term, the artifact records
gradient L2 norm, share, and cosine with the total objective for the encoder, branch
transformer, dynamics, controllability, action head, horizon head, KEEP head, decoder,
and their deduplicated union. Grounded multistep component scalars are preserved
separately even though their gradient is intentionally reported as one effective
`multistep` term.

Candidate scale-only guardrails require: shared-union candidate:baseline total norm at
most 1.5; every named-module ratio at most 2.0; no term share over 0.80; and finite
action/horizon/KEEP/decoder path norms above `1e-8`. A failed guardrail remains a valid
diagnostic result. Any non-finite input, loss, metric, or gradient fails without
publishing.

Artifacts are self-hashed, content-addressed, exclusively created as mode `0444`, and
written only beneath `outputs/treewm-gradient-audit-v1`. The formal run tree, config,
model tensors, and `Parameter.grad` state are checked for non-mutation.

Launch all ten independent GPU jobs from an immutable source root with:

```bash
sbatch experiments/diagnostics/treewm-gradient-audit-v1/gradient_audit.slurm
```

`TREEWM_SOURCE_ROOT` selects diagnostic/candidate code; `TREEWM_PROJECT_ROOT` selects
the live project containing paused checkpoints and the separate output root.
