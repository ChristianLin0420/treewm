# TreeWM support-threshold diagnostic v1

This is a read-only post-hoc diagnostic for the eight paused puzzle checkpoints in
`treewm-grounded-formal-v1`. It does not alter the training objective or weaken any
stage gate.

The tool verifies each checkpoint and its 25k stage marker against the exact formal
launch, activates the launch's sealed source snapshot, and directly opens the already
materialized cache and immutable future recipe. It samples 4,096 training anchors by a
fixed one-per-stratum counter-hash rule, recomputes the exact Hungarian branch targets,
and reports recall, precision, raw KEEP, top-one fallback, multi-child, and a clearly
labelled static node-count proxy for thresholds 0.25 through 0.55 in increments of
0.025. It also reports the intercept-only projection implied by balanced BCE; that row
is diagnostic evidence, not a claim about the result of retraining.

Artifacts are self-hashed, content-addressed, published with exclusive creation and
mode `0444`, and written only under
`outputs/treewm-support-threshold-diagnostic-v1`. The program rejects an output root
inside the formal campaign tree and verifies that all formal run-tree metadata remain
unchanged before and after scoring.

From the repository root, launch all four seeds for both puzzle settings with:

```bash
sbatch experiments/diagnostics/treewm-support-threshold-v1/support_threshold.slurm
```

The Slurm array uses one GPU per task and the same pinned Python and compute-node Slurm
client as formal campaign 14. `TREEWM_SOURCE_ROOT` (default: Slurm submit directory)
selects the preferably immutable diagnostic source snapshot, while
`TREEWM_PROJECT_ROOT` selects the live project containing paused checkpoints and the
separate diagnostic output root. No submission is performed by this package itself.
