# Campaign provenance

Checked on 2026-08-22 UTC.

## Primary sources

- Paper: Aditya Oberai, Seohong Park, and Sergey Levine, *Reversal
  Q-Learning*, arXiv:2606.17551, <https://arxiv.org/abs/2606.17551>.
- Official implementation: <https://github.com/aoberai/rql>, immutable commit
  `229c956efb4494c2b9bb0bbddbd67b761c93f1cc` (2026-06-17).  The official
  per-setting commands are in `hyperparameters.sh`; its SHA-256 at that commit
  is `6554f19ee0468acd96a71693fadecb2a68580bb2b36150319051501257f6caf1`.
- OGBench API/task naming and public datasets: OGBench 1.2.x,
  <https://github.com/seohongpark/ogbench>.  The official RQL requirements leave
  OGBench unpinned; the reviewed formal environment uses OGBench 1.2.1.
  OGBench registers five numbered single-task variants per setting and stores
  standard data at
  <https://rail.eecs.berkeley.edu/datasets/ogbench/>.
- The official RQL README identifies `puzzle-4x4-play-100m-v0` and
  `cube-quadruple-play-100m-v0` as the paper's large datasets.  Each directory
  has 100 numbered train shards and a paired validation shard per number.

## Protocol resolution

The public repository's released commands specify one million offline steps.
RQL Appendix Table 1 instead describes two million gradient steps for its
four-seed table, with 50 evaluation episodes and 95% confidence intervals.
This repository's evidence ledger records that distinction in
[`../../BASELINES.md`](../../BASELINES.md).  At the user's explicit request,
this formal campaign pins the released-command budget of exactly 1,000,000
updates, task IDs 1–5, four seeds, and a 50-episode final evaluation.  Results
must therefore be labelled as the 1M released-command protocol, not as a
strict reproduction of the paper's 2M step budget.

The semantic manifest hash is stored in `protocol.sha256`.  It hashes compact,
sorted-key JSON after parsing, so whitespace and key order do not change the
protocol identity.  Every trainer receives this 64-character hash, embeds it
in checkpoint identity and completion metadata, and refuses mismatched resume.

## Separation of concerns

- `manifest.json` is the sole experimental matrix.
- `campaign.py`, `dispatcher.py`, `prepare_data.py`, `gpu_preflight.py`,
  `aggregate.py`, `submit.py`, `train.slurm`, `aggregate.slurm`, and
  `stage_data.slurm` are local campaign infrastructure.  They contain no RQL
  learning rule.
- `upstream_rql/` is a vendored source snapshot plus the narrowly documented
  lifecycle/rematerialization patch.  Its exact source inventory and
  modification boundary are recorded in
  [`upstream_rql/UPSTREAM_PROVENANCE.md`](./upstream_rql/UPSTREAM_PROVENANCE.md).
- Trainer checkpoints also carry a sorted SHA-256 manifest of every
  runtime-critical patched source file, preventing resume after silent code
  drift even when the campaign manifest is unchanged.

No W&B or Hugging Face token is recorded in the manifest, source, provenance,
Slurm scripts, command metadata, checkpoints, or completion sentinels.
