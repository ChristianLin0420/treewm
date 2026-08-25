# Upstream provenance and local lifecycle patch

This directory was mechanically vendored from the official Reversal Q-Learning
(RQL) repository, <https://github.com/aoberai/rql>, at the immutable commit:

```
229c956efb4494c2b9bb0bbddbd67b761c93f1cc
```

The commit was authored on 2026-06-17 (`add teaser gif`). `LICENSE.md` is the
upstream MIT license copied without modification. The initial vendor copy
included the complete upstream `agents/`, `envs/`, `utils/`, and `assets/`
trees plus `main.py`, `README.md`, `hyperparameters.sh`, and
`requirements.txt`.

For independent verification, selected files in the unmodified commit have
these SHA-256 digests:

| Upstream path | SHA-256 at `229c956` |
| --- | --- |
| `main.py` | `f13be07a89d92c7ff7b9e8088573cf441d68ccf18ebca7587d00d24bd1c8c08e` |
| `agents/rql.py` | `486c7495cd676ce7a224e5a560e3cf30af26222f7bcda93c2df7412c12629388` |
| `utils/evaluation.py` | `5fec38e442cad756e9060998e920297fed74b289e621db6463187667f6a05ef3` |
| `utils/log_utils.py` | `7bf01cfc6f984ea4b17dfb0224d1400e7a0d475b72b2557487d30378a84c3705` |
| `utils/networks.py` | `444ea6031ff500a8029619aa2c924bacc5016f75cdeb867082765dc4ec5f696a` |
| `envs/env_utils.py` | `d0a7b1dd7c4437a828454cd61d98a78929eb8cffb78335c62dc198ee82b5f98c` |
| `LICENSE.md` | `a944518d183e6b092d6ad6eb4d6cf8e10f6a6b321afed8832989b116fbe58247` |
| `hyperparameters.sh` | `6554f19ee0468acd96a71693fadecb2a68580bb2b36150319051501257f6caf1` |

## Local changes

The RQL loss, reversing-flow update, target updates, optimizer, model widths,
ensemble semantics, and tuned task hyperparameters remain upstream. Local
changes are limited to formal-run infrastructure:

- The terminal update is controlled by `offline_steps`; the current formal
  protocol uses the upstream default of exactly 1,000,000 updates. Evaluation,
  checkpoint, resume, and completion state all use that same absolute terminal
  step. Scheduler array width is intentionally outside this single-run code.
- `main.py` uses required stable run directory/name/W&B ID and campaign
  protocol hash, W&B `resume="allow"`, atomic versioned checkpoints, deferred
  `SIGUSR1`/`SIGTERM` handling, an internal wall-clock deadline, exit code 75,
  append-only logs, and a completion sentinel written only after the final
  checkpoint, full 50-episode evaluation, and W&B flush.
- The checkpoint contains the host-materialized Flax agent (including its JAX
  RNG and optimizer), completed `global_step`, explicit `next_step`, global
  NumPy/Python RNG states, the rotating 100M shard index, pending/last/final
  evaluation state, elapsed time, stable run identity, campaign protocol hash,
  hashes of every runtime-critical local source file, and the Python/JAX/JAXlib/
  Flax/Optax/Distrax/Einops/ml-collections/Gymnasium/OGBench/W&B/NumPy/MuJoCo
  software versions. Software-version drift
  is resume-incompatible; OS/architecture/libc details are recorded separately
  without binding to a hostname so Slurm may resume on another node.
  The dependency-light `trainer_code_fingerprint()` helper exposes the same
  authoritative source manifest to campaign completion validation, and the
  completion sentinel stores the full immutable identity as well as its hash.
- `utils/evaluation.py` deterministically seeds each episode from training
  seed, absolute evaluation step, and episode index, and checks a stop callback
  before/after every environment step. An interrupted evaluation remains
  pending and is repeated before the next training update.
- `utils/csv_logger.py` provides append-only, one-row-per-step CSV logging with
  no header rewrite, truncation, or duplicate step on resume.
- `utils/resume.py` provides dependency-light atomic I/O, identity validation,
  RNG capture, signal/deadline handling, evaluation-boundary logic, and
  absolute-step shard selection. The 100M loader fails closed unless its
  dedicated directory contains exactly the official contiguous train shards
  `000`–`099` and all paired validation shards, with no additional NPZ files.
- `utils/networks.py` and `agents/rql.py` enable Flax `remat` for the actor and
  value MLPs. The flag is required true and is recorded in flags, metadata,
  checkpoint identity, and completion output.
- `envs/env_utils.py` accepts an explicit standard OGBench cache and exposes
  the existing wrapper stack for the custom 100M loader, avoiding an unrelated
  standard-dataset download.
- For offline-only runs (`online_steps=0`), `main.py` does not allocate the
  upstream default 100M `ReplayBuffer`. Upstream aliases that copied buffer as
  `train_dataset` and never mutates it on the offline path, so sampling still
  uses the same `Dataset.sample` implementation and the official update is
  unchanged. This prevents otherwise catastrophic host-memory multiplication
  across workers.

No credential is accepted by the trainer CLI or stored in source, flags,
metadata, checkpoints, CSV files, or the completion sentinel. W&B/Hugging Face
authentication is supplied only by the job environment.
