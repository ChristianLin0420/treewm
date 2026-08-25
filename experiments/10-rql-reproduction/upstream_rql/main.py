"""Paper-aligned RQL training with atomic, Slurm-safe continuation.

The learning update, model, and tuned agent configuration are vendored from
official RQL commit 229c956efb4494c2b9bb0bbddbd67b761c93f1cc. This entry
point adds only run-lifecycle infrastructure needed by the four-hour cluster
queue; see ``UPSTREAM_PROVENANCE.md`` for the complete modification list.
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import sys
import time
from datetime import datetime, timezone

import flax
import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets, wrap_envs
from envs.ogbench_utils import make_ogbench_env_and_datasets
from utils.csv_logger import CsvLogger
from utils.datasets import Dataset
from utils.evaluation import evaluate
from utils.log_utils import get_flag_dict, get_wandb_video, setup_wandb
from utils.resume import (
    COMPLETION_SCHEMA_VERSION,
    GRACEFUL_EXIT_CODE,
    GracefulStop,
    StopController,
    atomic_json_dump,
    atomic_pickle_dump,
    capture_rng_state,
    collect_runtime_provenance,
    discover_official_100m_shards,
    evaluation_due,
    install_stop_handlers,
    load_checkpoint,
    make_checkpoint,
    restore_rng_state,
    shard_index_for_step,
    stable_json_hash,
    trainer_code_fingerprint,
    to_jsonable,
)


UPSTREAM_DIR = pathlib.Path(__file__).resolve().parent
UPSTREAM_COMMIT = "229c956efb4494c2b9bb0bbddbd67b761c93f1cc"
FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'RQL-Reproduction', 'W&B run group.')
flags.DEFINE_string('run_name', None, 'Stable run name (required).')
flags.DEFINE_string('run_dir', None, 'Stable per-run output directory (required).')
flags.DEFINE_string('wandb_id', None, 'Stable W&B run ID reused across requeues (required).')
flags.DEFINE_string('wandb_project', 'rql-reproduction', 'W&B project.')
flags.DEFINE_string('wandb_entity', None, 'Optional W&B entity.')
flags.DEFINE_string('protocol_sha256', None, 'Canonical campaign protocol SHA-256 (required).')
flags.DEFINE_enum('wandb_mode', 'online', ['online', 'offline', 'disabled'], 'W&B mode.')
flags.DEFINE_bool('resume', True, 'Restore checkpoint.pkl when it exists.')
flags.DEFINE_float(
    'walltime_seconds',
    13800.0,
    'Process wall-clock budget; 0 disables it (default leaves 10m of a 4h allocation).',
)
flags.DEFINE_bool('gradient_checkpointing', True, 'Enable Flax remat for actor and critic MLPs.')

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', '', 'Environment (dataset) name.')
flags.DEFINE_string('ogbench_standard_dataset_dir', None, 'Cache directory for standard OGBench datasets.')
flags.DEFINE_string('ogbench_dataset_dir', None, 'Directory of rotating 100M OGBench shards.')
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Absolute-step interval for rotating 100M shards.')
flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline updates.')
flags.DEFINE_integer('online_steps', 0, 'Number of online updates (formal runner requires zero).')
flags.DEFINE_integer('buffer_size', 100000000, 'Upstream online replay capacity (unused offline).')
flags.DEFINE_integer('log_interval', 5000, 'Training logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Periodic evaluation interval; 0 disables periodic only.')
flags.DEFINE_integer('save_interval', 1000000, 'Upstream-compatible checkpoint interval fallback.')
flags.DEFINE_integer('checkpoint_interval', 0, 'Atomic checkpoint interval; 0 uses save_interval.')
flags.DEFINE_integer('eval_episodes', 50, 'Episodes in periodic evaluations.')
flags.DEFINE_integer('final_eval_episodes', 50, 'Episodes in the mandatory final evaluation.')
flags.DEFINE_integer('video_episodes', 0, 'Video-only episodes in each evaluation.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_bool('sparse', False, 'Make the task sparse reward.')
flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('utd', 1, 'UTD (formal offline path performs one update per step).')

config_flags.DEFINE_config_file(
    'agent',
    str(UPSTREAM_DIR / 'agents' / 'rql.py'),
    lock_config=False,
)


def _validate_flags() -> None:
    required = {
        'run_name': FLAGS.run_name,
        'run_dir': FLAGS.run_dir,
        'wandb_id': FLAGS.wandb_id,
        'protocol_sha256': FLAGS.protocol_sha256,
        'env_name': FLAGS.env_name,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing required stable run flags: {', '.join(missing)}")
    if re.fullmatch(r'[0-9a-f]{64}', FLAGS.protocol_sha256 or '') is None:
        raise ValueError('protocol_sha256 must be a lowercase 64-character SHA-256')
    if FLAGS.seed < 0:
        raise ValueError('seed must be non-negative')
    if FLAGS.offline_steps <= 0:
        raise ValueError('offline_steps must be positive')
    if FLAGS.online_steps != 0:
        raise ValueError(
            'This requeue-safe paper runner is offline-only; use online_steps=0. '
            'Upstream online replay/environment state is not checkpointed.'
        )
    if FLAGS.log_interval <= 0:
        raise ValueError('log_interval must be positive')
    if FLAGS.eval_interval < 0:
        raise ValueError('eval_interval must be non-negative')
    if FLAGS.eval_episodes <= 0 or FLAGS.final_eval_episodes != 50:
        raise ValueError('eval_episodes must be positive and final_eval_episodes must be exactly 50')
    if FLAGS.checkpoint_interval < 0 or FLAGS.save_interval <= 0:
        raise ValueError('checkpoint_interval must be non-negative and save_interval must be positive')
    if FLAGS.walltime_seconds < 0:
        raise ValueError('walltime_seconds must be non-negative')
    if not FLAGS.gradient_checkpointing:
        raise ValueError('formal RQL reproduction requires gradient_checkpointing=true')
    if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval <= 0:
        raise ValueError('rotating 100M datasets require dataset_replace_interval > 0')


def _discover_shards() -> list[str]:
    if FLAGS.ogbench_dataset_dir is None:
        return []
    return discover_official_100m_shards(FLAGS.ogbench_dataset_dir, FLAGS.env_name)


def _run_identity(
    config,
    dataset_paths: list[str],
    run_dir: pathlib.Path,
    code_files: dict[str, str],
    code_manifest_sha256: str,
    runtime_software: dict,
) -> dict:
    return {
        'upstream_commit': UPSTREAM_COMMIT,
        'run_name': FLAGS.run_name,
        'run_dir': str(run_dir),
        'wandb_id': FLAGS.wandb_id,
        'wandb_project': FLAGS.wandb_project,
        'protocol_sha256': FLAGS.protocol_sha256,
        'code_manifest_sha256': code_manifest_sha256,
        'code_files': code_files,
        'runtime_software': runtime_software,
        'run_group': FLAGS.run_group,
        'env_name': FLAGS.env_name,
        'seed': FLAGS.seed,
        'offline_steps': FLAGS.offline_steps,
        'online_steps': FLAGS.online_steps,
        'sparse': FLAGS.sparse,
        'p_aug': FLAGS.p_aug,
        'frame_stack': FLAGS.frame_stack,
        'utd': FLAGS.utd,
        'eval_interval': FLAGS.eval_interval,
        'eval_episodes': FLAGS.eval_episodes,
        'final_eval_episodes': FLAGS.final_eval_episodes,
        'video_episodes': FLAGS.video_episodes,
        'video_frame_skip': FLAGS.video_frame_skip,
        'gradient_checkpointing': FLAGS.gradient_checkpointing,
        'ogbench_standard_dataset_dir': (
            str(pathlib.Path(FLAGS.ogbench_standard_dataset_dir).expanduser().resolve())
            if FLAGS.ogbench_standard_dataset_dir is not None
            else None
        ),
        'dataset_replace_interval': FLAGS.dataset_replace_interval,
        'dataset_paths': dataset_paths,
        'agent': config.to_dict(),
    }


def _read_json(path: pathlib.Path) -> dict:
    with path.open(encoding='utf-8') as source:
        return json.load(source)


def _main(_argv) -> int:
    # Defer signal work to safe Python points and count wall time from process start.
    stop = StopController(FLAGS.walltime_seconds)
    install_stop_handlers(stop)
    _validate_flags()

    run_dir = pathlib.Path(FLAGS.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / 'checkpoint.pkl'
    completion_path = run_dir / 'COMPLETED.json'
    checkpoint_interval = FLAGS.checkpoint_interval or FLAGS.save_interval

    config = FLAGS.agent
    config['gradient_checkpointing'] = FLAGS.gradient_checkpointing
    dataset_paths = _discover_shards()
    code_fingerprint = trainer_code_fingerprint(UPSTREAM_DIR)
    code_files = code_fingerprint['files']
    code_manifest_sha256 = code_fingerprint['manifest_sha256']
    runtime_provenance = collect_runtime_provenance()
    runtime_software = {
        'python': runtime_provenance['python'],
        'packages': runtime_provenance['packages'],
    }
    runtime_software_sha256 = stable_json_hash(runtime_software)
    identity = _run_identity(
        config,
        dataset_paths,
        run_dir,
        code_files,
        code_manifest_sha256,
        runtime_software,
    )
    identity_hash = stable_json_hash(identity)

    if completion_path.exists():
        completion = _read_json(completion_path)
        if (
            completion.get('schema_version') != COMPLETION_SCHEMA_VERSION
            or completion.get('status') != 'complete'
            or completion.get('identity') != identity
            or completion.get('identity_sha256') != identity_hash
            or completion.get('global_step') != FLAGS.offline_steps
            or completion.get('final_eval_step') != FLAGS.offline_steps
            or completion.get('final_eval_episodes') != 50
        ):
            raise ValueError(f'Completion sentinel does not match requested run: {completion_path}')
        print(f'Run already complete at step {completion["global_step"]}: {completion_path}', flush=True)
        return 0

    resume_checkpoint = None
    if checkpoint_path.exists():
        if not FLAGS.resume:
            raise FileExistsError(f'Checkpoint exists but --resume=false: {checkpoint_path}')
        # Fail closed on code/protocol/run drift before touching the original
        # run's flags, provenance, W&B history, or datasets.
        resume_checkpoint = load_checkpoint(checkpoint_path, identity)
    else:
        stale_logs = [
            path for path in (run_dir / 'train.csv', run_dir / 'eval.csv', run_dir / 'final_eval.csv')
            if path.exists() and path.stat().st_size > 0
        ]
        if stale_logs:
            raise FileExistsError(f'Run logs exist without a checkpoint: {stale_logs}')

    flags_record = get_flag_dict()
    flags_record['agent'] = config.to_dict()
    flags_record['upstream_commit'] = UPSTREAM_COMMIT
    atomic_json_dump(flags_record, run_dir / 'flags.json')
    atomic_json_dump(
        {
            'upstream_repository': 'https://github.com/aoberai/rql',
            'upstream_commit': UPSTREAM_COMMIT,
            'upstream_license': 'MIT; see LICENSE.md',
            'identity_sha256': identity_hash,
            'identity': identity,
            'protocol_sha256': FLAGS.protocol_sha256,
            'code_manifest_sha256': code_manifest_sha256,
            'code_files': code_files,
            'runtime_software_sha256': runtime_software_sha256,
            'runtime': runtime_provenance,
            'gradient_checkpointing': {
                'enabled': True,
                'implementation': 'flax.linen.remat around actor and value MLPs',
            },
            'offline_replay_allocation': 'disabled because online_steps=0; immutable dataset sampling is unchanged',
        },
        run_dir / 'run_metadata.json',
    )

    # Authentication comes only from the job environment; no credential flag is
    # accepted or persisted.
    setup_wandb(
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_entity,
        group=FLAGS.run_group,
        name=FLAGS.run_name,
        mode=FLAGS.wandb_mode,
        run_id=FLAGS.wandb_id,
        resume='allow' if FLAGS.resume else None,
        output_dir=str(run_dir / 'wandb'),
    )
    wandb.define_metric('global_step')
    wandb.define_metric('*', step_metric='global_step')
    wandb_finished = False

    def process_train_dataset(ds):
        ds = Dataset.create(**ds)
        if FLAGS.sparse:
            sparse_rewards = (ds['rewards'] != 0.0) * -1.0
            ds_dict = {key: value for key, value in ds.items()}
            ds_dict['rewards'] = sparse_rewards
            ds = Dataset.create(**ds_dict)
        return ds

    dataset_idx = 0
    if dataset_paths:
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
        env, eval_env = wrap_envs(
            env,
            eval_env,
            frame_stack=FLAGS.frame_stack,
            agent_config=config,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
            FLAGS.env_name,
            frame_stack=FLAGS.frame_stack,
            agent_config=config,
            dataset_dir=FLAGS.ogbench_standard_dataset_dir,
        )

    train_dataset = process_train_dataset(train_dataset)
    val_dataset = process_train_dataset(val_dataset)

    # Upstream aliases a 100M ReplayBuffer as train_dataset but never mutates it
    # for online_steps=0. Keeping the immutable Dataset avoids a per-process 100M
    # copy while preserving offline sampling and update semantics.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    train_dataset = Dataset.create(**train_dataset)
    for dataset in (train_dataset, val_dataset):
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            dataset.config = config
    ex_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        ex_batch['observations'],
        ex_batch['actions'],
        config,
    )
    print('offline dataset size:', train_dataset.size, flush=True)
    print('gradient checkpointing: enabled (actor + value MLP remat)', flush=True)

    train_logger = CsvLogger(str(run_dir / 'train.csv'))
    eval_logger = CsvLogger(str(run_dir / 'eval.csv'))
    final_eval_logger = CsvLogger(str(run_dir / 'final_eval.csv'))

    session_start = time.time()
    last_log_time = session_start
    elapsed_before_session = 0.0
    global_step = 0
    pending_eval_step = None
    last_eval_step = None
    final_eval_done = False
    final_eval_metrics = None
    online_rng = jax.random.PRNGKey(FLAGS.seed)

    def configure_dataset(ds):
        ds = process_train_dataset(ds)
        ds.p_aug = FLAGS.p_aug
        ds.frame_stack = FLAGS.frame_stack
        ds.config = config
        return ds

    def load_shard(index: int):
        next_train_dataset, next_val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[index],
            compact_dataset=False,
            dataset_only=True,
            cur_env=env,
        )
        return configure_dataset(next_train_dataset), configure_dataset(next_val_dataset)

    def checkpoint_payload(reason: str) -> dict:
        jax.block_until_ready(agent.network.params)
        return make_checkpoint(
            {
                'saved_at_utc': datetime.now(timezone.utc).isoformat(),
                'reason': reason,
                'agent': jax.device_get(flax.serialization.to_state_dict(agent)),
                'global_step': global_step,
                'next_step': global_step + 1,
                'rng_state': capture_rng_state(),
                'online_rng': np.asarray(online_rng),
                'dataset_idx': dataset_idx,
                'pending_eval_step': pending_eval_step,
                'last_eval_step': last_eval_step,
                'final_eval_done': final_eval_done,
                'final_eval_metrics': final_eval_metrics,
                'elapsed_training_seconds': elapsed_before_session + (time.time() - session_start),
            },
            identity,
        )

    def save_checkpoint(reason: str) -> None:
        atomic_pickle_dump(checkpoint_payload(reason), checkpoint_path)
        print(f'Atomic checkpoint at step {global_step} ({reason}): {checkpoint_path}', flush=True)

    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint
        agent = flax.serialization.from_state_dict(agent, checkpoint['agent'])
        global_step = int(checkpoint['global_step'])
        if not 0 <= global_step <= FLAGS.offline_steps:
            raise ValueError(f'Invalid restored global_step: {global_step}')
        if int(checkpoint.get('next_step', -1)) != global_step + 1:
            raise ValueError('checkpoint next_step is inconsistent with global_step')
        dataset_idx = int(checkpoint['dataset_idx'])
        pending_eval_step = checkpoint.get('pending_eval_step')
        last_eval_step = checkpoint.get('last_eval_step')
        final_eval_done = bool(checkpoint.get('final_eval_done', False))
        final_eval_metrics = checkpoint.get('final_eval_metrics')
        elapsed_before_session = float(checkpoint.get('elapsed_training_seconds', 0.0))
        online_rng = jax.numpy.asarray(checkpoint.get('online_rng', online_rng), dtype=np.uint32)
        if pending_eval_step is not None and int(pending_eval_step) != global_step:
            raise ValueError('pending_eval_step must equal restored global_step')
        if last_eval_step is not None and not 1 <= int(last_eval_step) <= global_step:
            raise ValueError('last_eval_step is outside restored training range')
        if final_eval_done and (
            global_step != FLAGS.offline_steps
            or last_eval_step != FLAGS.offline_steps
            or pending_eval_step is not None
            or final_eval_metrics is None
        ):
            raise ValueError('inconsistent final-evaluation checkpoint state')
        if dataset_paths:
            if not 0 <= dataset_idx < len(dataset_paths):
                raise ValueError(f'Invalid restored dataset_idx: {dataset_idx}')
            if dataset_idx != 0:
                train_dataset, val_dataset = load_shard(dataset_idx)
        elif dataset_idx != 0:
            raise ValueError('standard dataset checkpoint has a nonzero shard index')
        # Reconstruction may consume RNG, so restore it last.
        restore_rng_state(checkpoint['rng_state'])
        print(f'Restored completed step {global_step}; next update is {global_step + 1}', flush=True)

    def log_wandb(metrics: dict, step: int) -> None:
        # Internal W&B history stays monotonic across resume. global_step is the
        # scientific x-axis and permits train/eval records at the same update.
        wandb.log({'global_step': step, **metrics})

    def run_evaluation(step: int) -> None:
        nonlocal pending_eval_step, last_eval_step, final_eval_done, final_eval_metrics
        periodic, final = evaluation_due(step, FLAGS.offline_steps, FLAGS.eval_interval)
        if not (periodic or final):
            raise ValueError(f'Unexpected pending evaluation at step {step}')
        pending_eval_step = step
        save_checkpoint('evaluation-pending')
        stop.raise_if_requested()
        episodes = FLAGS.final_eval_episodes if final else FLAGS.eval_episodes
        eval_info, _trajs, renders = evaluate(
            agent=agent,
            env=eval_env,
            env_name=FLAGS.env_name,
            config=config,
            num_eval_episodes=episodes,
            num_video_episodes=FLAGS.video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
            training_seed=FLAGS.seed,
            eval_step=step,
            stop_callback=stop.raise_if_requested,
        )
        stop.raise_if_requested()
        raw_metrics = to_jsonable(eval_info)
        wandb_metrics = {}
        if periodic:
            periodic_metrics = {f'evaluation/{key}': value for key, value in eval_info.items()}
            eval_logger.log(periodic_metrics, step=step)
            wandb_metrics.update(periodic_metrics)
        if final:
            final_metrics = {f'final_evaluation/{key}': value for key, value in eval_info.items()}
            final_eval_logger.log(final_metrics, step=step)
            wandb_metrics.update(final_metrics)
        if FLAGS.video_episodes > 0:
            video = get_wandb_video(renders=renders)
            if periodic:
                wandb_metrics['evaluation/video'] = video
            if final:
                wandb_metrics['final_evaluation/video'] = video
        # One payload with distinct namespaces handles coincident periodic/final
        # boundaries without duplicate W&B step commits.
        log_wandb(wandb_metrics, step)
        for key, value in raw_metrics.items():
            print(f'{key}: {value}', flush=True)
        last_eval_step = step
        pending_eval_step = None
        if final:
            final_eval_done = True
            final_eval_metrics = raw_metrics
        save_checkpoint('evaluation-complete')

    def write_completion() -> None:
        if not final_eval_done or final_eval_metrics is None or last_eval_step != FLAGS.offline_steps:
            raise RuntimeError('refusing to mark completion before the full final evaluation')
        completion = {
            'schema_version': COMPLETION_SCHEMA_VERSION,
            'status': 'complete',
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'identity_sha256': identity_hash,
            'identity': identity,
            'protocol_sha256': FLAGS.protocol_sha256,
            'code_manifest_sha256': code_manifest_sha256,
            'runtime_software_sha256': runtime_software_sha256,
            'run_name': FLAGS.run_name,
            'wandb_id': FLAGS.wandb_id,
            'env_name': FLAGS.env_name,
            'seed': FLAGS.seed,
            'global_step': global_step,
            'final_eval_step': last_eval_step,
            'final_eval_episodes': FLAGS.final_eval_episodes,
            'final_evaluation': final_eval_metrics,
            'checkpoint': checkpoint_path.name,
            'upstream_commit': UPSTREAM_COMMIT,
            'gradient_checkpointing': True,
        }
        atomic_json_dump(completion, completion_path)
        print(f'Run complete: {completion_path}', flush=True)

    try:
        # Finish an interrupted absolute-step evaluation before taking step + 1.
        if pending_eval_step is not None:
            run_evaluation(int(pending_eval_step))

        if global_step == FLAGS.offline_steps:
            if not final_eval_done:
                raise RuntimeError('terminal checkpoint is missing mandatory final evaluation')
            save_checkpoint('final')
            wandb.finish(exit_code=0)
            wandb_finished = True
            write_completion()
            return 0

        progress = tqdm.tqdm(
            range(global_step + 1, FLAGS.offline_steps + 1),
            initial=global_step,
            total=FLAGS.offline_steps,
            smoothing=0.1,
            dynamic_ncols=True,
        )
        for step in progress:
            stop.raise_if_requested()
            if dataset_paths:
                desired_idx = shard_index_for_step(
                    step,
                    FLAGS.dataset_replace_interval,
                    len(dataset_paths),
                )
                if desired_idx != dataset_idx:
                    pre_load_rng = capture_rng_state()
                    try:
                        train_dataset, val_dataset = load_shard(desired_idx)
                    except BaseException:
                        restore_rng_state(pre_load_rng)
                        raise
                    dataset_idx = desired_idx
                    print(f'Using shard {dataset_idx}: {dataset_paths[dataset_idx]}', flush=True)
                stop.raise_if_requested()

            # Roll sampling RNG back if an exception prevents committing this update.
            pre_update_rng = capture_rng_state()
            try:
                batch = train_dataset.sample(config['batch_size'])
                next_agent, update_info = agent.update(batch)
            except BaseException:
                restore_rng_state(pre_update_rng)
                raise
            agent = next_agent
            global_step = step

            periodic_eval, final_eval = evaluation_due(step, FLAGS.offline_steps, FLAGS.eval_interval)
            if periodic_eval or final_eval:
                # A signal during the update cannot make resume skip this boundary.
                pending_eval_step = step

            if step % FLAGS.log_interval == 0:
                now = time.time()
                train_metrics = {f'training/{key}': value for key, value in update_info.items()}
                train_metrics['time/epoch_time'] = (now - last_log_time) / FLAGS.log_interval
                train_metrics['time/total_time'] = elapsed_before_session + (now - session_start)
                last_log_time = now
                train_logger.log(train_metrics, step=step)
                log_wandb(train_metrics, step)

            stop.raise_if_requested()
            if pending_eval_step is not None:
                run_evaluation(step)

            if step % checkpoint_interval == 0:
                save_checkpoint('periodic')

        if global_step != FLAGS.offline_steps or not final_eval_done:
            raise RuntimeError('training ended without mandatory final evaluation')
        save_checkpoint('final')
        wandb.finish(exit_code=0)
        wandb_finished = True
        write_completion()
        return 0
    except GracefulStop as exc:
        save_checkpoint(f'graceful-stop:{exc.reason}')
        if not wandb_finished:
            # Tell W&B this stable-ID run is expected to resume, but only after
            # the durable local checkpoint is known to exist.
            try:
                wandb.mark_preempting()
            except BaseException as wandb_exc:
                print(f'W&B preemption mark failed: {wandb_exc!r}', file=sys.stderr, flush=True)
            try:
                wandb.finish(exit_code=GRACEFUL_EXIT_CODE)
            except BaseException as wandb_exc:
                print(f'W&B shutdown failed: {wandb_exc!r}', file=sys.stderr, flush=True)
            wandb_finished = True
        print(f'Graceful stop ({exc.reason}); checkpointed step {global_step}', file=sys.stderr, flush=True)
        return GRACEFUL_EXIT_CODE
    except BaseException as exc:
        # Never mask the primary failure with an emergency-save/shutdown failure.
        try:
            save_checkpoint(f'exception:{type(exc).__name__}')
        except BaseException as checkpoint_exc:
            print(f'Emergency checkpoint failed: {checkpoint_exc!r}', file=sys.stderr, flush=True)
        if not wandb_finished:
            try:
                wandb.finish(exit_code=1)
            except BaseException as wandb_exc:
                print(f'W&B shutdown failed: {wandb_exc!r}', file=sys.stderr, flush=True)
        raise
    finally:
        train_logger.close()
        eval_logger.close()
        final_eval_logger.close()


if __name__ == '__main__':
    app.run(_main)
