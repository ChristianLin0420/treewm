import random

import numpy as np
import pytest

from utils.resume import (
    GracefulStop,
    StopController,
    atomic_json_dump,
    atomic_pickle_dump,
    capture_rng_state,
    collect_runtime_provenance,
    discover_official_100m_shards,
    evaluation_due,
    evaluation_episode_seeds,
    load_checkpoint,
    make_checkpoint,
    restore_rng_state,
    shard_index_for_step,
    stable_json_hash,
    trainer_code_fingerprint,
)


def test_atomic_checkpoint_round_trip_and_identity_guard(tmp_path):
    identity = {'run': 'task-seed-0', 'protocol_sha256': 'a' * 64}
    payload = {
        'agent': {'weights': np.arange(4)},
        'global_step': 17,
        'next_step': 18,
        'dataset_idx': 3,
        'pending_eval_step': None,
    }
    checkpoint = make_checkpoint(payload, identity)
    destination = tmp_path / 'checkpoint.pkl'
    atomic_pickle_dump(checkpoint, destination)

    restored = load_checkpoint(destination, identity)
    assert restored['global_step'] == 17
    assert restored['next_step'] == 18
    np.testing.assert_array_equal(restored['agent']['weights'], np.arange(4))
    assert not list(tmp_path.glob('.checkpoint.pkl.*.tmp'))

    with pytest.raises(ValueError, match='identity'):
        load_checkpoint(destination, {**identity, 'protocol_sha256': 'b' * 64})


def test_global_numpy_and_python_rng_round_trip():
    np.random.seed(123)
    random.seed(456)
    state = capture_rng_state()
    expected_numpy = np.random.randint(0, 1_000_000, size=8)
    expected_python = [random.random() for _ in range(8)]

    np.random.seed(999)
    random.seed(999)
    restore_rng_state(state)
    np.testing.assert_array_equal(np.random.randint(0, 1_000_000, size=8), expected_numpy)
    assert [random.random() for _ in range(8)] == expected_python


def test_stop_controller_defers_signal_and_walltime():
    now = [100.0]
    controller = StopController(5.0, clock=lambda: now[0])
    controller.raise_if_requested()
    now[0] = 105.0
    with pytest.raises(GracefulStop, match='walltime') as stopped:
        controller.raise_if_requested()
    assert stopped.value.reason == 'walltime'

    requested = StopController(0.0)
    requested.request('SIGUSR1')
    requested.request('SIGTERM')
    with pytest.raises(GracefulStop) as stopped:
        requested.raise_if_requested()
    assert stopped.value.reason == 'SIGUSR1'


@pytest.mark.parametrize(
    ('step', 'total', 'interval', 'expected'),
    [
        (1, 1_000_000, 100_000, (True, False)),
        (99_999, 1_000_000, 100_000, (False, False)),
        (100_000, 1_000_000, 100_000, (True, False)),
        (1_000_000, 1_000_000, 0, (False, True)),
        (1_000_000, 1_000_000, 100_000, (True, True)),
    ],
)
def test_evaluation_boundaries_include_unconditional_final(step, total, interval, expected):
    assert evaluation_due(step, total, interval) == expected


def test_absolute_shard_rotation_is_idempotent_at_resume_boundary():
    assert shard_index_for_step(1, 1000, 100) == 0
    assert shard_index_for_step(999, 1000, 100) == 0
    assert shard_index_for_step(1000, 1000, 100) == 1
    assert shard_index_for_step(1001, 1000, 100) == 1
    assert shard_index_for_step(100_000, 1000, 100) == 0
    # Recomputing an interrupted absolute step cannot rotate twice.
    assert shard_index_for_step(17_000, 1000, 100) == shard_index_for_step(17_000, 1000, 100)


def test_one_million_terminal_resume_contract(tmp_path):
    identity = {
        'offline_steps': 1_000_000,
        'final_eval_episodes': 50,
        'protocol_sha256': 'a' * 64,
    }
    checkpoint_path = tmp_path / 'checkpoint.pkl'
    atomic_pickle_dump(
        make_checkpoint(
            {
                'global_step': 999_999,
                'next_step': 1_000_000,
                'pending_eval_step': None,
                'last_eval_step': 900_000,
                'final_eval_done': False,
            },
            identity,
        ),
        checkpoint_path,
    )

    restored = load_checkpoint(checkpoint_path, identity)
    assert restored['global_step'] == 999_999
    assert restored['next_step'] == 1_000_000
    assert restored['final_eval_done'] is False
    assert evaluation_due(restored['next_step'], identity['offline_steps'], 100_000) == (True, True)


def test_evaluation_episode_seeds_are_deterministic_and_keyed_by_step_episode():
    seed = evaluation_episode_seeds(4, 1_000_000, 49)
    assert seed == evaluation_episode_seeds(4, 1_000_000, 49)
    assert seed != evaluation_episode_seeds(4, 1_000_000, 48)
    assert seed != evaluation_episode_seeds(4, 900_000, 49)


def test_atomic_json_and_stable_hash(tmp_path):
    destination = tmp_path / 'COMPLETED.json'
    atomic_json_dump({'metric': np.float32(0.75), 'step': 1_000_000}, destination)
    assert destination.read_text().endswith('\n')
    assert stable_json_hash({'a': 1, 'b': 2}) == stable_json_hash({'b': 2, 'a': 1})


def test_runtime_provenance_is_dependency_light_and_complete():
    provenance = collect_runtime_provenance()
    assert set(provenance) == {'python', 'packages', 'platform'}
    assert provenance['python']['version']
    assert provenance['python']['implementation']
    assert set(provenance['packages']) == {
        'jax', 'jaxlib', 'flax', 'optax', 'distrax', 'einops', 'ml_collections',
        'gymnasium', 'ogbench', 'wandb', 'numpy', 'mujoco'
    }
    assert provenance['packages']['numpy'] is not None
    assert provenance['platform']['system']
    assert provenance['platform']['machine']


def test_dependency_light_trainer_code_fingerprint_covers_runtime_sources():
    upstream_dir = __import__('pathlib').Path(__file__).resolve().parents[1]
    fingerprint = trainer_code_fingerprint(upstream_dir)
    assert fingerprint['manifest_sha256'] == stable_json_hash(fingerprint['files'])
    assert {
        'main.py',
        'agents/rql.py',
        'utils/resume.py',
        'utils/networks.py',
        'envs/env_utils.py',
    }.issubset(fingerprint['files'])


def make_official_100m_directory(directory, stem='puzzle-4x4-play-v0'):
    directory.mkdir()
    for index in range(100):
        (directory / f'{stem}-{index:03d}.npz').touch()
        (directory / f'{stem}-{index:03d}-val.npz').touch()


def test_official_100m_shards_require_exact_contiguous_train_and_validation_files(tmp_path):
    directory = tmp_path / '100m'
    make_official_100m_directory(directory)
    paths = discover_official_100m_shards(
        directory,
        'puzzle-4x4-play-singletask-task3-v0',
    )
    assert len(paths) == 100
    assert paths[0].endswith('puzzle-4x4-play-v0-000.npz')
    assert paths[-1].endswith('puzzle-4x4-play-v0-099.npz')


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        (lambda path: (path / 'puzzle-4x4-play-v0-042.npz').unlink(), 'missing 1'),
        (lambda path: (path / 'puzzle-4x4-play-v0-042-val.npz').unlink(), 'missing 1'),
        (lambda path: (path / 'puzzle-4x4-play-v0-100.npz').touch(), 'unexpected 1'),
        (lambda path: (path / 'puzzle-4x4-play-v0-0042.npz').touch(), 'unexpected 1'),
        (lambda path: (path / 'unrelated.npz').touch(), 'unexpected 1'),
        (lambda path: (path / 'unrelated.NPZ').touch(), 'unexpected 1'),
    ],
)
def test_official_100m_shards_reject_missing_extra_and_duplicate_style_entries(
    tmp_path,
    mutation,
    message,
):
    directory = tmp_path / '100m'
    make_official_100m_directory(directory)
    mutation(directory)
    with pytest.raises(ValueError, match=message):
        discover_official_100m_shards(directory, 'puzzle-4x4-play-singletask-v0')


def test_official_100m_shards_reject_wrong_environment_stem(tmp_path):
    directory = tmp_path / '100m'
    make_official_100m_directory(directory)
    with pytest.raises(ValueError, match='missing 200'):
        discover_official_100m_shards(
            directory,
            'cube-quadruple-play-singletask-task1-v0',
        )
