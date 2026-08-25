"""CPU-only integration checks for real RQL remat/checkpoint semantics."""

import random

import numpy as np
import pytest


jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
flax = pytest.importorskip('flax')
pytest.importorskip('distrax')
pytest.importorskip('einops')
pytest.importorskip('ml_collections')
pytest.importorskip('optax')

from agents.rql import RQLAgent, get_config
from utils.resume import (
    atomic_pickle_dump,
    capture_rng_state,
    load_checkpoint,
    make_checkpoint,
    restore_rng_state,
)


def tiny_config(*, gradient_checkpointing):
    config = get_config()
    config.h = 1
    config.batch_size = 4
    config.ensemble_ct = 2
    config.actor_hidden_dims = (8, 8)
    config.value_hidden_dims = (8, 8)
    config.flow_steps = 2
    config.gradient_checkpointing = gradient_checkpointing
    return config


def create_agent(*, gradient_checkpointing, seed=7):
    return RQLAgent.create(
        seed,
        jnp.zeros((1, 3), dtype=jnp.float32),
        jnp.zeros((1, 2), dtype=jnp.float32),
        tiny_config(gradient_checkpointing=gradient_checkpointing),
    )


def batch_source():
    # RQL consumes h+1 trajectory positions along axis 0.
    observations = np.arange(2 * 32 * 3, dtype=np.float32).reshape(2, 32, 3) / 100
    actions = np.arange(2 * 32 * 2, dtype=np.float32).reshape(2, 32, 2) / 100
    rewards = np.arange(2 * 32, dtype=np.float32).reshape(2, 32) / 100
    terminals = np.zeros((2, 32), dtype=np.float32)
    masks = np.ones((2, 32), dtype=np.float32)
    return observations, actions, rewards, terminals, masks


def sample_batch(source):
    indices = np.random.randint(0, source[0].shape[1], size=4)
    reward_scale = 0.5 + random.random()
    observations, actions, rewards, terminals, masks = source
    return {
        'observations': observations[:, indices],
        'actions': actions[:, indices],
        'rewards': rewards[:, indices] * reward_scale,
        'terminals': terminals[:, indices],
        'masks': masks[:, indices],
    }


def assert_trees_close(left, right, *, rtol=1e-6, atol=1e-6):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_allclose(np.asarray(left_leaf), np.asarray(right_leaf), rtol=rtol, atol=atol)


def test_real_agent_checkpoint_resume_matches_uninterrupted_second_update(tmp_path):
    source = batch_source()
    np.random.seed(101)
    random.seed(202)
    initial_agent = create_agent(gradient_checkpointing=True)
    first_batch = sample_batch(source)
    once_updated, _ = initial_agent.update(first_batch)
    jax.block_until_ready(once_updated.network.params)

    identity = {'kind': 'real-rql-resume-test', 'gradient_checkpointing': True}
    checkpoint = make_checkpoint(
        {
            'agent': jax.device_get(flax.serialization.to_state_dict(once_updated)),
            'rng_state': capture_rng_state(),
            'global_step': 1,
            'next_step': 2,
            'dataset_idx': 0,
            'pending_eval_step': None,
        },
        identity,
    )
    checkpoint_path = tmp_path / 'checkpoint.pkl'
    atomic_pickle_dump(checkpoint, checkpoint_path)

    uninterrupted_batch = sample_batch(source)
    uninterrupted_agent, uninterrupted_info = once_updated.update(uninterrupted_batch)
    jax.block_until_ready(uninterrupted_agent.network.params)

    restored_payload = load_checkpoint(checkpoint_path, identity)
    restored_template = create_agent(gradient_checkpointing=True)
    restored_agent = flax.serialization.from_state_dict(restored_template, restored_payload['agent'])
    restore_rng_state(restored_payload['rng_state'])
    resumed_batch = sample_batch(source)
    for key in uninterrupted_batch:
        np.testing.assert_array_equal(uninterrupted_batch[key], resumed_batch[key])
    resumed_agent, resumed_info = restored_agent.update(resumed_batch)
    jax.block_until_ready(resumed_agent.network.params)

    assert_trees_close(uninterrupted_agent.network.params, resumed_agent.network.params)
    assert_trees_close(uninterrupted_agent.network.opt_state, resumed_agent.network.opt_state)
    assert_trees_close(uninterrupted_agent.rng, resumed_agent.rng)
    assert_trees_close(uninterrupted_info, resumed_info)


def test_remat_on_and_off_have_equivalent_update_semantics():
    source = batch_source()
    np.random.seed(303)
    random.seed(404)
    batch = sample_batch(source)
    remat_agent = create_agent(gradient_checkpointing=True)
    plain_agent = create_agent(gradient_checkpointing=False)

    remat_updated, remat_info = remat_agent.update(batch)
    plain_updated, plain_info = plain_agent.update(batch)
    jax.block_until_ready((remat_updated.network.params, plain_updated.network.params))

    assert_trees_close(remat_updated.network.params, plain_updated.network.params)
    assert_trees_close(remat_updated.network.opt_state, plain_updated.network.opt_state)
    assert_trees_close(remat_info, plain_info)
