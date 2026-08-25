from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
import cv2

from utils.resume import evaluation_episode_seeds

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    env_name,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    training_seed=None,
    eval_step=None,
    stop_callback=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        env_name: Environment name.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        training_seed: Training seed used with ``eval_step`` for deterministic episodes.
        eval_step: Absolute training step used with ``training_seed`` for deterministic episodes.
        stop_callback: Optional callback checked before every episode and environment step.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    if (training_seed is None) != (eval_step is None):
        raise ValueError('training_seed and eval_step must be provided together')
    actor_fn = None
    if training_seed is None:
        actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        if stop_callback is not None:
            stop_callback()
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        if training_seed is not None:
            env_seed, actor_seed = evaluation_episode_seeds(training_seed, eval_step, i)
            actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(actor_seed))
            observation, info = env.reset(seed=env_seed)
        else:
            observation, info = env.reset()
        goal_frame = info.get('goal_rendered')

        done = False
        step = 0
        render = []
        while not done:
            if stop_callback is not None:
                stop_callback()
            action = actor_fn(obs=observation, temperature=eval_temperature) # goal=info.get("goal")
            action = np.array(action)
            if not np.issubdtype(action.dtype, np.integer):
                action = np.clip(action, -1, 1)
            next_observation, reward, terminated, truncated, info = env.step(action)
            if stop_callback is not None:
                stop_callback()
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                # Get a rendered frame from the environment.
                if env_name == 'fmb':
                    frame = env.unwrapped._get_obs_with_sensor_data({})['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()
                else:
                    frame = env.render().copy()
                if goal_frame is not None:
                    frame = np.concatenate([goal_frame, frame], axis=0)
                render.append(frame)
                # cv2.imshow("win", frame)
                # cv2.waitKey(5000)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
