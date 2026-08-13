"""Execute a whole generated tree in the simulator to get *actual* node endpoints.

Every node's root-to-node action chunks are replayed, so we learn where the agent really
lands versus where the model predicted. Implemented as a DFS with MuJoCo state
save/restore: each node's chunk is executed exactly once (shared prefixes are not
re-executed), so the cost is the sum of chunk lengths over the tree -- roughly 1k
simulator steps for a 64-node tree, which is negligible next to the model forward pass.

This is a *diagnostic* only. Nothing here may be used by a policy: it reads privileged
simulator state that a deployed planner does not have.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def save_state(env) -> tuple[np.ndarray, np.ndarray]:
    u = env.unwrapped
    return u.data.qpos.copy(), u.data.qvel.copy()


def restore_state(env, state: tuple[np.ndarray, np.ndarray]) -> None:
    env.unwrapped.set_state(state[0], state[1])


def current_xy(env) -> np.ndarray:
    return np.asarray(env.unwrapped.get_ob(), dtype=np.float32)[:2]


def ground_tree(env, tree, normalizer, index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Replay every node's path from the current simulator state.

    Returns ``(endpoints [N, 2], grounded [N] bool)``. The simulator is restored to its
    entry state before returning, so the caller's episode is unaffected.
    """
    u = env.unwrapped
    entry = save_state(env)

    parent = tree.parent_index[index].cpu().numpy()
    valid = tree.valid[index].cpu().numpy()
    chunk = tree.action_chunk[index].float().cpu().numpy()
    mask = tree.action_mask[index].float().cpu().numpy()
    n = len(valid)

    children: dict[int, list[int]] = defaultdict(list)
    for i in range(1, n):
        if valid[i] and parent[i] >= 0:
            children[int(parent[i])].append(i)

    endpoints = np.full((n, 2), np.nan, dtype=np.float32)
    grounded = np.zeros(n, dtype=bool)
    endpoints[0] = current_xy(env)
    grounded[0] = True

    states: dict[int, tuple[np.ndarray, np.ndarray]] = {0: entry}
    stack = [0]
    while stack:
        node = stack.pop()
        state = states.pop(node, None)
        if state is None:
            continue
        for child in children.get(node, []):
            restore_state(env, state)
            steps = int(mask[child].sum())
            if steps <= 0:
                continue
            actions = normalizer.denorm_act(chunk[child][:steps])
            for action in actions:
                env.step(np.clip(action, -1.0, 1.0))
            endpoints[child] = current_xy(env)
            grounded[child] = True
            states[child] = save_state(env)
            stack.append(child)

    restore_state(env, entry)
    return endpoints, grounded


def predicted_xy(model, tree, normalizer, index: int = 0) -> np.ndarray:
    """Decoded model-predicted positions for every node. ``[N, 2]``."""
    import torch

    with torch.no_grad():
        decoded = model.decoder(tree.latent[index]).float().cpu().numpy()
    return normalizer.denorm_obs(decoded)[:, :2]


def selection_disagreement(
    pred_xy: np.ndarray,
    actual_xy: np.ndarray,
    grounded: np.ndarray,
    valid: np.ndarray,
    goal_xy: np.ndarray,
) -> dict:
    """Compare leaf ranking under predicted vs actual endpoints.

    ``disagree`` is the quantity that decides whether long-horizon endpoint error is the
    bottleneck: if the planner's best-by-prediction node is usually not the best-by-reality
    node, the tree may contain a good future that selection cannot find.
    """
    usable = valid.copy()
    usable[0] = False
    idx = np.flatnonzero(usable & grounded)
    if idx.size == 0:
        return {}

    d_pred = np.linalg.norm(pred_xy[idx] - goal_xy[None, :2], axis=1)
    d_act = np.linalg.norm(actual_xy[idx] - goal_xy[None, :2], axis=1)
    best_pred = int(idx[int(np.argmin(d_pred))])
    best_act = int(idx[int(np.argmin(d_act))])

    # How good is the node the planner would pick, measured by where it really lands?
    realised_of_pred = float(
        np.linalg.norm(actual_xy[best_pred] - goal_xy[:2])
    )
    return {
        "best_pred_node": best_pred,
        "best_act_node": best_act,
        "disagree": float(best_pred != best_act),
        "min_predicted_goal_distance": float(d_pred.min()),
        "min_actual_goal_distance": float(d_act.min()),
        "realised_goal_distance_of_predicted_choice": realised_of_pred,
        "regret": realised_of_pred - float(d_act.min()),
        "num_grounded": int(idx.size),
    }
