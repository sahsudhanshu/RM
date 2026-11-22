"""Rule-based route policy (nearest feasible task; land near the remaining work).

Used as (a) the imitation teacher that warm-starts the neural policy and (b) a
non-learning reference. It is NOT the base paper's DRL policy."""

from __future__ import annotations

import numpy as np

from .env import MissionEnv, MissionResult


def nearest_feasible(env: MissionEnv, mask: np.ndarray) -> int:
    u, K = env._cur_uav, env.K
    c = env.cur[u]
    visit = mask[:K]
    if visit.any():
        cand = np.flatnonzero(visit)
        return int(cand[np.argmin(env.task_dist[c, cand])])
    rech = np.flatnonzero(mask[K:])
    remaining = np.flatnonzero(~env.visited)
    if remaining.size:
        score = env.task_dist[np.ix_(rech, remaining)].min(axis=1) + 0.5 * env.task_dist[c, rech]
    else:
        score = env.task_dist[c, rech]
    return int(K + rech[np.argmin(score)])


def run_policy(env: MissionEnv, policy) -> MissionResult:
    """Drive `env` to completion with `policy(env, mask) -> action`."""
    while env.active_uav() is not None:
        mask = env.mask()
        if env.done:
            break
        env.step(int(policy(env, mask)))
    return env.result()
