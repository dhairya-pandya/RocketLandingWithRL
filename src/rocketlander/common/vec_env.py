"""Minimal synchronous vector env: steps N copies and resets finished ones immediately."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from rocketlander.envs.rocket_env import RocketLanderEnv


@dataclass
class VecStep:
    obs: np.ndarray  # (n, obs_dim): next observation; already reset where an episode ended
    rewards: np.ndarray  # (n,)
    terminated: np.ndarray  # (n,) bool
    truncated: np.ndarray  # (n,) bool
    final_obs: np.ndarray  # (n, obs_dim): last observation of an episode that just ended
    infos: list[dict[str, Any]]


class VecEnv:
    def __init__(self, make_env: Callable[[], RocketLanderEnv], n: int, seed: int) -> None:
        self.envs = [make_env() for _ in range(n)]
        self.seed = seed

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    def reset(self) -> np.ndarray:
        return np.stack([env.reset(seed=self.seed + i)[0] for i, env in enumerate(self.envs)])

    def step(self, actions: np.ndarray) -> VecStep:
        results = [env.step(a) for env, a in zip(self.envs, actions, strict=True)]
        obs = np.stack([r[0] for r in results])
        final_obs = obs.copy()
        terminated = np.array([r[2] for r in results])
        truncated = np.array([r[3] for r in results])
        for i in np.flatnonzero(terminated | truncated):
            obs[i] = self.envs[i].reset()[0]  # RNG continues, so each episode differs
        return VecStep(
            obs=obs,
            rewards=np.array([r[1] for r in results], dtype=np.float64),
            terminated=terminated,
            truncated=truncated,
            final_obs=final_obs,
            infos=[r[4] for r in results],
        )
