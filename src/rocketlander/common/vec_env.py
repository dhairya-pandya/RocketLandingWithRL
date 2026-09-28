"""Minimal synchronous vector env: steps N copies and resets finished ones immediately."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from rocketlander.common.curriculum import LevelSchedule
from rocketlander.envs.rocket_env import RocketLanderEnv
from rocketlander.evaluation import EVAL_SEEDS


@dataclass
class VecStep:
    obs: np.ndarray  # (n, obs_dim): next observation; already reset where an episode ended
    rewards: np.ndarray  # (n,)
    terminated: np.ndarray  # (n,) bool
    truncated: np.ndarray  # (n,) bool
    final_obs: np.ndarray  # (n, obs_dim): last observation of an episode that just ended
    infos: list[dict[str, Any]]


class VecEnv:
    def __init__(
        self,
        make_env: Callable[[], RocketLanderEnv],
        n: int,
        seed: int,
        schedule: LevelSchedule | None = None,
    ) -> None:
        if seed + n > min(EVAL_SEEDS):
            raise ValueError(
                f"Training seeds {seed}..{seed + n - 1} would reach the evaluation seeds"
            )
        self.envs = [make_env() for _ in range(n)]
        self.seed = seed
        self.schedule = schedule
        # Off while a demonstrator acts: the schedule should track the learner.
        self.record_outcomes = True

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    def _options(self) -> dict | None:
        return {"level": self.schedule.sample()} if self.schedule else None

    def reset(self) -> np.ndarray:
        return np.stack(
            [
                env.reset(seed=self.seed + i, options=self._options())[0]
                for i, env in enumerate(self.envs)
            ]
        )

    def step(self, actions: np.ndarray) -> VecStep:
        results = [env.step(a) for env, a in zip(self.envs, actions, strict=True)]
        obs = np.stack([r[0] for r in results])
        final_obs = obs.copy()
        terminated = np.array([r[2] for r in results])
        truncated = np.array([r[3] for r in results])
        for i in np.flatnonzero(terminated | truncated):
            if self.schedule and self.record_outcomes:
                info = results[i][4]
                self.schedule.record(info["level"], info["outcome"] == "landed")
            obs[i] = self.envs[i].reset(options=self._options())[0]  # RNG continues
        return VecStep(
            obs=obs,
            rewards=np.array([r[1] for r in results], dtype=np.float64),
            terminated=terminated,
            truncated=truncated,
            final_obs=final_obs,
            infos=[r[4] for r in results],
        )
