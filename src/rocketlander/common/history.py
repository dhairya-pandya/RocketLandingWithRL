"""A short memory for memoryless networks: the last k observations and the actions before them.

The policy cannot observe mass, thrust, engine lag or wind, but it can infer them from how the
rocket responded to its own recent actions. Each entry is (obs_t, action_{t-1}), oldest first; a
fresh episode starts with every entry equal to its first observation and zero actions.
"""

from __future__ import annotations

import numpy as np

from rocketlander.common.vec_env import VecEnv, VecStep
from rocketlander.envs.rocket_env import ACT_SIZE, OBS_SIZE

ENTRY_SIZE = OBS_SIZE + ACT_SIZE


def history_size(length: int) -> int:
    """Input size of a network that sees `length` entries (1 = the plain observation)."""
    return OBS_SIZE if length == 1 else length * ENTRY_SIZE


class History:
    def __init__(self, length: int, n: int) -> None:
        self.entries = np.zeros((n, length, ENTRY_SIZE), dtype=np.float32)

    def reset(self, i: int, obs: np.ndarray) -> None:
        self.entries[i] = 0.0
        self.entries[i, :, :OBS_SIZE] = obs

    def push(self, obs: np.ndarray, prev_actions: np.ndarray) -> None:
        self.entries = np.roll(self.entries, -1, axis=1)
        self.entries[:, -1] = np.concatenate([obs, prev_actions], axis=-1)

    def stacked(self) -> np.ndarray:
        return self.entries.reshape(len(self.entries), -1).copy()


class HistoryVecEnv:
    """VecEnv whose observations are stacked histories; the same interface as VecEnv."""

    def __init__(self, envs: VecEnv, length: int) -> None:
        self.envs, self.history = envs, History(length, envs.num_envs)

    @property
    def num_envs(self) -> int:
        return self.envs.num_envs

    def reset(self) -> np.ndarray:
        for i, obs in enumerate(self.envs.reset()):
            self.history.reset(i, obs)
        return self.history.stacked()

    def step(self, actions: np.ndarray) -> VecStep:
        step = self.envs.step(actions)
        self.history.push(step.final_obs, actions)  # final_obs is step.obs where nothing ended
        final = self.history.stacked()
        for i in np.flatnonzero(step.terminated | step.truncated):
            self.history.reset(i, step.obs[i])
        return VecStep(
            self.history.stacked(), step.rewards, step.terminated, step.truncated, final, step.infos
        )
