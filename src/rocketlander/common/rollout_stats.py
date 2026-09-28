"""Success rate and task return over the most recent finished training episodes."""

from __future__ import annotations

from collections import deque

import numpy as np


class RolloutStats:
    def __init__(self, num_envs: int, window: int = 100) -> None:
        self.task_returns = np.zeros(num_envs)
        self.finished: deque[tuple[bool, float]] = deque(maxlen=window)

    def add(self, infos: list[dict], dones: np.ndarray) -> None:
        self.task_returns += [info["task_reward"] for info in infos]
        for i in np.flatnonzero(dones):
            self.finished.append((infos[i]["outcome"] == "landed", self.task_returns[i]))
            self.task_returns[i] = 0.0

    def summary(self) -> dict[str, float]:
        if not self.finished:
            return {}
        landed, returns = zip(*self.finished, strict=True)
        return {
            "train/success_rate": float(np.mean(landed)),
            "train/task_return": float(np.mean(returns)),
        }
