"""Running mean/std statistics for observation and reward normalization."""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Parallel (Chan et al.) update of mean and variance from batches."""

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, batch: np.ndarray) -> None:
        batch_mean, batch_var, n = batch.mean(axis=0), batch.var(axis=0), batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        m2 = self.var * self.count + batch_var * n + delta**2 * self.count * n / total
        self.var = m2 / total
        self.count = total

    def soften(self, max_count: float) -> None:
        """Cap the sample count so new data (e.g. a new level when fine-tuning) moves the stats."""
        self.count = min(self.count, max_count)

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)

    def state_dict(self) -> dict:
        """Plain lists and floats, so checkpoints load with torch.load(weights_only=True)."""
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": float(self.count)}

    def load_state_dict(self, state: dict) -> None:
        self.mean, self.var = np.array(state["mean"]), np.array(state["var"])
        self.count = state["count"]


class RewardScaler:
    """Divides rewards by the std of the running discounted return (keeps value targets ~O(1))."""

    def __init__(self, num_envs: int, gamma: float) -> None:
        self.gamma = gamma
        self.returns = np.zeros(num_envs)
        self.rms = RunningMeanStd()

    def scale(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        self.returns = self.returns * self.gamma + rewards
        self.rms.update(self.returns)
        self.returns[dones] = 0.0
        return rewards / np.sqrt(self.rms.var + 1e-8)
