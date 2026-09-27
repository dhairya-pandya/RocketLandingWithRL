"""Uniformly random actions: the lower bound."""

from __future__ import annotations

import numpy as np


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
