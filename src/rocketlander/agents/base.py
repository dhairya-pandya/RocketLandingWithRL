"""Interface shared by every controller."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Agent(Protocol):
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return an action in [-1, 1]^3."""
        ...
