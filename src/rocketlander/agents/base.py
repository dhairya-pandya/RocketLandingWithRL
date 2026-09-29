"""Interface shared by every controller."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Agent(Protocol):
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return an action in [-1, 1]^3."""
        ...


def start_episode(agent: Agent) -> None:
    """Let agents that keep per-episode state (the MPC planner's plan) start afresh."""
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset()
