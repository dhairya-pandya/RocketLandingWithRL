"""Generalized Advantage Estimation (Schulman et al., 2016)."""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns for arrays shaped (T, n_envs).

    next_values[t] is the value to bootstrap from after step t: V(s_{t+1}), V(final obs) on
    truncation, or 0 on termination. dones[t] (terminated or truncated) stops the trace.
    """
    deltas = rewards + gamma * next_values - values
    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1])
    for t in reversed(range(rewards.shape[0])):
        running = deltas[t] + gamma * lam * (1.0 - dones[t]) * running
        advantages[t] = running
    return advantages, advantages + values
