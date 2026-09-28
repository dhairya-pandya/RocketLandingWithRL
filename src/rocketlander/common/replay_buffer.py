"""Uniform replay buffer for off-policy agents (stores raw observations)."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, seed: int) -> None:
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, obs, actions, rewards, next_obs, terminated) -> None:
        """Add a batch of transitions (one per env); next_obs must be the true final observation."""
        for i in range(len(rewards)):
            j = self.pos
            self.obs[j], self.actions[j], self.rewards[j] = obs[i], actions[i], rewards[i]
            self.next_obs[j], self.terminated[j] = next_obs[i], terminated[i]
            self.pos, self.size = (j + 1) % self.capacity, min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = self.rng.integers(self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": torch.as_tensor(self.actions[idx]),
            "rewards": torch.as_tensor(self.rewards[idx]),
            "next_obs": self.next_obs[idx],
            "terminated": torch.as_tensor(self.terminated[idx]),
        }
