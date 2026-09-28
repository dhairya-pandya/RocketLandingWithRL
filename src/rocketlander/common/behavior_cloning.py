"""Behavior cloning of the PID controller: the "supervised fine-tuning" step before GRPO."""

from __future__ import annotations

import numpy as np
import torch

from rocketlander.agents.pid import PIDAgent
from rocketlander.common.vec_env import VecEnv
from rocketlander.envs.rocket_env import RocketLanderEnv


def collect_pid_demonstrations(level: str, seed: int, steps: int, noise: float, num_envs: int = 16):
    """States visited by a noisy PID, labelled with the clean PID action (so the clone also
    learns how to recover from its own small mistakes)."""
    pid, rng = PIDAgent(), np.random.default_rng(seed)
    envs = VecEnv(lambda: RocketLanderEnv(level=level), num_envs, seed)
    obs, states, labels = envs.reset(), [], []
    for _ in range(steps // num_envs):
        clean = np.stack([pid.act(o) for o in obs])
        states.append(obs)
        labels.append(clean)
        noisy = np.clip(clean + rng.normal(0.0, noise, clean.shape), -1.0, 1.0)
        obs = envs.step(noisy.astype(np.float32)).obs
    return np.concatenate(states), np.concatenate(labels)


def clone_pid(agent, level: str, seed: int, steps: int, noise: float, epochs: int) -> float:
    """Fit the policy mean to the PID's actions by mean-squared error; returns the final loss."""
    states, labels = collect_pid_demonstrations(level, seed, steps, noise)
    agent.obs_rms.update(states)
    x, y = agent.normalized(states), torch.as_tensor(labels, dtype=torch.float32)
    optimizer = torch.optim.Adam(agent.policy.mean.parameters(), lr=1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(x)).split(256):
            loss = (agent.policy.mean(x[idx]) - y[idx]).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss.item()
