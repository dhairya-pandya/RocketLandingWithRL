"""Gaussian actor plus value critic with observation normalization, shared by PPO and REINFORCE."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rocketlander.common.history import History, history_size
from rocketlander.common.networks import GaussianPolicy, mlp
from rocketlander.common.normalization import RunningMeanStd
from rocketlander.envs.rocket_env import ACT_SIZE


class ActorCritic(nn.Module):
    """history > 1: the networks see the last `history` observations and actions (see
    common/history.py); `act` keeps that memory itself, one episode at a time."""

    def __init__(self, hidden: list[int], init_log_std: float, history: int = 1) -> None:
        super().__init__()
        self.hidden, self.init_log_std, self.history = list(hidden), init_log_std, history
        obs_dim = history_size(history)
        self.policy = GaussianPolicy(obs_dim, ACT_SIZE, self.hidden, init_log_std)
        self.critic = mlp([obs_dim, *self.hidden, 1], output_gain=1.0)
        self.obs_rms = RunningMeanStd((obs_dim,))
        self._memory = History(history, 1) if history > 1 else None
        self.reset()

    def reset(self) -> None:
        """Start a new episode: forget the remembered observations and actions."""
        self._fresh, self._last_action = True, np.zeros(ACT_SIZE, dtype=np.float32)

    def normalized(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(self.obs_rms.normalize(obs), dtype=torch.float32)

    def _remember(self, obs: np.ndarray) -> np.ndarray:
        if self._memory is None:
            return obs
        if self._fresh:
            self._memory.reset(0, obs)
            self._fresh = False
        else:
            self._memory.push(obs[None], self._last_action[None])
        return self._memory.stacked()[0]

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        dist = self.policy.dist(self.normalized(self._remember(obs)))
        action = dist.mean if deterministic else dist.sample()
        action = np.clip(action.numpy(), -1.0, 1.0).astype(np.float32)
        self._last_action = action
        return action

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        """V(s) of the latest observation passed to `act` (or of `obs` without a history)."""
        x = obs if self._memory is None else self._memory.stacked()[0]
        return float(self.critic(self.normalized(x)).squeeze())

    def architecture(self) -> dict:
        return {"hidden": self.hidden, "init_log_std": self.init_log_std, "history": self.history}

    def extra_state(self) -> dict:
        return {"obs_rms": self.obs_rms.state_dict()}

    def load_extra_state(self, state: dict) -> None:
        self.obs_rms.load_state_dict(state["obs_rms"])
