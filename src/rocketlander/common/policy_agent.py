"""A Gaussian policy with observation normalization and no critic (for GRPO)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rocketlander.common.actor_critic import ACT_SIZE
from rocketlander.common.networks import GaussianPolicy
from rocketlander.common.normalization import RunningMeanStd
from rocketlander.envs.rocket_env import OBS_SIZE


class PolicyAgent(nn.Module):
    def __init__(self, hidden: list[int], init_log_std: float) -> None:
        super().__init__()
        self.hidden, self.init_log_std = list(hidden), init_log_std
        self.policy = GaussianPolicy(OBS_SIZE, ACT_SIZE, self.hidden, init_log_std)
        self.obs_rms = RunningMeanStd((OBS_SIZE,))

    def normalized(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(self.obs_rms.normalize(obs), dtype=torch.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        dist = self.policy.dist(self.normalized(obs))
        action = dist.mean if deterministic else dist.sample()
        return np.clip(action.numpy(), -1.0, 1.0).astype(np.float32)

    def architecture(self) -> dict:
        return {"hidden": self.hidden, "init_log_std": self.init_log_std}

    def extra_state(self) -> dict:
        return {"obs_rms": self.obs_rms.state_dict()}

    def load_extra_state(self, state: dict) -> None:
        self.obs_rms.load_state_dict(state["obs_rms"])
