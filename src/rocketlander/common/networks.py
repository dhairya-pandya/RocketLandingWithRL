"""Small MLPs and the Gaussian policy used by the on-policy agents."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def mlp(sizes: list[int], output_gain: float = 1.0) -> nn.Sequential:
    """Tanh MLP with orthogonal init; the last layer's gain sets its initial output scale."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        last = i == len(sizes) - 2
        nn.init.orthogonal_(linear.weight, output_gain if last else np.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers += [linear] if last else [linear, nn.Tanh()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian with a state-independent learned log std."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int], init_log_std: float = 0.0):
        super().__init__()
        self.mean = mlp([obs_dim, *hidden, act_dim], output_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.mean(obs), self.log_std.exp())
