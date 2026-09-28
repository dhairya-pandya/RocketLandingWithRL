"""Small MLPs, policies and Q-networks shared by the learning agents."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def mlp(
    sizes: list[int], output_gain: float = 1.0, activation: type[nn.Module] = nn.Tanh
) -> nn.Sequential:
    """MLP with orthogonal init; the last layer's gain sets its initial output scale."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        last = i == len(sizes) - 2
        nn.init.orthogonal_(linear.weight, output_gain if last else np.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers += [linear] if last else [linear, activation()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian with a state-independent learned log std."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int], init_log_std: float = 0.0):
        super().__init__()
        self.mean = mlp([obs_dim, *hidden, act_dim], output_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.mean(obs), self.log_std.exp())


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class TanhGaussianActor(nn.Module):
    """SAC actor: a state-dependent Gaussian squashed by tanh into [-1, 1]."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int]) -> None:
        super().__init__()
        self.net = mlp([obs_dim, *hidden, 2 * act_dim], output_gain=0.01, activation=nn.ReLU)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized action sample and its log-probability (with the tanh correction)."""
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        u = mean + std * torch.randn_like(mean)
        action = torch.tanh(u)
        # log pi(a) = log N(u) - sum log(1 - tanh(u)^2): change of variables through tanh
        log_prob = torch.distributions.Normal(mean, std).log_prob(u)
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1)

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs).chunk(2, dim=-1)[0])


class DeterministicActor(nn.Module):
    """TD3 actor: a deterministic action in [-1, 1]."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int]) -> None:
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim], output_gain=0.01, activation=nn.ReLU)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class QNetwork(nn.Module):
    """Q(s, a) from the concatenated observation and action."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int]) -> None:
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden, 1], activation=nn.ReLU)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)
