"""Soft Actor-Critic (Haarnoja et al., 2018): maximum-entropy RL with twin Q critics and an
automatically tuned temperature."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from rocketlander.common.actor_critic import ACT_SIZE
from rocketlander.common.logger import Logger
from rocketlander.common.networks import QNetwork, TanhGaussianActor
from rocketlander.common.normalization import RunningMeanStd
from rocketlander.common.off_policy import soft_update, train_off_policy
from rocketlander.envs.rocket_env import OBS_SIZE


@dataclass
class SACConfig:
    total_steps: int = 1_000_000
    num_envs: int = 8
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 50_000
    updates_per_step: int = 4  # gradient updates per vector-env step
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    reward_scale: float = 0.1
    warmup_policy: str = "pid"  # "pid" (noisy demonstrations) or "random"
    warmup_noise: float = 0.3
    hidden: list[int] = field(default_factory=lambda: [128, 128])
    log_interval: int = 10_000
    eval_interval: int = 100_000
    eval_episodes: int = 20


class SACAgent(nn.Module):
    def __init__(self, hidden: list[int]) -> None:
        super().__init__()
        self.hidden = list(hidden)
        self.actor = TanhGaussianActor(OBS_SIZE, ACT_SIZE, self.hidden)
        self.q1, self.q2 = (
            QNetwork(OBS_SIZE, ACT_SIZE, hidden),
            QNetwork(OBS_SIZE, ACT_SIZE, hidden),
        )
        self.q1_target, self.q2_target = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.log_alpha = nn.Parameter(torch.zeros(()))
        self.obs_rms = RunningMeanStd((OBS_SIZE,))

    def normalized(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(self.obs_rms.normalize(obs), dtype=torch.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        x = self.normalized(obs)
        action = self.actor.mean_action(x) if deterministic else self.actor(x)[0]
        return action.numpy().astype(np.float32)

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        x = self.normalized(obs)[None]
        a = self.actor.mean_action(x)
        return float(torch.min(self.q1(x, a), self.q2(x, a)))

    def architecture(self) -> dict:
        return {"hidden": self.hidden}

    def extra_state(self) -> dict:
        return {"obs_rms": self.obs_rms.state_dict()}

    def load_extra_state(self, state: dict) -> None:
        self.obs_rms.load_state_dict(state["obs_rms"])


def train_sac(
    config: SACConfig, level: str, seed: int, logger: Logger, agent: SACAgent | None = None
) -> SACAgent:
    torch.manual_seed(seed)
    agent = agent or SACAgent(config.hidden)
    q_opt = torch.optim.Adam(
        [*agent.q1.parameters(), *agent.q2.parameters()], lr=config.learning_rate
    )
    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=config.learning_rate)
    alpha_opt = torch.optim.Adam([agent.log_alpha], lr=config.learning_rate)
    target_entropy = -float(ACT_SIZE)

    def explore(obs: torch.Tensor) -> np.ndarray:
        return agent.actor(obs)[0].numpy()

    def update(batch: dict) -> dict[str, float]:
        o, o2 = agent.normalized(batch["obs"]), agent.normalized(batch["next_obs"])
        a, r, done = batch["actions"], batch["rewards"], batch["terminated"]
        alpha = agent.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = agent.actor(o2)
            q_next = torch.min(agent.q1_target(o2, a2), agent.q2_target(o2, a2)) - alpha * logp2
            target = r + config.gamma * (1.0 - done) * q_next
        q_loss = (agent.q1(o, a) - target).pow(2).mean() + (agent.q2(o, a) - target).pow(2).mean()
        q_opt.zero_grad()
        q_loss.backward()
        q_opt.step()

        a_new, logp = agent.actor(o)
        q_new = torch.min(agent.q1(o, a_new), agent.q2(o, a_new))
        actor_loss = (alpha * logp - q_new).mean()  # maximize Q + alpha * entropy
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        alpha_loss = -(agent.log_alpha * (logp.detach() + target_entropy)).mean()
        alpha_opt.zero_grad()
        alpha_loss.backward()
        alpha_opt.step()

        soft_update(agent.q1_target, agent.q1, config.tau)
        soft_update(agent.q2_target, agent.q2, config.tau)
        return {
            "loss/q": q_loss.item(),
            "loss/actor": actor_loss.item(),
            "policy/alpha": alpha.item(),
            "policy/entropy": -logp.mean().item(),
        }

    return train_off_policy(agent, config, level, seed, logger, explore, update)
