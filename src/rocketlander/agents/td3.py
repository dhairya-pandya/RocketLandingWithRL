"""Twin Delayed DDPG (Fujimoto et al., 2018): a deterministic actor with twin critics, target
policy smoothing and delayed actor updates."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from rocketlander.common.actor_critic import ACT_SIZE
from rocketlander.common.logger import Logger
from rocketlander.common.networks import DeterministicActor, QNetwork
from rocketlander.common.normalization import RunningMeanStd
from rocketlander.common.off_policy import soft_update, train_off_policy
from rocketlander.envs.rocket_env import OBS_SIZE


@dataclass
class TD3Config:
    total_steps: int = 1_000_000
    num_envs: int = 8
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 50_000
    updates_per_step: int = 4
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    reward_scale: float = 0.1
    warmup_policy: str = "pid"  # "pid" (noisy demonstrations) or "random"
    warmup_noise: float = 0.3
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    hidden: list[int] = field(default_factory=lambda: [128, 128])
    log_interval: int = 10_000
    eval_interval: int = 100_000
    eval_episodes: int = 20


class TD3Agent(nn.Module):
    def __init__(self, hidden: list[int]) -> None:
        super().__init__()
        self.hidden = list(hidden)
        self.actor = DeterministicActor(OBS_SIZE, ACT_SIZE, self.hidden)
        self.q1, self.q2 = (
            QNetwork(OBS_SIZE, ACT_SIZE, hidden),
            QNetwork(OBS_SIZE, ACT_SIZE, hidden),
        )
        self.actor_target = copy.deepcopy(self.actor)
        self.q1_target, self.q2_target = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.obs_rms = RunningMeanStd((OBS_SIZE,))
        self.exploration_noise = 0.1

    def normalized(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(self.obs_rms.normalize(obs), dtype=torch.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action = self.actor(self.normalized(obs))
        if not deterministic:
            action = (action + self.exploration_noise * torch.randn_like(action)).clamp(-1, 1)
        return action.numpy().astype(np.float32)

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        x = self.normalized(obs)[None]
        a = self.actor(x)
        return float(torch.min(self.q1(x, a), self.q2(x, a)))

    def architecture(self) -> dict:
        return {"hidden": self.hidden}

    def extra_state(self) -> dict:
        return {"obs_rms": self.obs_rms.state_dict()}

    def load_extra_state(self, state: dict) -> None:
        self.obs_rms.load_state_dict(state["obs_rms"])


def train_td3(
    config: TD3Config, level: str, seed: int, logger: Logger, agent: TD3Agent | None = None
) -> TD3Agent:
    torch.manual_seed(seed)
    agent = agent or TD3Agent(config.hidden)
    agent.exploration_noise = config.exploration_noise
    q_opt = torch.optim.Adam(
        [*agent.q1.parameters(), *agent.q2.parameters()], lr=config.learning_rate
    )
    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=config.learning_rate)
    updates = 0
    actor_loss = torch.zeros(())

    def explore(obs: torch.Tensor) -> np.ndarray:
        action = agent.actor(obs)
        return (action + config.exploration_noise * torch.randn_like(action)).clamp(-1, 1).numpy()

    def update(batch: dict) -> dict[str, float]:
        nonlocal updates, actor_loss
        o, o2 = agent.normalized(batch["obs"]), agent.normalized(batch["next_obs"])
        a, r, done = batch["actions"], batch["rewards"], batch["terminated"]
        with torch.no_grad():  # target policy smoothing: noisy target actions
            noise = (config.policy_noise * torch.randn_like(a)).clamp(
                -config.noise_clip, config.noise_clip
            )
            a2 = (agent.actor_target(o2) + noise).clamp(-1.0, 1.0)
            q_next = torch.min(agent.q1_target(o2, a2), agent.q2_target(o2, a2))
            target = r + config.gamma * (1.0 - done) * q_next
        q_loss = (agent.q1(o, a) - target).pow(2).mean() + (agent.q2(o, a) - target).pow(2).mean()
        q_opt.zero_grad()
        q_loss.backward()
        q_opt.step()

        updates += 1
        if updates % config.policy_delay == 0:  # delayed actor and target updates
            actor_loss = -agent.q1(o, agent.actor(o)).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            soft_update(agent.actor_target, agent.actor, config.tau)
            soft_update(agent.q1_target, agent.q1, config.tau)
            soft_update(agent.q2_target, agent.q2, config.tau)
        return {"loss/q": q_loss.item(), "loss/actor": actor_loss.item()}

    return train_off_policy(agent, config, level, seed, logger, explore, update)
