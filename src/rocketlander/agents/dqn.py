"""Double DQN (van Hasselt et al., 2016) on a grid of discrete actions.

The rocket's three continuous controls are cut into 5 throttle x 5 gimbal x 3 RCS levels, so a
Q-network can score all 75 actions at once and act greedily, like DQN on Atari. The levels are
dense near hovering: a uniform grid cannot hold a gentle descent, and the PID snapped to it never
lands; snapped to this grid it lands as often as the continuous PID.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from rocketlander.common.logger import Logger
from rocketlander.common.networks import mlp
from rocketlander.common.normalization import RunningMeanStd
from rocketlander.common.off_policy import soft_update, train_off_policy
from rocketlander.envs.rocket_env import OBS_SIZE

LEVELS = ([-1, -0.2, 0, 0.2, 1], [-0.3, -0.05, 0, 0.05, 0.3], [-0.5, 0, 0.5])  # per control
ACTIONS = np.array(list(itertools.product(*LEVELS)), dtype=np.float32)  # (75, 3)
BRANCHES = [len(levels) for levels in LEVELS]


class BranchingQ(nn.Module):
    """Q(s, a) = V(s) + sum over controls of a centred advantage A_d(s, a_d) (Tavakoli et al.,
    2018). Every transition trains every branch, so rarely tried combinations still get sensible
    values, and the best action is the best level of each control on its own."""

    def __init__(self, hidden: list[int]) -> None:
        super().__init__()
        self.net = mlp([OBS_SIZE, *hidden, 1 + sum(BRANCHES)], activation=nn.ReLU)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Q-values of all 75 grid actions, in the order of ACTIONS."""
        value, throttle, gimbal, rcs = self.net(obs).split([1, *BRANCHES], dim=-1)
        throttle, gimbal, rcs = (a - a.mean(dim=-1, keepdim=True) for a in (throttle, gimbal, rcs))
        q = (
            value[..., None, None]
            + throttle[..., :, None, None]
            + gimbal[..., None, :, None]
            + rcs[..., None, None, :]
        )  # (..., 5, 5, 3), the product grid
        return q.flatten(start_dim=-3)


def nearest_action(actions: np.ndarray) -> np.ndarray:
    """Index of the grid action closest to each continuous action (n, 3)."""
    return np.argmin(((actions[:, None, :] - ACTIONS[None]) ** 2).sum(-1), axis=1)


@dataclass
class DQNConfig:
    total_steps: int = 2_000_000
    num_envs: int = 8
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 50_000
    updates_per_step: int = 2
    gamma: float = 0.99
    tau: float = 0.001  # slow target-network updates keep Q-learning stable
    learning_rate: float = 1e-4
    reward_scale: float = 0.1
    warmup_policy: str = "pid"  # PID demonstrations snapped to the action grid
    warmup_noise: float = 0.3
    epsilon_start: float = 0.02  # little random exploration: the PID warm-up supplies coverage
    epsilon_end: float = 0.005
    epsilon_decay_steps: int = 300_000
    hidden: list[int] = field(default_factory=lambda: [128, 128])
    log_interval: int = 10_000
    eval_interval: int = 100_000
    eval_episodes: int = 20


class DQNAgent(nn.Module):
    def __init__(self, hidden: list[int]) -> None:
        super().__init__()
        self.hidden = list(hidden)
        self.q = BranchingQ(self.hidden)
        self.q_target = copy.deepcopy(self.q)
        self.obs_rms = RunningMeanStd((OBS_SIZE,))

    def normalized(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(self.obs_rms.normalize(obs), dtype=torch.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return ACTIONS[self.q(self.normalized(obs)).argmax(dim=-1).numpy()]  # one obs or a batch

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        return float(self.q(self.normalized(obs)).max())

    def architecture(self) -> dict:
        return {"hidden": self.hidden}

    def extra_state(self) -> dict:
        return {"obs_rms": self.obs_rms.state_dict()}

    def load_extra_state(self, state: dict) -> None:
        self.obs_rms.load_state_dict(state["obs_rms"])


def train_dqn(
    config: DQNConfig, level: str, seed: int, logger: Logger, agent: DQNAgent | None = None
) -> DQNAgent:
    torch.manual_seed(seed)
    agent = agent or DQNAgent(config.hidden)
    optimizer = torch.optim.Adam(agent.q.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(seed + 1)
    explored = 0

    def explore(obs: torch.Tensor) -> np.ndarray:
        nonlocal explored
        explored += len(obs)
        progress = min(1.0, explored / config.epsilon_decay_steps)
        epsilon = config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)
        greedy = agent.q(obs).argmax(dim=1).numpy()
        random = rng.integers(len(ACTIONS), size=len(obs))
        return ACTIONS[np.where(rng.random(len(obs)) < epsilon, random, greedy)]

    def update(batch: dict) -> dict[str, float]:
        o, o2 = agent.normalized(batch["obs"]), agent.normalized(batch["next_obs"])
        a = torch.as_tensor(nearest_action(batch["actions"].numpy()))
        r, done = batch["rewards"], batch["terminated"]
        with torch.no_grad():  # Double DQN: the online net picks, the target net evaluates
            best = agent.q(o2).argmax(dim=1, keepdim=True)
            target = r + config.gamma * (1.0 - done) * agent.q_target(o2).gather(1, best)[:, 0]
        q = agent.q(o).gather(1, a[:, None])[:, 0]
        loss = nn.functional.smooth_l1_loss(q, target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.q.parameters(), 10.0)
        optimizer.step()
        soft_update(agent.q_target, agent.q, config.tau)
        return {"loss/q": loss.item(), "q/mean": q.mean().item()}

    return train_off_policy(
        agent,
        config,
        level,
        seed,
        logger,
        explore,
        update,
        snap=lambda a: ACTIONS[nearest_action(a)],
    )
