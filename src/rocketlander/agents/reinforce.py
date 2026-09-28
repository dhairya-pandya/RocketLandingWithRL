"""REINFORCE with a learned value baseline (Williams, 1992): Monte Carlo returns, one update
per batch of complete episodes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from rocketlander.common.actor_critic import ActorCritic
from rocketlander.common.curriculum import parse_level
from rocketlander.common.logger import Logger
from rocketlander.common.normalization import RewardScaler
from rocketlander.common.rollout_stats import RolloutStats
from rocketlander.common.vec_env import VecEnv
from rocketlander.envs.rocket_env import RocketLanderEnv
from rocketlander.evaluation import eval_metrics


@dataclass
class ReinforceConfig:
    total_steps: int = 5_000_000
    num_envs: int = 16
    episodes_per_update: int = 32
    gamma: float = 0.999
    learning_rate: float = 3e-4
    value_epochs: int = 5
    max_grad_norm: float = 0.5
    hidden: list[int] = field(default_factory=lambda: [64, 64])
    init_log_std: float = -0.5
    eval_interval: int = 500_000
    eval_episodes: int = 20


def discounted_returns(rewards: np.ndarray, gamma: float, bootstrap: float = 0.0) -> np.ndarray:
    """G_t for one episode; bootstrap is V(final obs) when the episode was truncated."""
    returns = np.zeros(len(rewards))
    running = bootstrap
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def train_reinforce(
    config: ReinforceConfig,
    level: str,
    seed: int,
    logger: Logger,
    agent: ActorCritic | None = None,
) -> ActorCritic:
    torch.manual_seed(seed)
    agent = agent or ActorCritic(config.hidden, config.init_log_std)
    policy_opt = torch.optim.Adam(agent.policy.parameters(), lr=config.learning_rate)
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=config.learning_rate)
    eval_level, schedule = parse_level(level, seed)
    envs = VecEnv(lambda: RocketLanderEnv(level=eval_level), config.num_envs, seed, schedule)
    scaler = RewardScaler(config.num_envs, config.gamma)
    stats = RolloutStats(config.num_envs)
    live = [{"obs": [], "act": [], "rew": []} for _ in range(config.num_envs)]

    obs = envs.reset()
    global_step, next_eval, start = 0, config.eval_interval, time.time()
    while global_step < config.total_steps:
        episodes = []
        while len(episodes) < config.episodes_per_update:
            agent.obs_rms.update(obs)
            with torch.no_grad():
                actions = agent.policy.dist(agent.normalized(obs)).sample().numpy()
            step = envs.step(np.clip(actions, -1.0, 1.0))
            dones = step.terminated | step.truncated
            rewards = scaler.scale(step.rewards, dones)
            stats.add(step.infos, dones)
            global_step += config.num_envs
            for i in range(config.num_envs):
                live[i]["obs"].append(obs[i])
                live[i]["act"].append(actions[i])
                live[i]["rew"].append(rewards[i])
                if dones[i]:
                    bootstrap = 0.0 if step.terminated[i] else agent.value(step.final_obs[i])
                    returns = discounted_returns(np.array(live[i]["rew"]), config.gamma, bootstrap)
                    episodes.append((np.array(live[i]["obs"]), np.array(live[i]["act"]), returns))
                    live[i] = {"obs": [], "act": [], "rew": []}
            obs = step.obs

        metrics = _update(agent, policy_opt, critic_opt, config, episodes)
        metrics |= stats.summary() | (schedule.summary() if schedule else {})
        metrics["train/steps_per_second"] = global_step / (time.time() - start)
        if global_step >= next_eval or global_step >= config.total_steps:
            metrics |= eval_metrics(agent, eval_level, config.eval_episodes)
            next_eval += config.eval_interval
        logger.log(global_step, metrics)
    return agent


def _update(agent, policy_opt, critic_opt, config, episodes) -> dict:
    obs = agent.normalized(np.concatenate([e[0] for e in episodes]))
    actions = torch.as_tensor(np.concatenate([e[1] for e in episodes]), dtype=torch.float32)
    returns = torch.as_tensor(np.concatenate([e[2] for e in episodes]), dtype=torch.float32)

    with torch.no_grad():
        advantages = returns - agent.critic(obs).squeeze(-1)  # the baseline: "better than usual?"
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    log_prob = agent.policy.dist(obs).log_prob(actions).sum(-1)
    policy_loss = -(advantages * log_prob).mean()
    policy_opt.zero_grad()
    policy_loss.backward()
    nn.utils.clip_grad_norm_(agent.policy.parameters(), config.max_grad_norm)
    policy_opt.step()

    for _ in range(config.value_epochs):
        value_loss = 0.5 * (agent.critic(obs).squeeze(-1) - returns).pow(2).mean()
        critic_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(agent.critic.parameters(), config.max_grad_norm)
        critic_opt.step()
    return {
        "loss/policy": policy_loss.item(),
        "loss/value": value_loss.item(),
        "policy/std": agent.policy.log_std.exp().mean().item(),
        "train/episodes_per_update": float(len(episodes)),
    }
