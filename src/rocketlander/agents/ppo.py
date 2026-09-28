"""Proximal Policy Optimization (Schulman et al., 2017) with GAE and a clipped surrogate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from rocketlander.common.actor_critic import ACT_SIZE, ActorCritic
from rocketlander.common.curriculum import parse_level
from rocketlander.common.gae import compute_gae
from rocketlander.common.logger import Logger
from rocketlander.common.normalization import RewardScaler
from rocketlander.common.rollout_stats import RolloutStats
from rocketlander.common.vec_env import VecEnv
from rocketlander.envs.rocket_env import OBS_SIZE, RocketLanderEnv
from rocketlander.evaluation import eval_metrics


@dataclass
class PPOConfig:
    total_steps: int = 5_000_000
    num_envs: int = 32
    num_steps: int = 256  # steps per env per update
    gamma: float = 0.999
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 10
    num_minibatches: int = 32
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: list[int] = field(default_factory=lambda: [64, 64])
    init_log_std: float = -0.5
    eval_interval: int = 500_000
    eval_episodes: int = 20


def train_ppo(
    config: PPOConfig, level: str, seed: int, logger: Logger, agent: ActorCritic | None = None
) -> ActorCritic:
    torch.manual_seed(seed)
    agent = agent or ActorCritic(config.hidden, config.init_log_std)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    eval_level, schedule = parse_level(level, seed)
    envs = VecEnv(lambda: RocketLanderEnv(level=eval_level), config.num_envs, seed, schedule)
    scaler = RewardScaler(config.num_envs, config.gamma)
    stats = RolloutStats(config.num_envs)
    n_steps, n_envs = config.num_steps, config.num_envs
    num_updates = max(1, config.total_steps // (n_steps * n_envs))

    obs_buf = np.zeros((n_steps, n_envs, OBS_SIZE), dtype=np.float32)
    act_buf = np.zeros((n_steps, n_envs, ACT_SIZE), dtype=np.float32)
    logp_buf, rew_buf, val_buf, next_val_buf, done_buf = (
        np.zeros((n_steps, n_envs)) for _ in range(5)
    )

    obs = envs.reset()
    agent.obs_rms.update(obs)
    next_eval, start = config.eval_interval, time.time()
    for update in range(1, num_updates + 1):
        if config.anneal_lr:
            fraction_left = 1 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = config.learning_rate * fraction_left

        for t in range(n_steps):
            obs_buf[t] = obs
            with torch.no_grad():
                x = agent.normalized(obs)
                dist = agent.policy.dist(x)
                action = dist.sample()
                logp_buf[t] = dist.log_prob(action).sum(-1).numpy()
                val_buf[t] = agent.critic(x).squeeze(-1).numpy()
            act_buf[t] = action.numpy()
            step = envs.step(np.clip(act_buf[t], -1.0, 1.0))
            dones = step.terminated | step.truncated
            rew_buf[t] = scaler.scale(step.rewards, dones)
            done_buf[t] = dones
            stats.add(step.infos, dones)
            with torch.no_grad():  # bootstrap: V(final obs) on truncation, 0 on termination
                next_val_buf[t] = agent.critic(agent.normalized(step.final_obs)).squeeze(-1).numpy()
            next_val_buf[t][step.terminated] = 0.0
            obs = step.obs
            agent.obs_rms.update(obs)

        advantages, returns = compute_gae(
            rew_buf, val_buf, next_val_buf, done_buf, config.gamma, config.gae_lambda
        )
        metrics = _update(agent, optimizer, config, obs_buf, act_buf, logp_buf, advantages, returns)

        global_step = update * n_steps * n_envs
        metrics |= stats.summary() | (schedule.summary() if schedule else {})
        metrics["train/steps_per_second"] = global_step / (time.time() - start)
        if global_step >= next_eval or update == num_updates:
            metrics |= eval_metrics(agent, eval_level, config.eval_episodes)
            next_eval += config.eval_interval
        logger.log(global_step, metrics)
    return agent


def _update(agent, optimizer, config, obs_buf, act_buf, logp_buf, advantages, returns) -> dict:
    b_obs = agent.normalized(obs_buf.reshape(-1, OBS_SIZE))
    b_act = torch.as_tensor(act_buf.reshape(-1, ACT_SIZE))
    b_logp = torch.as_tensor(logp_buf.reshape(-1), dtype=torch.float32)
    b_adv = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32)
    b_ret = torch.as_tensor(returns.reshape(-1), dtype=torch.float32)
    batch_size = len(b_adv)
    clip_fracs, kls = [], []
    for _ in range(config.update_epochs):
        for idx in torch.randperm(batch_size).split(batch_size // config.num_minibatches):
            dist = agent.policy.dist(b_obs[idx])
            logp = dist.log_prob(b_act[idx]).sum(-1)
            ratio = (logp - b_logp[idx]).exp()
            adv = (b_adv[idx] - b_adv[idx].mean()) / (b_adv[idx].std() + 1e-8)
            clipped = ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef)
            policy_loss = torch.max(-adv * ratio, -adv * clipped).mean()
            value_loss = 0.5 * (agent.critic(b_obs[idx]).squeeze(-1) - b_ret[idx]).pow(2).mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                kls.append(((ratio - 1) - (logp - b_logp[idx])).mean().item())
                clip_fracs.append(((ratio - 1).abs() > config.clip_coef).float().mean().item())
    return {
        "loss/policy": policy_loss.item(),
        "loss/value": value_loss.item(),
        "loss/entropy": entropy.item(),
        "loss/approx_kl": float(np.mean(kls)),
        "loss/clip_fraction": float(np.mean(clip_fracs)),
        "policy/std": agent.policy.log_std.exp().mean().item(),
    }
