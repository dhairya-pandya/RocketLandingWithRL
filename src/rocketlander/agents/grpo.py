"""Group Relative Policy Optimization (Shao et al., 2024) for control: no critic. Each update
restarts a group of rollouts from the same snapshot and scores every rollout against its group."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from rocketlander.common.behavior_cloning import clone_pid
from rocketlander.common.logger import Logger
from rocketlander.common.policy_agent import PolicyAgent
from rocketlander.envs.rocket_env import RocketLanderEnv
from rocketlander.evaluation import EVAL_SEEDS, eval_metrics


@dataclass
class GRPOConfig:
    total_steps: int = 5_000_000
    starts_per_update: int = 8  # different start states ("prompts") per update
    group_size: int = 8  # rollouts from each start state ("completions")
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 4
    num_minibatches: int = 8
    clip_coef: float = 0.2
    max_grad_norm: float = 0.5
    advantage: str = "outcome"  # "outcome": one score per rollout; "process": per-step credit
    process_gamma: float = 0.99
    pretrain: str = "none"  # "bc_pid": clone the PID first, like SFT before RLHF
    pretrain_steps: int = 200_000
    pretrain_noise: float = 0.3
    pretrain_epochs: int = 10
    hidden: list[int] = field(default_factory=lambda: [64, 64])
    init_log_std: float = -0.5
    eval_interval: int = 500_000
    eval_episodes: int = 20


@dataclass
class Group:
    obs: list[np.ndarray]
    actions: list[np.ndarray]
    log_probs: list[np.ndarray]
    rewards: list[np.ndarray]
    returns: np.ndarray  # undiscounted shaped return of each rollout
    landed: np.ndarray
    task_returns: np.ndarray


def rollout_groups(
    agent: PolicyAgent, level: str, start_seeds: list[int], group_size: int
) -> list[Group]:
    """Run group_size stochastic rollouts from each start state, all envs stepped in lockstep."""
    envs, obs = [], []
    for seed in start_seeds:
        first = RocketLanderEnv(level=level)
        first.reset(seed=seed)
        snapshot = first.get_state()  # includes the RNG: every member faces the same wind
        for _ in range(group_size):
            env = RocketLanderEnv(level=level)
            obs.append(env.reset(options={"state": snapshot})[0])
            envs.append(env)
    n = len(envs)
    obs = np.stack(obs)
    traj = [{"obs": [], "act": [], "logp": [], "rew": []} for _ in range(n)]
    returns, task_returns, landed = np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)
    while active.any():
        idx = np.flatnonzero(active)
        with torch.no_grad():
            dist = agent.policy.dist(agent.normalized(obs[idx]))
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(-1).numpy()
        actions = actions.numpy()
        for j, i in enumerate(idx):
            traj[i]["obs"].append(obs[i].copy())  # obs[i] is overwritten by the step below
            traj[i]["act"].append(actions[j])
            traj[i]["logp"].append(log_probs[j])
            obs[i], reward, terminated, truncated, info = envs[i].step(np.clip(actions[j], -1, 1))
            traj[i]["rew"].append(reward)
            returns[i] += reward
            task_returns[i] += info["task_reward"]
            if terminated or truncated:
                active[i] = False
                landed[i] = info["outcome"] == "landed"
    groups = []
    for k in range(len(start_seeds)):
        members = range(k * group_size, (k + 1) * group_size)
        groups.append(
            Group(
                obs=[np.array(traj[i]["obs"]) for i in members],
                actions=[np.array(traj[i]["act"]) for i in members],
                log_probs=[np.array(traj[i]["logp"]) for i in members],
                rewards=[np.array(traj[i]["rew"]) for i in members],
                returns=returns[list(members)],
                landed=landed[list(members)],
                task_returns=task_returns[list(members)],
            )
        )
    return groups


def group_advantages(returns: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Outcome supervision: (R_i - mean) / std within one group, for every step of rollout i."""
    return (returns - returns.mean()) / (returns.std() + eps)


def process_advantages(rewards: list[np.ndarray], gamma: float, eps: float = 1e-8):
    """Process supervision: normalize every step reward across the group, then give each step
    the discounted sum of the normalized rewards that follow it."""
    flat = np.concatenate(rewards)
    mean, std = flat.mean(), flat.std() + eps
    advantages = []
    for r in rewards:
        normalized, running = (r - mean) / std, 0.0
        adv = np.zeros(len(r))
        for t in reversed(range(len(r))):
            running = normalized[t] + gamma * running
            adv[t] = running
        advantages.append(adv)
    return advantages


def step_advantages(group: Group, config: GRPOConfig) -> np.ndarray:
    if config.advantage == "process":
        return np.concatenate(process_advantages(group.rewards, config.process_gamma))
    scores = group_advantages(group.returns)
    return np.concatenate([np.repeat(a, len(o)) for a, o in zip(scores, group.obs, strict=True)])


def train_grpo(
    config: GRPOConfig, level: str, seed: int, logger: Logger, agent: PolicyAgent | None = None
) -> PolicyAgent:
    torch.manual_seed(seed)
    if agent is None:
        agent = PolicyAgent(config.hidden, config.init_log_std)
        if config.pretrain == "bc_pid":
            loss = clone_pid(
                agent,
                level,
                seed,
                config.pretrain_steps,
                config.pretrain_noise,
                config.pretrain_epochs,
            )
            logger.log(
                0, {"pretrain/bc_loss": loss} | eval_metrics(agent, level, config.eval_episodes)
            )
    optimizer = torch.optim.Adam(agent.policy.parameters(), lr=config.learning_rate, eps=1e-5)
    rng = np.random.default_rng(seed)
    global_step, next_eval, update, start = 0, config.eval_interval, 0, time.time()
    while global_step < config.total_steps:
        update += 1
        if config.anneal_lr:
            fraction_left = 1 - global_step / config.total_steps
            optimizer.param_groups[0]["lr"] = config.learning_rate * fraction_left
        start_seeds = rng.integers(0, min(EVAL_SEEDS), size=config.starts_per_update).tolist()
        groups = rollout_groups(agent, level, start_seeds, config.group_size)

        obs = np.concatenate([o for g in groups for o in g.obs])
        actions = np.concatenate([a for g in groups for a in g.actions])
        old_logp = np.concatenate([lp for g in groups for lp in g.log_probs])
        advantages = np.concatenate([step_advantages(g, config) for g in groups])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        global_step += len(obs)
        metrics = _update(agent, optimizer, config, obs, actions, old_logp, advantages)
        agent.obs_rms.update(obs)  # after the update, so old and new log-probs share statistics

        metrics |= {
            "train/success_rate": float(np.mean([g.landed.mean() for g in groups])),
            "train/task_return": float(np.mean([g.task_returns.mean() for g in groups])),
            "train/group_return_std": float(np.mean([g.returns.std() for g in groups])),
            "train/steps_per_second": global_step / (time.time() - start),
        }
        if global_step >= next_eval or global_step >= config.total_steps:
            metrics |= eval_metrics(agent, level, config.eval_episodes)
            next_eval += config.eval_interval
        logger.log(global_step, metrics)
    return agent


def _update(agent, optimizer, config, obs, actions, old_logp, advantages) -> dict:
    b_obs = agent.normalized(obs)
    b_act = torch.as_tensor(actions, dtype=torch.float32)
    b_logp = torch.as_tensor(old_logp, dtype=torch.float32)
    b_adv = torch.as_tensor(advantages, dtype=torch.float32)
    n = len(b_adv)
    clip_fracs = []
    for _ in range(config.update_epochs):
        for idx in torch.randperm(n).split(max(1, n // config.num_minibatches)):
            dist = agent.policy.dist(b_obs[idx])
            ratio = (dist.log_prob(b_act[idx]).sum(-1) - b_logp[idx]).exp()
            clipped = ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef)
            loss = -torch.min(ratio * b_adv[idx], clipped * b_adv[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.policy.parameters(), config.max_grad_norm)
            optimizer.step()
            clip_fracs.append(((ratio - 1).abs() > config.clip_coef).float().mean().item())
    return {
        "loss/policy": loss.item(),
        "loss/clip_fraction": float(np.mean(clip_fracs)),
        "policy/std": agent.policy.log_std.exp().mean().item(),
    }
