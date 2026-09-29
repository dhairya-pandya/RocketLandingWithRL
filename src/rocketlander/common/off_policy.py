"""Training loop shared by the off-policy agents (SAC, TD3): act, store, replay, update."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from rocketlander.agents.pid import PIDAgent
from rocketlander.common.actor_critic import ACT_SIZE
from rocketlander.common.curriculum import parse_level
from rocketlander.common.logger import Logger
from rocketlander.common.replay_buffer import ReplayBuffer
from rocketlander.common.rollout_stats import RolloutStats
from rocketlander.common.vec_env import VecEnv
from rocketlander.envs.rocket_env import OBS_SIZE, RocketLanderEnv
from rocketlander.evaluation import eval_metrics


def train_off_policy(
    agent: Any,
    config: Any,
    level: str,
    seed: int,
    logger: Logger,
    explore: Callable[[torch.Tensor], np.ndarray],
    update: Callable[[dict], dict[str, float]],
    snap: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Any:
    """Collect with `explore`, store every transition, and call `update` on replayed batches.

    `snap` maps warm-up actions onto the actions the agent can take (DQN's grid)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    eval_level, schedule = parse_level(level, seed)
    envs = VecEnv(lambda: RocketLanderEnv(level=eval_level), config.num_envs, seed, schedule)
    buffer = ReplayBuffer(config.buffer_size, OBS_SIZE, ACT_SIZE, seed)
    stats = RolloutStats(config.num_envs)

    warmup_agent = PIDAgent() if config.warmup_policy == "pid" else None
    obs = envs.reset()
    agent.obs_rms.update(obs)
    global_step, next_log, next_eval, start = (
        0,
        config.log_interval,
        config.eval_interval,
        time.time(),
    )
    metrics: dict[str, float] = {}
    while global_step < config.total_steps:
        if global_step < config.learning_starts and warmup_agent is not None:
            # noisy demonstrations: off-policy learners can learn from another policy's data
            actions = np.stack([warmup_agent.act(o) for o in obs])
            actions += rng.normal(0.0, config.warmup_noise, actions.shape)
            actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
            if snap is not None:
                actions = snap(actions)
        elif global_step < config.learning_starts:  # uniform random actions fill the buffer
            actions = rng.uniform(-1.0, 1.0, size=(config.num_envs, ACT_SIZE)).astype(np.float32)
            if snap is not None:
                actions = snap(actions)
        else:
            with torch.no_grad():
                actions = explore(agent.normalized(obs))
        envs.record_outcomes = global_step >= config.learning_starts
        step = envs.step(actions)
        buffer.add(
            obs, actions, step.rewards * config.reward_scale, step.final_obs, step.terminated
        )
        stats.add(step.infos, step.terminated | step.truncated)
        obs = step.obs
        agent.obs_rms.update(obs)
        global_step += config.num_envs

        if global_step >= config.learning_starts:
            for _ in range(config.updates_per_step):
                metrics = update(buffer.sample(config.batch_size))

        if global_step >= next_log:
            row = metrics | stats.summary() | (schedule.summary() if schedule else {})
            row["train/steps_per_second"] = global_step / (time.time() - start)
            if global_step >= next_eval:
                row |= eval_metrics(agent, eval_level, config.eval_episodes)
                next_eval += config.eval_interval
            logger.log(global_step, row)
            next_log += config.log_interval
    logger.log(global_step, eval_metrics(agent, eval_level, config.eval_episodes))
    return agent


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """Polyak averaging: target <- (1 - tau) target + tau source."""
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters(), strict=True):
            t.lerp_(s, tau)
