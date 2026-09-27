"""Run an agent on fixed start seeds and summarize the results."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from rocketlander.agents.base import Agent
from rocketlander.envs.rocket_env import RocketLanderEnv


@dataclass
class EvalSummary:
    level: str
    episodes: int
    success_rate: float
    mean_return: float
    mean_touchdown_speed: float  # landings only; nan if none
    mean_fuel_used: float
    reasons: Counter = field(default_factory=Counter)


def evaluate(agent: Agent, level: str, seeds: list[int]) -> EvalSummary:
    env = RocketLanderEnv(level=level)
    returns, fuel, speeds = [], [], []
    reasons: Counter = Counter()
    landed = 0
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        episode_return, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(agent.act(obs, deterministic=True))
            episode_return += reward
            done = terminated or truncated
        returns.append(episode_return)
        fuel.append(info["fuel_used"])
        reasons[info["reason"]] += 1
        if info["outcome"] == "landed":
            landed += 1
            speeds.append(info["touchdown_speed"])
    return EvalSummary(
        level=level,
        episodes=len(seeds),
        success_rate=landed / len(seeds),
        mean_return=float(np.mean(returns)),
        mean_touchdown_speed=float(np.mean(speeds)) if speeds else float("nan"),
        mean_fuel_used=float(np.mean(fuel)),
        reasons=reasons,
    )
