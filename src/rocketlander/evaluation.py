"""Run an agent on fixed start seeds and summarize the results."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from rocketlander.agents.base import Agent
from rocketlander.envs.levels import LevelConfig
from rocketlander.envs.rocket_env import RocketLanderEnv
from rocketlander.stats import wilson_interval

EVAL_SEEDS = list(range(10_000, 10_100))  # held out: training envs are seeded below 10 000


@dataclass
class EvalSummary:
    level: str
    episodes: int
    success_rate: float
    mean_task_return: float  # unshaped reward, comparable across agents
    mean_touchdown_speed: float  # landings only; nan if none
    mean_fuel_used: float
    reasons: Counter = field(default_factory=Counter)

    @property
    def success_ci(self) -> tuple[float, float]:
        """95% Wilson interval: 100 episodes pin a success rate down to roughly ±10 points."""
        return wilson_interval(round(self.success_rate * self.episodes), self.episodes)


def evaluate(agent: Agent, level: str | LevelConfig, seeds: list[int]) -> EvalSummary:
    env = RocketLanderEnv(level=level)
    returns, fuel, speeds = [], [], []
    reasons: Counter = Counter()
    landed = 0
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        episode_return, done = 0.0, False
        while not done:
            obs, _, terminated, truncated, info = env.step(agent.act(obs, deterministic=True))
            episode_return += info["task_reward"]
            done = terminated or truncated
        returns.append(episode_return)
        fuel.append(info["fuel_used"])
        reasons[info["reason"]] += 1
        if info["outcome"] == "landed":
            landed += 1
            speeds.append(info["touchdown_speed"])
    return EvalSummary(
        level=env.level.name,
        episodes=len(seeds),
        success_rate=landed / len(seeds),
        mean_task_return=float(np.mean(returns)),
        mean_touchdown_speed=float(np.mean(speeds)) if speeds else float("nan"),
        mean_fuel_used=float(np.mean(fuel)),
        reasons=reasons,
    )


def eval_metrics(agent: Agent, level: str, episodes: int) -> dict[str, float]:
    """Success rate and task return on the first `episodes` held-out seeds, for logging."""
    summary = evaluate(agent, level, EVAL_SEEDS[:episodes])
    return {"eval/success_rate": summary.success_rate, "eval/task_return": summary.mean_task_return}
