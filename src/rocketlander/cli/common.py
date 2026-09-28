"""Helpers shared by the command-line tools."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np

from rocketlander.agents.base import Agent
from rocketlander.agents.pid import PIDAgent
from rocketlander.agents.random_agent import RandomAgent
from rocketlander.envs.levels import available_levels
from rocketlander.envs.rocket_env import RocketLanderEnv
from rocketlander.render.renderer import Renderer

AGENTS: dict[str, Callable[[], Agent]] = {"pid": PIDAgent, "random": RandomAgent}


def make_agent(name: str) -> Agent:
    if name not in AGENTS:
        raise ValueError(f"Unknown agent {name!r}. Choose from {sorted(AGENTS)}.")
    return AGENTS[name]()


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--level", default="L0", choices=available_levels())
    parser.add_argument("--seed", type=int, default=0)
    return parser


def step_and_draw(
    env: RocketLanderEnv, renderer: Renderer, agent: Agent, obs: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Advance one step with the agent, record telemetry and draw. Returns (obs, done)."""
    action = agent.act(obs)
    value = agent.value(obs) if hasattr(agent, "value") else None
    obs, _, terminated, truncated, info = env.step(action)
    renderer.telemetry.record(action, info["task_reward"], value)
    renderer.draw(env.frame())
    return obs, terminated or truncated
