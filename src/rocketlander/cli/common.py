"""Helpers shared by the command-line tools."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rocketlander.agents.base import Agent
from rocketlander.agents.mpc import MPCAgent
from rocketlander.agents.pid import PIDAgent
from rocketlander.agents.random_agent import RandomAgent
from rocketlander.common.checkpoint import load_checkpoint
from rocketlander.envs.levels import available_levels
from rocketlander.envs.rocket_env import RocketLanderEnv
from rocketlander.render.renderer import Renderer

AGENTS: dict[str, Callable[[], Agent]] = {"pid": PIDAgent, "random": RandomAgent, "mpc": MPCAgent}


def make_agent(name: str) -> Agent:
    """A built-in agent by name, or a trained agent from a checkpoint path (*.pt)."""
    if name.endswith(".pt"):
        return load_checkpoint(Path(name))[0]
    if name not in AGENTS:
        raise ValueError(f"Unknown agent {name!r}. Choose from {sorted(AGENTS)} or a .pt path.")
    return AGENTS[name]()


def agent_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent", default="pid", help=f"{', '.join(sorted(AGENTS))} or a .pt path")


def base_parser(
    description: str, seed: bool = True, level_specs: bool = False
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    if level_specs:  # training also accepts "mix:L0,L3" and "curriculum:L0,L1,L2,L3,L4"
        parser.add_argument("--level", default="L0")
    else:
        parser.add_argument("--level", default="L0", choices=available_levels())
    if seed:
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
