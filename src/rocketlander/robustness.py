"""Sweep one physical parameter beyond its training range and measure success at each value."""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from importlib import resources

import yaml

from rocketlander.agents.base import Agent
from rocketlander.envs.levels import LevelConfig, get_level
from rocketlander.evaluation import EvalSummary, evaluate


@cache
def sweep_values() -> dict[str, list[float]]:
    path = resources.files("rocketlander") / "configs" / "robustness.yaml"
    return {
        name: [float(v) for v in values]
        for name, values in yaml.safe_load(path.read_text()).items()
    }


def perturbed(level: LevelConfig, param: str, value: float) -> LevelConfig:
    """`level` with `param` fixed to `value` in every episode."""
    return replace(level, name=f"{level.name} {param}={value:g}", **{param: (value, value)})


def sweep(
    agent: Agent, level: str, param: str, seeds: list[int]
) -> list[tuple[float, EvalSummary]]:
    if param not in sweep_values():
        raise ValueError(f"Unknown sweep {param!r}. Choose from {sorted(sweep_values())}.")
    base = get_level(level)
    return [
        (value, evaluate(agent, perturbed(base, param, value), seeds))
        for value in sweep_values()[param]
    ]
