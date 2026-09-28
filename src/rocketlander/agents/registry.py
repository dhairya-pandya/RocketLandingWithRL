"""Maps algorithm names to their config class and training function."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rocketlander.agents.ppo import PPOConfig, train_ppo
from rocketlander.agents.reinforce import ReinforceConfig, train_reinforce


@dataclass(frozen=True)
class Algorithm:
    config_cls: type
    train: Callable[..., Any]


ALGORITHMS: dict[str, Algorithm] = {
    "ppo": Algorithm(PPOConfig, train_ppo),
    "reinforce": Algorithm(ReinforceConfig, train_reinforce),
}
