"""Maps algorithm names to their config class and training function."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rocketlander.agents.ppo import PPOConfig, train_ppo
from rocketlander.agents.reinforce import ReinforceConfig, train_reinforce
from rocketlander.agents.sac import SACAgent, SACConfig, train_sac
from rocketlander.agents.td3 import TD3Agent, TD3Config, train_td3
from rocketlander.common.actor_critic import ActorCritic


@dataclass(frozen=True)
class Algorithm:
    config_cls: type
    train: Callable[..., Any]
    agent_cls: type  # rebuilt from a checkpoint's "architecture"


ALGORITHMS: dict[str, Algorithm] = {
    "ppo": Algorithm(PPOConfig, train_ppo, ActorCritic),
    "reinforce": Algorithm(ReinforceConfig, train_reinforce, ActorCritic),
    "sac": Algorithm(SACConfig, train_sac, SACAgent),
    "td3": Algorithm(TD3Config, train_td3, TD3Agent),
}
