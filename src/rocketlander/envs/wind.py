"""Horizontal wind as an Ornstein-Uhlenbeck process (mean plus decaying gusts)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class WindModel:
    mean: float = 0.0  # m/s, positive blows right
    reversion: float = 0.5  # 1/s
    volatility: float = 0.0  # gust strength


@dataclass(frozen=True)
class WindState:
    speed: float


def step_wind(state: WindState, model: WindModel, dt: float, rng: np.random.Generator) -> WindState:
    drift = model.reversion * (model.mean - state.speed) * dt
    noise = model.volatility * np.sqrt(dt) * rng.standard_normal()
    return replace(state, speed=state.speed + drift + noise)
