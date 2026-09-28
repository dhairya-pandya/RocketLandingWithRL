"""Difficulty levels loaded from rocketlander/configs/levels/*.yaml.

Each level is a set of (low, high) ranges; every reset samples one concrete episode from them.
Keys ending in `_deg` are given in degrees in YAML and stored in radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from functools import cache
from importlib import resources
from typing import Any

import numpy as np
import yaml

from rocketlander.envs.physics import RocketParams, RocketState
from rocketlander.envs.ship import ShipMotion
from rocketlander.envs.wind import WindModel, WindState

Range = tuple[float, float]
NO_RANGE: Range = (0.0, 0.0)


@dataclass(frozen=True)
class LevelConfig:
    name: str
    description: str
    start_offset_x: Range  # relative to the deck centre
    start_altitude: Range
    start_vx: Range
    start_vy: Range
    start_theta: Range
    sway_amplitude: Range = NO_RANGE
    sway_frequency: Range = NO_RANGE
    drift_speed: Range = NO_RANGE
    heave_amplitude: Range = NO_RANGE
    heave_frequency: Range = NO_RANGE
    roll_amplitude: Range = NO_RANGE
    roll_frequency: Range = NO_RANGE
    wind_mean: Range = NO_RANGE
    wind_volatility: Range = NO_RANGE
    mass_scale: Range = (1.0, 1.0)
    thrust_scale: Range = (1.0, 1.0)
    engine_lag: Range = NO_RANGE
    fuel: Range = (5_000.0, 5_000.0)  # kg; limited fuel is what makes hovering a dead end
    max_seconds: float = 40.0


@dataclass(frozen=True)
class EpisodeSetup:
    rocket: RocketState
    params: RocketParams
    ship: ShipMotion
    wind_model: WindModel
    wind: WindState


def _levels_dir():
    return resources.files("rocketlander") / "configs" / "levels"


def available_levels() -> list[str]:
    return sorted(
        p.name.removesuffix(".yaml") for p in _levels_dir().iterdir() if p.name.endswith(".yaml")
    )


def level_from_dict(raw: dict[str, Any]) -> LevelConfig:
    known = {f.name for f in fields(LevelConfig)}
    values: dict[str, Any] = {}
    for key, value in raw.items():
        if key.endswith("_deg"):
            key = key.removesuffix("_deg")
            value = [math.radians(v) for v in value]
        if key not in known:
            raise ValueError(f"Unknown level field {key!r}")
        values[key] = tuple(float(v) for v in value) if isinstance(value, list) else value
    return LevelConfig(**values)


@cache
def get_level(name: str) -> LevelConfig:
    path = _levels_dir() / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"Unknown level {name!r}. Choose from {available_levels()}.")
    return level_from_dict(yaml.safe_load(path.read_text()))


def sample_setup(level: LevelConfig, rng: np.random.Generator) -> EpisodeSetup:
    def draw(bounds: Range) -> float:
        return float(rng.uniform(bounds[0], bounds[1]))

    base = RocketParams()
    params = replace(
        base,
        dry_mass=base.dry_mass * draw(level.mass_scale),
        max_thrust=base.max_thrust * draw(level.thrust_scale),
        engine_lag=draw(level.engine_lag),
        initial_fuel=draw(level.fuel),
    )
    ship = ShipMotion(
        sway_amplitude=draw(level.sway_amplitude),
        sway_frequency=draw(level.sway_frequency),
        sway_phase=draw((0.0, 2 * np.pi)),
        drift_speed=draw(level.drift_speed),
        heave_amplitude=draw(level.heave_amplitude),
        heave_frequency=draw(level.heave_frequency),
        heave_phase=draw((0.0, 2 * np.pi)),
        roll_amplitude=draw(level.roll_amplitude),
        roll_frequency=draw(level.roll_frequency),
        roll_phase=draw((0.0, 2 * np.pi)),
    )
    wind_model = WindModel(mean=draw(level.wind_mean), volatility=draw(level.wind_volatility))
    rocket = RocketState(
        x=draw(level.start_offset_x),
        y=draw(level.start_altitude),
        vx=draw(level.start_vx),
        vy=draw(level.start_vy),
        theta=draw(level.start_theta),
        omega=0.0,
        fuel=params.initial_fuel,
    )
    return EpisodeSetup(rocket, params, ship, wind_model, WindState(speed=wind_model.mean))
