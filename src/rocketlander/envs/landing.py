"""Touchdown judge: ends the episode when a leg reaches the deck or the sea."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from rocketlander.envs.physics import SEA_LEVEL, RocketParams, RocketState, body_to_world
from rocketlander.envs.ship import DeckState

MAX_VERTICAL_SPEED = 2.0  # relative to the deck
MAX_HORIZONTAL_SPEED = 1.5
MAX_TILT = np.radians(10.0)
MAX_SPIN = 0.3


class Outcome(Enum):
    IN_FLIGHT = "in_flight"
    LANDED = "landed"
    CRASHED = "crashed"
    FAILED = "failed"


@dataclass(frozen=True)
class Judgement:
    outcome: Outcome
    reason: str = ""
    touchdown_speed: float = 0.0

    @property
    def done(self) -> bool:
        return self.outcome is not Outcome.IN_FLIGHT


@dataclass(frozen=True)
class Bounds:
    """Flight box relative to the deck centre."""

    max_horizontal_offset: float = 400.0
    max_altitude: float = 1_500.0


def judge(rocket: RocketState, params: RocketParams, deck: DeckState, bounds: Bounds) -> Judgement:
    tips = body_to_world(rocket, params.leg_tips_body())
    on_deck = [deck.contains(x) and y <= deck.surface_height(x) for x, y in tips]
    in_sea = [y <= SEA_LEVEL for _, y in tips]

    if any(on_deck):
        return _judge_touchdown(rocket, deck, tips)
    if any(in_sea):
        return Judgement(Outcome.CRASHED, "missed the ship")
    if abs(rocket.x - deck.x) > bounds.max_horizontal_offset or rocket.y > bounds.max_altitude:
        return Judgement(Outcome.FAILED, "out of bounds")
    if rocket.fuel <= 0.0:
        return Judgement(Outcome.FAILED, "out of fuel")
    return Judgement(Outcome.IN_FLIGHT)


def _judge_touchdown(rocket: RocketState, deck: DeckState, tips: np.ndarray) -> Judgement:
    vertical_speed = rocket.vy - deck.vy
    horizontal_speed = rocket.vx - deck.vx
    tilt = rocket.theta - deck.angle

    if not all(deck.contains(x) for x, _ in tips):
        reason = "leg off the deck edge"
    elif abs(vertical_speed) > MAX_VERTICAL_SPEED:
        reason = "too fast vertically"
    elif abs(horizontal_speed) > MAX_HORIZONTAL_SPEED:
        reason = "too fast sideways"
    elif abs(tilt) > MAX_TILT:
        reason = "tilted"
    elif abs(rocket.omega) > MAX_SPIN:
        reason = "spinning"
    else:
        return Judgement(Outcome.LANDED, "landed", touchdown_speed=abs(vertical_speed))
    return Judgement(Outcome.CRASHED, reason, touchdown_speed=abs(vertical_speed))
