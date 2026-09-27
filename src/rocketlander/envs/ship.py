"""Drone ship motion: sway, drift, heave and roll as closed-form functions of time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class ShipMotion:
    sway_amplitude: float = 0.0
    sway_frequency: float = 0.0
    sway_phase: float = 0.0
    drift_speed: float = 0.0
    heave_amplitude: float = 0.0
    heave_frequency: float = 0.0
    heave_phase: float = 0.0
    roll_amplitude: float = 0.0
    roll_frequency: float = 0.0
    roll_phase: float = 0.0
    deck_half_width: float = 15.0


@dataclass(frozen=True)
class DeckState:
    x: float
    y: float  # deck top at the centre
    vx: float
    vy: float
    angle: float
    angular_velocity: float
    half_width: float

    def surface_height(self, x: float) -> float:
        return self.y + (x - self.x) * np.tan(self.angle)

    def contains(self, x: float) -> bool:
        return abs(x - self.x) <= self.half_width * np.cos(self.angle)


def _wave(amplitude: float, frequency: float, phase: float, t: float) -> tuple[float, float]:
    """Value and time derivative of amplitude * sin(2 pi f t + phase)."""
    arg = TWO_PI * frequency * t + phase
    return amplitude * np.sin(arg), amplitude * TWO_PI * frequency * np.cos(arg)


def deck_state(motion: ShipMotion, t: float) -> DeckState:
    sway, sway_rate = _wave(motion.sway_amplitude, motion.sway_frequency, motion.sway_phase, t)
    heave, heave_rate = _wave(motion.heave_amplitude, motion.heave_frequency, motion.heave_phase, t)
    roll, roll_rate = _wave(motion.roll_amplitude, motion.roll_frequency, motion.roll_phase, t)
    return DeckState(
        x=sway + motion.drift_speed * t,
        y=heave,
        vx=sway_rate + motion.drift_speed,
        vy=heave_rate,
        angle=roll,
        angular_velocity=roll_rate,
        half_width=motion.deck_half_width,
    )
