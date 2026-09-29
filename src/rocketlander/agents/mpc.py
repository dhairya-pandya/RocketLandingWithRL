"""Model-predictive control: plan with a simulator, apply the first action, plan again.

Every `hold` steps the planner runs the cross-entropy method over `knots` actions, each held for
`hold` steps, in a *nominal* model built from the observation alone: nominal mass and thrust, no
wind, a deck that does not move. Like the PID, it is only as good as that model. A plan is scored
like the env's task reward (time and fuel costs per step, the terminal reward when the episode
would end), plus, where the horizon ends mid-air, a penalty for moving faster than the rocket could
still brake from (without it, a 1 s plan prefers hovering to descending).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rocketlander.common.config import load_config
from rocketlander.envs.landing import Bounds, judge
from rocketlander.envs.physics import RocketParams, RocketState, decode_action, step_rocket
from rocketlander.envs.rocket_env import (
    FUEL_COST,
    FUEL_SCALE,
    PHYSICS_DT,
    PHYSICS_STEPS_PER_ACTION,
    TIME_COST,
    terminal_reward,
)
from rocketlander.envs.ship import DeckState, ShipMotion


@dataclass
class MPCConfig:
    knots: int = 6  # actions per plan
    hold: int = 5  # agent steps each action is held (and steps between re-plans)
    candidates: int = 48
    elites: int = 6
    iterations: int = 3
    init_std: float = 0.5
    min_std: float = 0.05
    braking: float = 4.0  # m/s^2 the terminal penalty assumes the rocket can still brake with
    seed: int = 0


def nominal_model(obs: np.ndarray) -> tuple[RocketState, DeckState]:
    """The rocket relative to a deck fixed at the origin, decoded from the observation."""
    rocket = RocketState(
        x=float(obs[0]) * 100.0,
        y=float(obs[1]) * 100.0,
        vx=float(obs[2]) * 20.0,
        vy=float(obs[3]) * 20.0,
        theta=float(np.arctan2(obs[4], obs[5])),
        omega=float(obs[6]),
        fuel=float(obs[9]) * FUEL_SCALE,
        throttle=float(obs[10]),
    )
    deck = DeckState(
        0.0, 0.0, 0.0, 0.0, float(obs[7]) * 0.1, float(obs[8]) * 0.1, ShipMotion().deck_half_width
    )
    return rocket, deck


class MPCAgent:
    def __init__(self, config: MPCConfig | None = None) -> None:
        self.config = config or load_config(MPCConfig, "mpc")
        self.params, self.bounds = RocketParams(), Bounds()
        self.reset()

    def reset(self) -> None:
        """Forget the last episode's plan, so every episode is planned the same way."""
        self.rng = np.random.default_rng(self.config.seed)
        self.mean = np.zeros((self.config.knots, 3))
        self.plan, self.steps = self.mean, 0

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        c = self.config
        if self.steps % c.hold == 0:
            self.plan = self._search(*nominal_model(obs))
            self.mean = np.vstack([self.plan[1:], self.plan[-1:]])  # warm start, one knot on
        self.steps += 1
        return self.plan[0].astype(np.float32)

    def _search(self, rocket: RocketState, deck: DeckState) -> np.ndarray:
        c = self.config
        mean, std = self.mean, np.full(self.mean.shape, c.init_std)
        for _ in range(c.iterations):
            plans = np.clip(mean + std * self.rng.normal(size=(c.candidates, *mean.shape)), -1, 1)
            plans[0] = mean  # keep the current best guess in the running
            scores = np.array([self._score(rocket, deck, plan) for plan in plans])
            elite = plans[np.argsort(scores)[-c.elites :]]
            mean, std = elite.mean(axis=0), elite.std(axis=0) + c.min_std
        return mean

    def _score(self, rocket: RocketState, deck: DeckState, plan: np.ndarray) -> float:
        """Predicted task return of a plan in the nominal model."""
        total = 0.0
        for action in plan:
            controls = decode_action(action, self.params)
            for _ in range(self.config.hold):
                for _ in range(PHYSICS_STEPS_PER_ACTION):
                    rocket, _ = step_rocket(rocket, controls, self.params, 0.0, PHYSICS_DT)
                    judgement = judge(rocket, self.params, deck, self.bounds)
                    if judgement.done:
                        return total + terminal_reward(judgement)
                total -= TIME_COST + FUEL_COST * controls.throttle
        altitude = max(rocket.y - self.params.leg_clearance, 0.0)
        safe_speed = 1.0 + np.sqrt(2.0 * self.config.braking * altitude)
        speed = np.hypot(rocket.vx, rocket.vy)
        tilt = abs(rocket.theta - deck.angle)
        return total - (
            0.5 * np.hypot(rocket.x, rocket.y) + 2.0 * max(0.0, speed - safe_speed) + 30.0 * tilt
        )
