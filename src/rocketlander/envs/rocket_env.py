"""Gymnasium environment: land a rocket on a moving drone ship."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import numpy as np

from rocketlander.envs.landing import Bounds, Judgement, Outcome, judge
from rocketlander.envs.levels import EpisodeSetup, LevelConfig, get_level, sample_setup
from rocketlander.envs.physics import Forces, RocketState, decode_action, step_rocket
from rocketlander.envs.ship import DeckState, deck_state
from rocketlander.envs.wind import WindState, step_wind

PHYSICS_DT = 1.0 / 60.0
PHYSICS_STEPS_PER_ACTION = 2  # agent acts at 30 Hz
SHAPING_GAMMA = 0.99
FUEL_COST = 0.05
LANDING_REWARD = 100.0
SOFTNESS_BONUS = 50.0
CRASH_PENALTY = -100.0
OBS_SIZE = 11


@dataclass
class EnvState:
    """Everything needed to restore the env exactly, including the RNG."""

    setup: EpisodeSetup
    rocket: RocketState
    wind: WindState
    time: float
    steps: int
    potential: float
    rng_state: dict[str, Any]


@dataclass(frozen=True)
class Frame:
    """Read-only snapshot for the renderer."""

    rocket: RocketState
    deck: DeckState
    wind_speed: float
    forces: Forces | None
    judgement: Judgement
    time: float
    level: str


class RocketLanderEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, level: str = "L0", reward_mode: str = "shaped") -> None:
        if reward_mode not in ("shaped", "sparse"):
            raise ValueError(f"reward_mode must be 'shaped' or 'sparse', got {reward_mode!r}")
        self.level: LevelConfig = get_level(level)
        self.reward_mode = reward_mode
        self.bounds = Bounds()
        self.max_steps = int(self.level.max_seconds / (PHYSICS_DT * PHYSICS_STEPS_PER_ACTION))
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self._state: EnvState | None = None
        self._last_forces: Forces | None = None
        self._last_judgement = Judgement(Outcome.IN_FLIGHT)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options and "state" in options:
            self.set_state(options["state"])
        else:
            setup = sample_setup(self.level, self.np_random)
            deck = deck_state(setup.ship, 0.0)
            rocket = replace(setup.rocket, x=setup.rocket.x + deck.x, y=setup.rocket.y + deck.y)
            self._state = EnvState(
                setup=setup,
                rocket=rocket,
                wind=setup.wind,
                time=0.0,
                steps=0,
                potential=self._potential(rocket, deck),
                rng_state={},
            )
            self._last_forces = None
            self._last_judgement = Judgement(Outcome.IN_FLIGHT)
        return self._observation(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        s = self._state
        controls = decode_action(action, s.setup.params)

        judgement = Judgement(Outcome.IN_FLIGHT)
        for _ in range(PHYSICS_STEPS_PER_ACTION):
            s.wind = step_wind(s.wind, s.setup.wind_model, PHYSICS_DT, self.np_random)
            s.rocket, self._last_forces = step_rocket(
                s.rocket, controls, s.setup.params, s.wind.speed, PHYSICS_DT
            )
            s.time += PHYSICS_DT
            judgement = judge(s.rocket, s.setup.params, self._deck(), self.bounds)
            if judgement.done:
                break
        s.steps += 1
        self._last_judgement = judgement

        terminated = judgement.done
        truncated = not terminated and s.steps >= self.max_steps
        reward = self._reward(controls.throttle, judgement)
        info = {
            "outcome": judgement.outcome.value,
            "reason": judgement.reason if judgement.done else ("time limit" if truncated else ""),
            "touchdown_speed": judgement.touchdown_speed,
            "fuel_used": s.setup.params.initial_fuel - s.rocket.fuel,
        }
        return self._observation(), reward, terminated, truncated, info

    def get_state(self) -> EnvState:
        if self._state is None:
            raise RuntimeError("Call reset() before get_state().")
        snapshot = copy.deepcopy(self._state)
        snapshot.rng_state = copy.deepcopy(self.np_random.bit_generator.state)
        return snapshot

    def set_state(self, state: EnvState) -> None:
        self._state = copy.deepcopy(state)
        if state.rng_state:
            self.np_random.bit_generator.state = copy.deepcopy(state.rng_state)
        self._last_forces = None
        self._last_judgement = Judgement(Outcome.IN_FLIGHT)

    def frame(self) -> Frame:
        if self._state is None:
            raise RuntimeError("Call reset() before frame().")
        return Frame(
            rocket=self._state.rocket,
            deck=self._deck(),
            wind_speed=self._state.wind.speed,
            forces=self._last_forces,
            judgement=self._last_judgement,
            time=self._state.time,
            level=self.level.name,
        )

    def _deck(self) -> DeckState:
        assert self._state is not None
        return deck_state(self._state.setup.ship, self._state.time)

    def _observation(self) -> np.ndarray:
        assert self._state is not None
        r, d = self._state.rocket, self._deck()
        obs = [
            (r.x - d.x) / 100.0,
            (r.y - d.y) / 100.0,
            (r.vx - d.vx) / 20.0,
            (r.vy - d.vy) / 20.0,
            np.sin(r.theta),
            np.cos(r.theta),
            r.omega,
            d.angle / 0.1,
            d.angular_velocity / 0.1,
            r.fuel / self._state.setup.params.initial_fuel,
            r.throttle,
        ]
        return np.asarray(obs, dtype=np.float32)

    @staticmethod
    def _potential(rocket: RocketState, deck: DeckState) -> float:
        """Higher when close to the deck, slow relative to it, and upright."""
        distance = np.hypot(rocket.x - deck.x, rocket.y - deck.y)
        speed = np.hypot(rocket.vx - deck.vx, rocket.vy - deck.vy)
        tilt = abs(rocket.theta - deck.angle)
        return -(0.5 * distance + 2.0 * speed + 30.0 * tilt)

    def _reward(self, throttle: float, judgement: Judgement) -> float:
        assert self._state is not None
        terminal = 0.0
        if judgement.outcome is Outcome.LANDED:
            softness = 1.0 - judgement.touchdown_speed / 2.0
            terminal = LANDING_REWARD + SOFTNESS_BONUS * softness
        elif judgement.outcome in (Outcome.CRASHED, Outcome.FAILED):
            terminal = CRASH_PENALTY

        if self.reward_mode == "sparse":
            return float(terminal)

        # Potential-based shaping with phi(terminal) = 0 keeps the optimal policy unchanged.
        new_potential = 0.0 if judgement.done else self._potential(self._state.rocket, self._deck())
        shaping = SHAPING_GAMMA * new_potential - self._state.potential
        self._state.potential = new_potential
        return float(shaping - FUEL_COST * throttle + terminal)
