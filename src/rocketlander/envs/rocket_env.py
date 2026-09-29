"""Gymnasium environment: land a rocket on a moving drone ship."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import numpy as np

from rocketlander.envs.landing import Bounds, Judgement, Outcome, judge
from rocketlander.envs.levels import EpisodeSetup, LevelConfig, get_level, sample_setup
from rocketlander.envs.physics import (
    Controls,
    Forces,
    RocketParams,
    RocketState,
    decode_action,
    step_rocket,
)
from rocketlander.envs.ship import DeckState, deck_state
from rocketlander.envs.wind import WindState, step_wind

PHYSICS_DT = 1.0 / 60.0
PHYSICS_STEPS_PER_ACTION = 2  # agent acts at 30 Hz
SHAPING_GAMMA = 1.0
FUEL_COST = 0.05
TIME_COST = 0.1  # per step, so hovering until the time limit is worse than trying to land
LANDING_REWARD = 100.0
SOFTNESS_BONUS = 50.0
CRASH_PENALTY = -100.0
DECK_CRASH_BASE = 20.0
DECK_CRASH_PER_MS = 8.0
OBS_SIZE = 11
ACT_SIZE = 3  # throttle, gimbal, RCS
FUEL_SCALE = 1_000.0  # observation reports fuel remaining in tonnes


def terminal_reward(judgement: Judgement) -> float:
    """Reward for how an episode ended (0 while still in flight)."""
    if judgement.outcome is Outcome.LANDED:
        softness = 1.0 - judgement.touchdown_speed / 2.0
        return LANDING_REWARD + SOFTNESS_BONUS * softness
    if judgement.outcome is Outcome.CRASHED and judgement.reason != "missed the ship":
        # Graded by impact speed: "almost landed" must beat "fell out of the sky".
        penalty = DECK_CRASH_BASE + DECK_CRASH_PER_MS * judgement.touchdown_speed
        return max(CRASH_PENALTY, -penalty)
    if judgement.outcome in (Outcome.CRASHED, Outcome.FAILED):
        return CRASH_PENALTY
    return 0.0


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
    params: RocketParams
    deck: DeckState
    wind_speed: float
    controls: Controls | None
    forces: Forces | None
    judgement: Judgement
    time: float
    level: str
    time_limit_reached: bool = False


class RocketLanderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        level: str = "L0",
        reward_mode: str = "shaped",
        shaping_gamma: float = SHAPING_GAMMA,
        render_mode: str | None = None,
    ) -> None:
        """shaping_gamma 1.0 (default): pure progress shaping, no hidden per-step bonus."""
        if reward_mode not in ("shaped", "sparse"):
            raise ValueError(f"reward_mode must be 'shaped' or 'sparse', got {reward_mode!r}")
        self._set_level(level)
        self.reward_mode = reward_mode
        self.shaping_gamma = shaping_gamma
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")
        self.render_mode = render_mode
        self._renderer = None
        self.bounds = Bounds()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self._state: EnvState | None = None
        self._last_forces: Forces | None = None
        self._last_controls: Controls | None = None
        self._last_judgement = Judgement(Outcome.IN_FLIGHT)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options and "level" in options:  # switch difficulty for this and later episodes
            self._set_level(options["level"])
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
            self._last_controls = None
            self._last_judgement = Judgement(Outcome.IN_FLIGHT)
        if self._renderer is not None:
            self._renderer.reset()
        if self.render_mode == "human":
            self.render()
        return self._observation(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        s = self._state
        controls = decode_action(action, s.setup.params)
        self._last_controls = controls

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
        terminal = terminal_reward(judgement)
        task_reward = terminal - FUEL_COST * controls.throttle - TIME_COST
        if self.reward_mode == "sparse":
            reward = terminal
        else:
            reward = task_reward + self._shaping(judgement)
        info = {
            "task_reward": task_reward,  # unshaped: use for evaluating and comparing agents
            "outcome": judgement.outcome.value,
            "reason": judgement.reason if judgement.done else ("time limit" if truncated else ""),
            "touchdown_speed": judgement.touchdown_speed,
            "fuel_used": s.setup.params.initial_fuel - s.rocket.fuel,
            "level": self.level.name,
        }
        if self.render_mode == "human":
            self.render()
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
        self._last_controls = None
        self._last_judgement = Judgement(Outcome.IN_FLIGHT)

    def frame(self) -> Frame:
        if self._state is None:
            raise RuntimeError("Call reset() before frame().")
        return Frame(
            rocket=self._state.rocket,
            params=self._state.setup.params,
            deck=self._deck(),
            wind_speed=self._state.wind.speed,
            controls=self._last_controls,
            forces=self._last_forces,
            judgement=self._last_judgement,
            time=self._state.time,
            level=self.level.name,
            time_limit_reached=self._state.steps >= self.max_steps,
        )

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from rocketlander.render.renderer import Renderer  # pygame is only needed to render

            self._renderer = Renderer(window=self.render_mode == "human")
        self._renderer.draw(self.frame())
        if self.render_mode == "human":
            self._renderer.show_frame(self.metadata["render_fps"])
            return None
        return self._renderer.to_array()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _set_level(self, level: str | LevelConfig) -> None:
        self.level: LevelConfig = level if isinstance(level, LevelConfig) else get_level(level)
        self.max_steps = int(self.level.max_seconds / (PHYSICS_DT * PHYSICS_STEPS_PER_ACTION))

    def deck_at(self, t: float) -> DeckState:
        """The deck at time t of the current episode (the ship's motion is a function of time)."""
        if self._state is None:
            raise RuntimeError("Call reset() before deck_at().")
        return deck_state(self._state.setup.ship, t)

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
            r.fuel / FUEL_SCALE,
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

    def _shaping(self, judgement: Judgement) -> float:
        """Progress reward gamma * phi(s') - phi(s). phi is kept at touchdown (not zeroed), so the
        shaping also grades how close, slow and upright the episode ended."""
        assert self._state is not None
        new_potential = self._potential(self._state.rocket, self._deck())
        shaping = self.shaping_gamma * new_potential - self._state.potential
        self._state.potential = new_potential
        return float(shaping)
