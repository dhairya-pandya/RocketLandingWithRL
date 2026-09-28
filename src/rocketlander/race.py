"""Several agents fly the same start state side by side (the compare viewer's simulation)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from rocketlander.agents.base import Agent
from rocketlander.envs.landing import Outcome
from rocketlander.envs.physics import RocketState
from rocketlander.envs.rocket_env import Frame, RocketLanderEnv
from rocketlander.envs.ship import DeckState

COLORS = [
    (255, 170, 60),
    (90, 200, 255),
    (140, 230, 120),
    (240, 110, 200),
    (250, 240, 110),
    (180, 150, 255),
]


@dataclass
class Racer:
    label: str
    color: tuple[int, int, int]
    agent: Agent
    env: RocketLanderEnv
    obs: np.ndarray
    frame: Frame
    trail: list[tuple[float, float]] = field(default_factory=list)
    pose_on_deck: np.ndarray | None = None  # after landing: (x, y, theta) in the deck's frame

    @property
    def done(self) -> bool:
        return self.frame.judgement.done or self.frame.time_limit_reached

    @property
    def status(self) -> str:
        j = self.frame.judgement
        if j.outcome is Outcome.LANDED:
            return f"landed {j.touchdown_speed:.1f} m/s"
        if j.done:
            return j.reason
        if self.frame.time_limit_reached:
            return "time limit"
        return f"flying {self.frame.rocket.y - self.frame.deck.y:.0f} m"


class Race:
    """Every racer starts from the same seed, so ship, wind and start state are identical."""

    def __init__(self, agents: list[tuple[str, Agent]], level: str, seed: int) -> None:
        self.racers = []
        for i, (label, agent) in enumerate(agents):
            env = RocketLanderEnv(level=level)
            obs, _ = env.reset(seed=seed)
            color = COLORS[i % len(COLORS)]
            self.racers.append(Racer(label, color, agent, env, obs, env.frame()))
        self.time = 0.0

    @property
    def finished(self) -> bool:
        return all(r.done for r in self.racers)

    def step(self) -> None:
        """Advance every racer still in flight by one agent step."""
        for r in self.racers:
            if r.done:
                continue
            r.obs, _, _, _, _ = r.env.step(r.agent.act(r.obs, deterministic=True))
            r.frame = r.env.frame()
            r.trail.append((r.frame.rocket.x, r.frame.rocket.y))
            self.time = max(self.time, r.frame.time)
            if r.frame.judgement.outcome is Outcome.LANDED:
                r.pose_on_deck = _to_deck(r.frame.rocket, r.frame.deck)

    def deck(self) -> DeckState:
        return self.racers[0].env.deck_at(self.time)

    def rocket(self, racer: Racer) -> RocketState:
        """Where to draw a racer now: a landed rocket rides along with the moving deck."""
        if racer.pose_on_deck is None:
            return racer.frame.rocket
        x, y, theta = _from_deck(racer.pose_on_deck, self.deck())
        return replace(racer.frame.rocket, x=x, y=y, theta=theta)


def _rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _to_deck(rocket: RocketState, deck: DeckState) -> np.ndarray:
    local = _rotation(-deck.angle) @ np.array([rocket.x - deck.x, rocket.y - deck.y])
    return np.array([local[0], local[1], rocket.theta - deck.angle])


def _from_deck(pose: np.ndarray, deck: DeckState) -> tuple[float, float, float]:
    world = _rotation(deck.angle) @ pose[:2] + np.array([deck.x, deck.y])
    return float(world[0]), float(world[1]), float(pose[2] + deck.angle)
