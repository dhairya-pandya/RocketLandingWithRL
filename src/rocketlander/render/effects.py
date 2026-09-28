"""World-space particles: engine flame, RCS puffs, wind streaks and explosions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLAME = (255, 170, 60)
SMOKE = (200, 200, 205)
SPARK = (255, 90, 40)
WIND = (235, 245, 255)


@dataclass
class Particles:
    pos: np.ndarray  # (n, 2) metres
    vel: np.ndarray  # (n, 2) m/s
    life: np.ndarray  # (n,) seconds left
    max_life: np.ndarray  # (n,)
    size: np.ndarray  # (n,) metres
    color: np.ndarray  # (n, 3)
    streak: np.ndarray  # (n,) bool: draw as a line along the velocity

    @classmethod
    def empty(cls) -> Particles:
        shapes = [(0, 2), (0, 2), (0,), (0,), (0,), (0, 3), (0,)]
        return cls(*(np.zeros(shape) for shape in shapes))

    def __len__(self) -> int:
        return len(self.life)


class ParticleSystem:
    def __init__(self, seed: int = 0, max_particles: int = 3_000) -> None:
        self.rng = np.random.default_rng(seed)
        self.max_particles = max_particles
        self.p = Particles.empty()

    def reset(self) -> None:
        self.p = Particles.empty()

    def _add(self, pos, vel, life, size, color, streak: bool = False) -> None:
        n = len(life)
        colors = np.tile(color, (n, 1)).astype(float)
        new = Particles(pos, vel, life, life.copy(), size, colors, np.full(n, streak))
        fields = ("pos", "vel", "life", "max_life", "size", "color", "streak")
        merged = [np.concatenate([getattr(self.p, f), getattr(new, f)]) for f in fields]
        self.p = Particles(*(m[-self.max_particles :] for m in merged))

    def emit_flame(self, nozzle: np.ndarray, direction: np.ndarray, throttle: float, dt: float):
        """Exhaust leaves the nozzle opposite to the thrust direction."""
        n = int(throttle * 900 * dt)
        if n == 0:
            return
        spread = self.rng.normal(0.0, 0.12, n)
        angle = np.arctan2(-direction[1], -direction[0]) + spread
        speed = self.rng.uniform(25.0, 45.0, n) * (0.5 + throttle)
        vel = np.stack([np.cos(angle), np.sin(angle)], axis=1) * speed[:, None]
        pos = nozzle + self.rng.normal(0.0, 0.3, (n, 2))
        life = self.rng.uniform(0.15, 0.35, n)
        self._add(pos, vel, life, self.rng.uniform(0.6, 1.4, n), FLAME)

    def emit_rcs(self, point: np.ndarray, outward: np.ndarray, dt: float) -> None:
        n = max(1, int(120 * dt))
        vel = outward * self.rng.uniform(8.0, 14.0, (n, 1)) + self.rng.normal(0.0, 1.5, (n, 2))
        self._add(
            np.tile(point, (n, 1)), vel, self.rng.uniform(0.2, 0.4, n), np.full(n, 0.5), SMOKE
        )

    def emit_wind(self, view_min: np.ndarray, view_max: np.ndarray, wind: float, dt: float):
        n = self.rng.poisson(abs(wind) * 6 * dt)
        if n == 0:
            return
        pos = self.rng.uniform(view_min, view_max, (n, 2))
        vel = np.column_stack([np.full(n, wind * 6.0), np.zeros(n)])
        self._add(pos, vel, np.full(n, 0.6), np.full(n, 0.3), WIND, streak=True)

    def explode(self, center: np.ndarray, n: int = 250) -> None:
        angle = self.rng.uniform(0.0, 2 * np.pi, n)
        speed = self.rng.uniform(5.0, 40.0, n)
        vel = np.stack([np.cos(angle), np.abs(np.sin(angle))], axis=1) * speed[:, None]
        life = self.rng.uniform(0.5, 1.5, n)
        self._add(center + self.rng.normal(0.0, 1.0, (n, 2)), vel, life, np.full(n, 1.2), SPARK)

    def update(self, dt: float, gravity: float = 9.81) -> None:
        p = self.p
        p.vel[:, 1] -= gravity * 0.3 * dt  # hot gas rises less than debris falls
        p.pos += p.vel * dt
        p.life -= dt
        alive = p.life > 0.0
        self.p = Particles(*(getattr(p, f)[alive] for f in Particles.__dataclass_fields__))

    def fade(self) -> np.ndarray:
        """Remaining life as a fraction in [0, 1] for each particle."""
        return np.clip(self.p.life / np.maximum(self.p.max_life, 1e-6), 0.0, 1.0)
