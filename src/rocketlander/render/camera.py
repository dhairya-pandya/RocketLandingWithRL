"""Camera that keeps both the rocket and the ship in view, zooming in smoothly near touchdown."""

from __future__ import annotations

import numpy as np

MIN_SCALE = 0.4  # pixels per metre when far away
MAX_SCALE = 9.0  # pixels per metre at touchdown
MARGIN = 1.6  # keep this much room around rocket and ship
SMOOTHING = 4.0  # 1/s, how fast the camera catches up


class Camera:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.center = np.zeros(2)
        self.scale = MAX_SCALE
        self._initialised = False

    def target(self, rocket_xy: np.ndarray, deck_xy: np.ndarray) -> tuple[np.ndarray, float]:
        """Centre and scale that frame both the rocket and the deck."""
        center = (rocket_xy + deck_xy) / 2.0
        span = np.abs(rocket_xy - deck_xy) * MARGIN + 40.0  # metres; 40 keeps the ship visible
        scale = min(self.width / span[0], self.height / span[1])
        return center, float(np.clip(scale, MIN_SCALE, MAX_SCALE))

    def follow(self, rocket_xy: np.ndarray, deck_xy: np.ndarray, dt: float) -> None:
        center, scale = self.target(np.asarray(rocket_xy), np.asarray(deck_xy))
        if not self._initialised:
            self.center, self.scale, self._initialised = center, scale, True
            return
        blend = 1.0 - np.exp(-SMOOTHING * dt)
        self.center = self.center + blend * (center - self.center)
        self.scale = self.scale + blend * (scale - self.scale)

    def reset(self) -> None:
        self._initialised = False

    def to_screen(self, points: np.ndarray) -> np.ndarray:
        """World metres (..., 2) to screen pixels (..., 2); screen y points down."""
        p = (np.asarray(points, dtype=np.float64) - self.center) * self.scale
        return np.stack([self.width / 2 + p[..., 0], self.height / 2 - p[..., 1]], axis=-1)

    def to_world(self, pixels: np.ndarray) -> np.ndarray:
        px = np.asarray(pixels, dtype=np.float64)
        x = (px[..., 0] - self.width / 2) / self.scale
        y = (self.height / 2 - px[..., 1]) / self.scale
        return np.stack([x, y], axis=-1) + self.center
