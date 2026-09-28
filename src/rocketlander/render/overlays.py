"""Debug overlays: force arrows, velocity vector, trajectory trail and live mini-plots."""

from __future__ import annotations

from collections import deque

import numpy as np
import pygame

from rocketlander.envs.rocket_env import Frame
from rocketlander.render.camera import Camera

THRUST = (255, 160, 50)
GRAVITY = (110, 160, 255)
DRAG = (120, 230, 230)
VELOCITY = (255, 255, 120)
TRAIL = (255, 255, 255)
PLOT_COLORS = [(255, 160, 50), (120, 230, 230), (230, 120, 230)]
HISTORY = 150
PLOT_W, PLOT_H = 220, 60


def _arrow(surface, start, vector_px, color, width=3) -> None:
    end = start + vector_px
    if np.linalg.norm(vector_px) < 2:
        return
    pygame.draw.line(surface, color, start, end, width)
    unit = vector_px / np.linalg.norm(vector_px)
    normal = np.array([-unit[1], unit[0]])
    pygame.draw.polygon(surface, color, [end + unit * 8, end + normal * 5, end - normal * 5])


def draw_forces(surface: pygame.Surface, camera: Camera, frame: Frame) -> None:
    if frame.forces is None:
        return
    center = camera.to_screen(np.array([frame.rocket.x, frame.rocket.y]))
    mass = frame.params.dry_mass + frame.rocket.fuel
    for force, color in (
        (frame.forces.thrust, THRUST),
        (frame.forces.gravity, GRAVITY),
        (frame.forces.drag, DRAG),
    ):
        accel = force / mass  # m/s^2; 6 px per m/s^2 keeps g about 60 px long
        _arrow(surface, center, np.array([accel[0], -accel[1]]) * 6.0, color)


def draw_velocity(surface: pygame.Surface, camera: Camera, frame: Frame) -> None:
    r, d = frame.rocket, frame.deck
    center = camera.to_screen(np.array([r.x, r.y]))
    _arrow(surface, center, np.array([r.vx - d.vx, -(r.vy - d.vy)]) * 4.0, VELOCITY, 2)


class Telemetry:
    """Rolling history of what the agent did and 'thought', for the mini-plots."""

    def __init__(self) -> None:
        self.trail: deque[tuple[float, float]] = deque(maxlen=2_000)
        self.actions: deque[np.ndarray] = deque(maxlen=HISTORY)
        self.rewards: deque[float] = deque(maxlen=HISTORY)
        self.values: deque[float] = deque(maxlen=HISTORY)

    def clear(self) -> None:
        for series in (self.trail, self.actions, self.rewards, self.values):
            series.clear()

    def record(self, action=None, reward=None, value=None) -> None:
        if action is not None:
            self.actions.append(np.clip(np.asarray(action, dtype=float), -1.0, 1.0))
        if reward is not None:
            self.rewards.append(float(reward))
        if value is not None:
            self.values.append(float(value))


def draw_trail(surface: pygame.Surface, camera: Camera, trail: deque) -> None:
    if len(trail) > 1:
        pygame.draw.lines(surface, TRAIL, False, camera.to_screen(np.array(trail)).tolist(), 1)


def _plot(surface, font, rect, title, series: list[np.ndarray], symmetric: bool) -> None:
    overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
    overlay.fill((10, 14, 30, 170))
    surface.blit(overlay, rect.topleft)
    surface.blit(font.render(title, True, (230, 235, 245)), (rect.x + 4, rect.y + 2))
    data = [s for s in series if len(s) > 1]
    if not data:
        return
    top = max(float(np.max(np.abs(s))) for s in data) or 1.0
    low, high = (-top, top) if symmetric else (min(float(np.min(s)) for s in data), top)
    span = (high - low) or 1.0
    for s, color in zip(data, PLOT_COLORS, strict=False):
        xs = rect.x + np.linspace(0, rect.width, HISTORY)[-len(s) :]
        ys = rect.bottom - 4 - (s - low) / span * (rect.height - 20)
        pygame.draw.lines(surface, color, False, np.column_stack([xs, ys]).tolist(), 2)


def draw_plots(surface: pygame.Surface, font: pygame.font.Font, telemetry: Telemetry) -> None:
    w, h = surface.get_size()
    x = w - PLOT_W - 10
    rects = [pygame.Rect(x, h - (PLOT_H + 8) * (i + 1), PLOT_W, PLOT_H) for i in range(3)]
    actions = np.array(telemetry.actions) if telemetry.actions else np.zeros((0, 3))
    _plot(
        surface,
        font,
        rects[2],
        "actions: throttle gimbal rcs",
        [actions[:, i] for i in range(3)],
        symmetric=True,
    )
    _plot(surface, font, rects[1], "reward per step", [np.array(telemetry.rewards)], False)
    _plot(surface, font, rects[0], "critic V(s)", [np.array(telemetry.values)], False)
