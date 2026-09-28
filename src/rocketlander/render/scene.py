"""Draws the world: sky, sea, drone ship and rocket."""

from __future__ import annotations

import numpy as np
import pygame

from rocketlander.envs.physics import SEA_LEVEL, Controls, RocketParams, RocketState, body_to_world
from rocketlander.envs.ship import DeckState
from rocketlander.render.camera import Camera

SKY_TOP = (12, 20, 48)
SKY_BOTTOM = (70, 110, 160)
SEA = (18, 52, 88)
WAVE = (90, 150, 200)
HULL = (48, 52, 60)
DECK = (150, 150, 155)
TARGET = (240, 200, 40)
ROCKET = (235, 235, 240)
ROCKET_DARK = (150, 150, 160)
LEG = (175, 175, 185)
NOZZLE = (60, 60, 70)

HULL_DEPTH = 6.0  # metres from deck top down into the sea


def sky_gradient(width: int, height: int) -> pygame.Surface:
    surface = pygame.Surface((width, height))
    top, bottom = np.array(SKY_TOP), np.array(SKY_BOTTOM)
    for y in range(height):
        color = top + (bottom - top) * y / max(height - 1, 1)
        pygame.draw.line(surface, color.astype(int).tolist(), (0, y), (width, y))
    return surface


def draw_sea(surface: pygame.Surface, camera: Camera, t: float) -> None:
    left, right = camera.to_world(np.array([[0, 0], [camera.width, 0]]))[:, 0]
    xs = np.linspace(left, right, 120)
    ys = SEA_LEVEL + 0.6 * np.sin(0.15 * xs + 1.3 * t) + 0.3 * np.sin(0.41 * xs - 2.1 * t)
    surface_pts = camera.to_screen(np.column_stack([xs, ys]))
    bottom = [(camera.width, camera.height), (0, camera.height)]
    pygame.draw.polygon(surface, SEA, [tuple(p) for p in surface_pts] + bottom)
    pygame.draw.lines(surface, WAVE, False, [tuple(p) for p in surface_pts], 2)


def _deck_to_world(deck: DeckState, points: np.ndarray) -> np.ndarray:
    c, s = np.cos(deck.angle), np.sin(deck.angle)
    return points @ np.array([[c, -s], [s, c]]).T + np.array([deck.x, deck.y])


def draw_ship(surface: pygame.Surface, camera: Camera, deck: DeckState) -> None:
    w = deck.half_width
    hull = np.array([[-w, 0.0], [w, 0.0], [w * 0.85, -HULL_DEPTH], [-w * 0.85, -HULL_DEPTH]])
    pygame.draw.polygon(surface, HULL, camera.to_screen(_deck_to_world(deck, hull)).tolist())
    deck_line = camera.to_screen(_deck_to_world(deck, np.array([[-w, 0.0], [w, 0.0]])))
    width = max(2, int(0.6 * camera.scale))
    pygame.draw.line(surface, DECK, deck_line[0], deck_line[1], width)
    center = camera.to_screen(_deck_to_world(deck, np.array([[0.0, 0.0]])))[0]
    radius = max(3, int(4.0 * camera.scale))
    pygame.draw.ellipse(surface, TARGET, (center[0] - radius, center[1] - 2, 2 * radius, 4), 2)


def nozzle_geometry(rocket: RocketState, params: RocketParams, gimbal: float):
    """Nozzle exit point (world) and unit thrust direction."""
    direction = np.array([-np.sin(rocket.theta + gimbal), np.cos(rocket.theta + gimbal)])
    base = body_to_world(rocket, np.array([[0.0, -params.length / 2]]))[0]
    return base - direction * 0.8, direction  # shorter than the legs, so it clears the deck


def draw_rocket(
    surface: pygame.Surface,
    camera: Camera,
    rocket: RocketState,
    params: RocketParams,
    controls: Controls | None,
) -> None:
    half_l, half_w = params.length / 2, params.width / 2
    body = np.array([[-half_w, -half_l], [half_w, -half_l], [half_w, half_l], [-half_w, half_l]])
    nose = np.array([[-half_w, half_l], [half_w, half_l], [0.0, half_l + 2.5]])

    def to_px(points_body: np.ndarray) -> list:
        return camera.to_screen(body_to_world(rocket, points_body)).tolist()

    gimbal = controls.gimbal if controls else 0.0
    exit_point, direction = nozzle_geometry(rocket, params, gimbal)
    base = body_to_world(rocket, np.array([[0.0, -half_l]]))[0]
    side = np.array([direction[1], -direction[0]]) * 0.7
    nozzle = [base + side * 0.6, base - side * 0.6, exit_point - side, exit_point + side]
    pygame.draw.polygon(surface, NOZZLE, camera.to_screen(np.array(nozzle)).tolist())

    hips = to_px(np.array([[-half_w, -half_l + 3.0], [half_w, -half_l + 3.0]]))
    for hip, tip in zip(hips, to_px(params.leg_tips_body()), strict=True):
        pygame.draw.line(surface, LEG, hip, tip, max(2, int(0.4 * camera.scale)))

    pygame.draw.polygon(surface, ROCKET, to_px(body))
    pygame.draw.polygon(surface, ROCKET_DARK, to_px(nose))
    stripe = np.array(
        [[-half_w, half_l - 4], [half_w, half_l - 4], [half_w, half_l - 3], [-half_w, half_l - 3]]
    )
    pygame.draw.polygon(surface, ROCKET_DARK, to_px(stripe))
