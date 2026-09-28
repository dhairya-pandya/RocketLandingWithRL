"""Composes scene, effects, overlays and HUD into one pygame surface."""

from __future__ import annotations

import os

import numpy as np
import pygame

from rocketlander.envs.landing import Outcome
from rocketlander.envs.physics import body_to_world
from rocketlander.envs.rocket_env import Frame
from rocketlander.render import overlays, scene
from rocketlander.render.camera import Camera
from rocketlander.render.effects import ParticleSystem
from rocketlander.render.hud import Hud

TOGGLE_KEYS = {"f": "forces", "v": "velocity", "t": "trail", "p": "plots", "h": "hud"}


class Renderer:
    def __init__(self, width: int = 1100, height: int = 720, window: bool = False) -> None:
        if not window:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self.window = pygame.display.set_mode((width, height)) if window else None
        if window:
            pygame.display.set_caption("RocketLandingWithRL")
        self.surface = pygame.Surface((width, height))
        self.sky = scene.sky_gradient(width, height)
        self.camera = Camera(width, height)
        self.particles = ParticleSystem()
        self.telemetry = overlays.Telemetry()
        self.hud = Hud()
        self.clock = pygame.time.Clock()
        self.show = {"forces": False, "velocity": False, "trail": True, "plots": True, "hud": True}
        self._last_time: float | None = None
        self._exploded = False

    def reset(self) -> None:
        self.camera.reset()
        self.particles.reset()
        self.telemetry.clear()
        self._last_time = None
        self._exploded = False

    def toggle(self, key: str) -> bool:
        """Flip the overlay bound to a key; returns True if the key was an overlay key."""
        name = TOGGLE_KEYS.get(key)
        if name is None:
            return False
        self.show[name] = not self.show[name]
        return True

    def draw(self, frame: Frame) -> None:
        dt = self._advance_time(frame.time)
        r, d = frame.rocket, frame.deck
        self.camera.follow(np.array([r.x, r.y]), np.array([d.x, d.y]), dt)
        self.telemetry.trail.append((r.x, r.y))
        self._emit_effects(frame, dt)
        self.particles.update(dt)

        s = self.surface
        s.blit(self.sky, (0, 0))
        scene.draw_sea(s, self.camera, frame.time)
        self._draw_particles()
        scene.draw_ship(s, self.camera, d)
        if self.show["trail"]:
            overlays.draw_trail(s, self.camera, self.telemetry.trail)
        if not self._exploded:
            scene.draw_rocket(s, self.camera, r, frame.params, frame.controls)
        if self.show["forces"]:
            overlays.draw_forces(s, self.camera, frame)
        if self.show["velocity"]:
            overlays.draw_velocity(s, self.camera, frame)
        if self.show["plots"]:
            overlays.draw_plots(s, self.hud.font, self.telemetry)
        if self.show["hud"]:
            self.hud.draw(s, frame)

    def show_frame(self, fps: int = 30) -> None:
        if self.window is not None:
            self.window.blit(self.surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(fps)

    def to_array(self) -> np.ndarray:
        """Current image as (height, width, 3) uint8."""
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2)).copy()

    def close(self) -> None:
        pygame.display.quit()

    def _advance_time(self, t: float) -> float:
        if self._last_time is None or t < self._last_time:
            self.reset()
            self._last_time = t
            return 1.0 / 30.0
        dt, self._last_time = t - self._last_time, t
        return max(dt, 1e-3)

    def _emit_effects(self, frame: Frame, dt: float) -> None:
        r, p, c = frame.rocket, frame.params, frame.controls
        if frame.judgement.outcome is Outcome.CRASHED and not self._exploded:
            self.particles.explode(np.array([r.x, r.y]))
            self._exploded = True
        if self._exploded or frame.judgement.done:
            return
        if r.throttle > 0.02 and r.fuel > 0.0:
            exit_point, direction = scene.nozzle_geometry(r, p, c.gimbal if c else 0.0)
            self.particles.emit_flame(exit_point, direction, r.throttle, dt)
        if c is not None and abs(c.rcs) > 0.1:
            side = -np.sign(c.rcs) * p.width / 2  # a CCW torque fires the right-side nose jet
            point = body_to_world(r, np.array([[side, p.length / 2 - 1.0]]))[0]
            outward = body_to_world(r, np.array([[side * 2, p.length / 2 - 1.0]]))[0] - point
            self.particles.emit_rcs(point, outward / np.linalg.norm(outward), dt)
        corners = self.camera.to_world(np.array([[0, self.camera.height], [self.camera.width, 0]]))
        self.particles.emit_wind(corners[0], corners[1], frame.wind_speed, dt)

    def _draw_particles(self) -> None:
        p = self.particles.p
        if len(p) == 0:
            return
        fade = self.particles.fade()
        pixels = self.camera.to_screen(p.pos)
        radii = np.maximum(1, (p.size * self.camera.scale * (0.4 + 0.6 * fade)).astype(int))
        sky = np.array(scene.SKY_BOTTOM)
        colors = (sky + (p.color - sky) * fade[:, None]).astype(int)
        tails = self.camera.to_screen(p.pos - p.vel * 0.08)
        for i, ((x, y), radius, color) in enumerate(zip(pixels, radii, colors, strict=True)):
            if p.streak[i]:
                pygame.draw.line(self.surface, color.tolist(), (x, y), tails[i], 1)
            else:
                pygame.draw.circle(self.surface, color.tolist(), (x, y), radius)
