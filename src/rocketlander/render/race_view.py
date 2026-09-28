"""Draws a race: several colour-coded rockets flying the same start state, with a legend."""

from __future__ import annotations

import numpy as np
import pygame

from rocketlander.envs.landing import Outcome
from rocketlander.race import Race
from rocketlander.render import scene
from rocketlander.render.hud import BAD, GOOD, TEXT, _panel
from rocketlander.render.renderer import Renderer


class RaceRenderer(Renderer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._wrecked: set[str] = set()  # labels of racers that exploded

    def reset(self) -> None:
        super().reset()
        self._wrecked = set()

    def draw_race(self, race: Race, level: str) -> None:
        dt = self._advance_time(race.time)
        deck = race.deck()
        rockets = {r.label: race.rocket(r) for r in race.racers}
        flying = [r for r in race.racers if not r.done] or race.racers
        points = np.array([[rockets[r.label].x, rockets[r.label].y] for r in flying])
        self.camera.follow(points, np.array([deck.x, deck.y]), dt)

        for r in race.racers:
            crashed = r.frame.judgement.outcome is Outcome.CRASHED
            if crashed and r.label not in self._wrecked:
                self.particles.explode(np.array([r.frame.rocket.x, r.frame.rocket.y]), n=120)
                self._wrecked.add(r.label)
            elif not r.done:
                self._emit_engine(r.frame, dt)
        corners = self.camera.to_world(np.array([[0, self.camera.height], [self.camera.width, 0]]))
        self.particles.emit_wind(corners[0], corners[1], race.racers[0].frame.wind_speed, dt)
        self.particles.update(dt)
        self._hide_particles_inside_hull(deck)

        s = self.surface
        s.blit(self.sky, (0, 0))
        scene.draw_sea(s, self.camera, race.time)
        self._draw_particles()
        scene.draw_ship(s, self.camera, deck)
        for r in race.racers:
            if len(r.trail) > 1:
                pixels = self.camera.to_screen(np.array(r.trail))
                pygame.draw.lines(s, r.color, False, pixels.tolist(), 1)
            if r.label not in self._wrecked:
                scene.draw_rocket(
                    s,
                    self.camera,
                    rockets[r.label],
                    r.frame.params,
                    r.frame.controls if not r.done else None,
                    color=r.color,
                )
        self._draw_legend(race, level)

    def _draw_legend(self, race: Race, level: str) -> None:
        font, s = self.hud.font, self.surface
        _panel(s, pygame.Rect(10, 10, 330, 34 + 24 * len(race.racers)))
        s.blit(font.render(f"{level}   t = {race.time:5.1f} s", True, TEXT), (20, 18))
        for i, r in enumerate(race.racers):
            y = 44 + 24 * i
            pygame.draw.rect(s, r.color, pygame.Rect(20, y + 2, 14, 14))
            outcome = r.frame.judgement.outcome
            color = {Outcome.LANDED: GOOD, Outcome.CRASHED: BAD, Outcome.FAILED: BAD}.get(
                outcome, TEXT
            )
            s.blit(font.render(r.label, True, TEXT), (42, y))
            s.blit(font.render(r.status, True, color), (190, y))
