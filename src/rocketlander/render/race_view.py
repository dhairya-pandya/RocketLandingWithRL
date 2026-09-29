"""Draws a race: several colour-coded rockets flying the same start state, with a legend."""

from __future__ import annotations

import numpy as np
import pygame

from rocketlander.envs.landing import Outcome
from rocketlander.race import Race, Racer
from rocketlander.render import scene
from rocketlander.render.hud import BAD, GOOD, TEXT, panel
from rocketlander.render.renderer import Renderer

STATUS_COLORS = {Outcome.LANDED: GOOD, Outcome.CRASHED: BAD, Outcome.FAILED: BAD}
GREY = np.array([120, 125, 135])


def display_color(racer: Racer) -> tuple[int, int, int]:
    """A racer's colour, faded toward grey once it has stopped without landing."""
    stopped = racer.done and racer.frame.judgement.outcome is not Outcome.LANDED
    if not stopped:
        return racer.color
    return tuple(int(c) for c in (np.array(racer.color) + GREY) // 2)


class RaceRenderer(Renderer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._wrecked: set[int] = set()  # indices of racers that exploded

    def reset(self) -> None:
        super().reset()
        self._wrecked = set()

    def draw_race(self, race: Race, level: str) -> None:
        dt = self._advance_time(race.time)
        deck = race.deck()
        rockets = [race.rocket(r) for r in race.racers]
        flying = [i for i, r in enumerate(race.racers) if not r.done] or range(len(rockets))
        points = np.array([[rockets[i].x, rockets[i].y] for i in flying])
        self.camera.follow(points, np.array([deck.x, deck.y]), dt)

        for i, r in enumerate(race.racers):
            crashed = r.frame.judgement.outcome is Outcome.CRASHED
            if crashed and i not in self._wrecked:
                self.particles.explode(np.array([r.frame.rocket.x, r.frame.rocket.y]), n=120)
                self._wrecked.add(i)
            elif not r.done:
                self._emit_engine(r.frame, dt)
        corners = self.camera.to_world(np.array([[0, self.camera.height], [self.camera.width, 0]]))
        self.particles.emit_wind(
            corners[0], corners[1], race.racers[flying[0]].frame.wind_speed, dt
        )
        self.particles.update(dt)
        self._hide_particles_inside_hull(deck)

        s = self.surface
        s.blit(self.sky, (0, 0))
        scene.draw_sea(s, self.camera, race.time)
        self._draw_particles()
        scene.draw_ship(s, self.camera, deck)
        for i, r in enumerate(race.racers):
            if len(r.trail) > 1:
                pixels = self.camera.to_screen(np.array(r.trail))
                pygame.draw.lines(s, r.color, False, pixels.tolist(), 1)
            if i not in self._wrecked:
                controls = r.frame.controls if not r.done else None
                color = display_color(r)
                scene.draw_rocket(s, self.camera, rockets[i], r.frame.params, controls, color)
        self._draw_legend(race, level)

    def _draw_legend(self, race: Race, level: str) -> None:
        font, s = self.hud.font, self.surface
        panel(s, pygame.Rect(10, 10, 330, 34 + 24 * len(race.racers)))
        s.blit(font.render(f"{level}   t = {race.time:5.1f} s", True, TEXT), (20, 18))
        for i, r in enumerate(race.racers):
            y = 44 + 24 * i
            pygame.draw.rect(s, r.color, pygame.Rect(20, y + 2, 14, 14))
            color = STATUS_COLORS.get(r.frame.judgement.outcome, TEXT)
            if r.frame.time_limit_reached and not r.frame.judgement.done:
                color = BAD
            s.blit(font.render(r.label, True, TEXT), (42, y))
            s.blit(font.render(r.status, True, color), (190, y))
