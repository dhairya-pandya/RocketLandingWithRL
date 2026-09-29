"""Heads-up display: flight readouts, control gauges, wind and the outcome banner."""

from __future__ import annotations

import numpy as np
import pygame

from rocketlander.envs.landing import (
    MAX_HORIZONTAL_SPEED,
    MAX_SPIN,
    MAX_TILT,
    MAX_VERTICAL_SPEED,
    Outcome,
)
from rocketlander.envs.rocket_env import Frame

TEXT = (230, 235, 245)
GOOD = (90, 220, 120)
BAD = (240, 90, 80)
WARN = (250, 180, 60)
PANEL = (10, 14, 30, 170)
BAR_BG = (60, 66, 90)


def panel(surface: pygame.Surface, rect: pygame.Rect) -> None:
    overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
    overlay.fill(PANEL)
    surface.blit(overlay, rect.topleft)


class Hud:
    def __init__(self) -> None:
        self.font = pygame.font.Font(None, 22)
        self.big = pygame.font.Font(None, 56)

    def _text(self, surface, text, pos, color=TEXT, font=None) -> None:
        surface.blit((font or self.font).render(text, True, color), pos)

    def _bar(self, surface, x, y, label, fraction, color, centered=False) -> None:
        self._text(surface, label, (x, y))
        rect = pygame.Rect(x + 92, y + 2, 118, 12)
        pygame.draw.rect(surface, BAR_BG, rect)
        if centered:  # -1..1 around the middle
            mid = rect.centerx
            end = mid + int(np.clip(fraction, -1, 1) * rect.width / 2)
            pygame.draw.rect(surface, color, pygame.Rect(min(mid, end), rect.y, abs(end - mid), 12))
            pygame.draw.line(surface, TEXT, (mid, rect.y - 2), (mid, rect.bottom + 1))
        else:
            width = int(np.clip(fraction, 0, 1) * rect.width)
            pygame.draw.rect(surface, color, pygame.Rect(rect.x, rect.y, width, 12))

    def draw(self, surface: pygame.Surface, frame: Frame) -> None:
        r, d, p = frame.rocket, frame.deck, frame.params
        altitude = r.y - d.y - p.length / 2 - p.leg_drop
        v_rel, h_rel = r.vy - d.vy, r.vx - d.vx
        tilt = r.theta - d.angle

        panel(surface, pygame.Rect(10, 10, 230, 230))
        rows = [
            ("ALTITUDE", f"{altitude:7.1f} m", None),
            ("VERT SPEED", f"{v_rel:7.1f} m/s", abs(v_rel) <= MAX_VERTICAL_SPEED),
            ("SIDE SPEED", f"{h_rel:7.1f} m/s", abs(h_rel) <= MAX_HORIZONTAL_SPEED),
            ("TILT", f"{np.degrees(tilt):7.1f} deg", abs(tilt) <= MAX_TILT),
            ("SPIN", f"{np.degrees(r.omega):7.1f} deg/s", abs(r.omega) <= MAX_SPIN),
        ]
        for i, (label, value, ok) in enumerate(rows):
            color = TEXT if ok is None else (GOOD if ok else BAD)
            self._text(surface, label, (20, 20 + 22 * i))
            self._text(surface, value, (120, 20 + 22 * i), color)

        c = frame.controls
        fuel = r.fuel / p.initial_fuel
        self._bar(surface, 20, 140, "FUEL", fuel, GOOD if fuel > 0.2 else BAD)
        self._bar(surface, 20, 162, "THROTTLE", r.throttle, WARN)
        self._bar(surface, 20, 184, "GIMBAL", (c.gimbal / p.max_gimbal) if c else 0.0, TEXT, True)
        self._bar(surface, 20, 206, "RCS", c.rcs if c else 0.0, TEXT, True)

        self._text(
            surface, f"{frame.level}   t = {frame.time:5.1f} s", (surface.get_width() // 2 - 60, 14)
        )
        self._draw_wind(surface, frame.wind_speed)
        self._draw_banner(surface, frame)

    def _draw_wind(self, surface: pygame.Surface, wind: float) -> None:
        x = surface.get_width() - 170
        panel(surface, pygame.Rect(x, 10, 160, 50))
        self._text(surface, f"WIND {wind:+5.1f} m/s", (x + 10, 16))
        mid, y = x + 80, 44
        end = mid + int(np.clip(wind / 10.0, -1, 1) * 60)
        pygame.draw.line(surface, TEXT, (mid, y), (end, y), 3)
        if end != mid:
            tip = np.sign(end - mid) * 8
            pygame.draw.polygon(surface, TEXT, [(end + tip, y), (end, y - 6), (end, y + 6)])

    def _draw_banner(self, surface: pygame.Surface, frame: Frame) -> None:
        j = frame.judgement
        if j.done:
            color = {Outcome.LANDED: GOOD, Outcome.CRASHED: BAD}.get(j.outcome, WARN)
            title = j.outcome.value.upper()
            landed = j.outcome is Outcome.LANDED
            detail = f"touchdown {j.touchdown_speed:.2f} m/s" if landed else j.reason
        elif frame.time_limit_reached:
            color, title, detail = WARN, "TIME LIMIT", "episode truncated"
        else:
            return
        w, h = surface.get_size()
        top = 44
        panel(surface, pygame.Rect(w // 2 - 220, top, 440, 100))
        text = self.big.render(title, True, color)
        surface.blit(text, (w // 2 - text.get_width() // 2, top + 12))
        sub = self.font.render(detail, True, TEXT)
        surface.blit(sub, (w // 2 - sub.get_width() // 2, top + 66))
