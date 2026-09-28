"""Fly the rocket yourself.

Keys: W/S throttle up/down, A/D gimbal, Q/E side thrusters, R restart, N new start,
F/V/T/P/H toggle forces/velocity/trail/plots/HUD, Esc quit.
"""

from __future__ import annotations

import numpy as np
import pygame

from rocketlander.cli.common import base_parser
from rocketlander.envs.rocket_env import RocketLanderEnv
from rocketlander.render.renderer import Renderer

THROTTLE_RATE = 0.8  # throttle change per second while W/S is held


class KeyboardPilot:
    """Turns held keys into an action; throttle is sticky, gimbal and RCS spring back."""

    def __init__(self) -> None:
        self.throttle = 0.0

    def action(self, held: dict[str, bool], dt: float) -> np.ndarray:
        change = held.get("w", False) - held.get("s", False)
        self.throttle = float(np.clip(self.throttle + change * THROTTLE_RATE * dt, 0.0, 1.0))
        gimbal = held.get("a", False) - held.get("d", False)  # A swings the nozzle left
        rcs = held.get("q", False) - held.get("e", False)  # Q spins counter-clockwise
        return np.array([2.0 * self.throttle - 1.0, gimbal, rcs], dtype=np.float32)


def held_keys() -> dict[str, bool]:
    pressed = pygame.key.get_pressed()
    return {name: bool(pressed[getattr(pygame, f"K_{name}")]) for name in "wsadqe"}


def main(argv: list[str] | None = None) -> None:
    parser = base_parser("Fly the rocket with the keyboard.")
    parser.add_argument("--headless", action="store_true", help="no window (for tests)")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)

    env = RocketLanderEnv(level=args.level)
    renderer = Renderer(window=not args.headless)
    pilot, seed = KeyboardPilot(), args.seed
    obs, _ = env.reset(seed=seed)
    done, frames = False, 0
    while args.max_frames is None or frames < args.max_frames:
        frames += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                return renderer.close()
            if event.type == pygame.KEYDOWN and event.unicode in ("r", "n"):
                seed += event.unicode == "n"
                obs, _ = env.reset(seed=seed)
                pilot, done = KeyboardPilot(), False
            elif event.type == pygame.KEYDOWN:
                renderer.toggle(event.unicode)
        if not done:
            action = pilot.action({} if args.headless else held_keys(), 1.0 / 30.0)
            obs, _, terminated, truncated, info = env.step(action)
            renderer.telemetry.record(action, info["task_reward"])
            done = terminated or truncated
        renderer.draw(env.frame())
        renderer.show_frame(30)
    renderer.close()


if __name__ == "__main__":
    main()
