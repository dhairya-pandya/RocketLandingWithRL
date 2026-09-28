"""Watch an agent fly. Keys: Space pause, N next episode, F/V/T/P/H overlays, Esc quit."""

from __future__ import annotations

import pygame

from rocketlander.cli.common import agent_argument, base_parser, make_agent, step_and_draw
from rocketlander.envs.rocket_env import RocketLanderEnv
from rocketlander.render.renderer import Renderer

HOLD_SECONDS = 2.0  # keep showing the outcome before the next episode


def main(argv: list[str] | None = None) -> None:
    parser = base_parser("Watch an agent land the rocket.")
    agent_argument(parser)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--headless", action="store_true", help="no window (for tests)")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)

    env = RocketLanderEnv(level=args.level)
    renderer = Renderer(window=not args.headless)
    agent = make_agent(args.agent)
    frames = 0
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        done, paused, hold = False, False, 0
        while hold < HOLD_SECONDS * 30:
            if args.max_frames is not None and frames >= args.max_frames:
                return renderer.close()
            frames += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    return renderer.close()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.type == pygame.KEYDOWN and event.unicode == "n":
                    hold = HOLD_SECONDS * 30
                elif event.type == pygame.KEYDOWN:
                    renderer.toggle(event.unicode)
            if paused:
                renderer.show_frame(30)
                continue
            if done:
                hold += 1
                renderer.draw(env.frame())
            else:
                obs, done = step_and_draw(env, renderer, agent, obs)
            renderer.show_frame(30)
    renderer.close()


if __name__ == "__main__":
    main()
