"""Record one episode of an agent to a GIF."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import pygame

from rocketlander.cli.common import agent_argument, base_parser, make_agent, step_and_draw
from rocketlander.envs.rocket_env import RocketLanderEnv
from rocketlander.render.renderer import Renderer


def record(
    agent_name: str,
    level: str,
    seed: int,
    out: Path,
    every: int = 2,
    scale: float = 0.5,
    hold_seconds: float = 1.5,
    max_steps: int | None = None,
) -> int:
    """Write a GIF and return the number of frames in it."""
    env = RocketLanderEnv(level=level)
    renderer = Renderer(window=False)
    agent = make_agent(agent_name)
    obs, _ = env.reset(seed=seed)
    size = (int(renderer.surface.get_width() * scale), int(renderer.surface.get_height() * scale))
    frames, step, done = [], 0, False

    def grab() -> None:
        small = pygame.transform.smoothscale(renderer.surface, size)
        frames.append(pygame.surfarray.array3d(small).transpose(1, 0, 2))

    while not done and (max_steps is None or step < max_steps):
        obs, done = step_and_draw(env, renderer, agent, obs)
        if step % every == 0:
            grab()
        step += 1
    for _ in range(int(hold_seconds * 30 / every)):
        renderer.draw(env.frame())
        grab()
    renderer.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frames, duration=1000 * every / 30, loop=0)
    return len(frames)


def main(argv: list[str] | None = None) -> None:
    parser = base_parser("Record an agent's episode to a GIF.")
    agent_argument(parser)
    parser.add_argument("--out", type=Path, default=Path("videos/episode.gif"))
    parser.add_argument("--every", type=int, default=2, help="keep every n-th step")
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args(argv)
    n = record(args.agent, args.level, args.seed, args.out, args.every, args.scale)
    print(f"Wrote {n} frames to {args.out}")


if __name__ == "__main__":
    main()
