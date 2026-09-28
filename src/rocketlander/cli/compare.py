"""Compare agents: rl-compare --viewer --agent pid checkpoints/ppo_L3.pt --level L3 --seed 4.

Every agent flies the same start state, ship motion and wind, drawn as colour-coded rockets.
Keys: Space pause, N next start, Esc quit. `--record race.gif` saves one race.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import pygame

from rocketlander.cli.common import base_parser, make_agent
from rocketlander.race import Race
from rocketlander.render.race_view import RaceRenderer

HOLD_STEPS = 60  # keep showing the result for 2 s


def label(name: str) -> str:
    return Path(name).stem if name.endswith(".pt") else name.upper()


def record_race(
    agents: list[str],
    level: str,
    seed: int,
    out: Path,
    every: int = 3,
    scale: float = 0.5,
    labels: list[str] | None = None,
) -> int:
    """Write a GIF of one race and return its number of frames."""
    labels = labels or [label(a) for a in agents]
    race = Race([(n, make_agent(a)) for n, a in zip(labels, agents, strict=True)], level, seed)
    renderer = RaceRenderer(window=False)
    size = (int(renderer.surface.get_width() * scale), int(renderer.surface.get_height() * scale))
    frames, step, hold = [], 0, 0
    while hold < HOLD_STEPS:
        if race.finished:
            hold += 1
        race.step()
        renderer.draw_race(race, level)
        if step % every == 0:
            small = pygame.transform.smoothscale(renderer.surface, size)
            frames.append(pygame.surfarray.array3d(small).transpose(1, 0, 2))
        step += 1
    renderer.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frames, duration=1000 * every / 30, loop=0)
    return len(frames)


def view(agents: list[str], level: str, seed: int, max_frames: int | None, headless: bool) -> None:
    renderer = RaceRenderer(window=not headless)
    loaded = [(label(a), make_agent(a)) for a in agents]
    frames = 0
    while True:
        race, hold, paused = Race(loaded, level, seed), 0, False
        while hold < HOLD_STEPS:
            if max_frames is not None and frames >= max_frames:
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
                    hold = HOLD_STEPS
            if not paused:
                hold += race.finished
                race.step()
                renderer.draw_race(race, level)
            renderer.show_frame(30)
        seed += 1


def main(argv: list[str] | None = None) -> None:
    parser = base_parser("Fly several agents from the same start state, side by side.")
    parser.add_argument("--agent", nargs="+", default=["pid", "random"], help="pid, random or .pt")
    parser.add_argument("--viewer", action="store_true", help="open the side-by-side viewer")
    parser.add_argument("--record", type=Path, default=None, help="write the race to a GIF")
    parser.add_argument("--headless", action="store_true", help="no window (for tests)")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)
    if args.record:
        n = record_race(args.agent, args.level, args.seed, args.record)
        print(f"Wrote {n} frames to {args.record}")
    if args.viewer:
        view(args.agent, args.level, args.seed, args.max_frames, args.headless)


if __name__ == "__main__":
    main()
