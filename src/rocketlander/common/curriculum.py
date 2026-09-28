"""Which level each training episode uses: one fixed level, a uniform mix, or a curriculum."""

from __future__ import annotations

from collections import deque

import numpy as np

from rocketlander.envs.levels import available_levels

MODES = ("mix", "curriculum")


class LevelSchedule:
    """Mix: every episode draws a level uniformly. Curriculum: start with the first level and
    unlock the next once the newest unlocked level reaches `threshold` success over `window`
    episodes; episodes draw uniformly from the unlocked levels so earlier skills are kept."""

    def __init__(
        self, mode: str, levels: list[str], seed: int, threshold: float = 0.8, window: int = 100
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"Unknown schedule {mode!r}; choose from {MODES}")
        unknown = set(levels) - set(available_levels())
        if unknown or not levels:
            raise ValueError(f"Unknown levels {sorted(unknown)}")
        self.mode, self.levels = mode, list(levels)
        self.threshold, self.window = threshold, window
        self.unlocked = 1 if mode == "curriculum" else len(levels)
        self.results = {level: deque(maxlen=window) for level in self.levels}
        self.rng = np.random.default_rng(seed)

    def sample(self) -> str:
        return self.levels[int(self.rng.integers(self.unlocked))]

    def record(self, level: str, landed: bool) -> None:
        self.results[level].append(landed)
        newest = self.levels[self.unlocked - 1]
        full = len(self.results[newest]) == self.window
        if (
            self.unlocked < len(self.levels)
            and full
            and np.mean(self.results[newest]) >= self.threshold
        ):
            self.unlocked += 1

    def summary(self) -> dict[str, float]:
        row = {"curriculum/unlocked": float(self.unlocked)}
        for level, results in self.results.items():
            if results:
                row[f"curriculum/success_{level}"] = float(np.mean(results))
        return row


def parse_level(spec: str, seed: int) -> tuple[str, LevelSchedule | None]:
    """'L2' -> ('L2', None); 'mix:L3,L4' or 'curriculum:L0,...,L4' -> (hardest level, schedule).

    The returned level is the one to evaluate on: the single level, or the last in the list."""
    if ":" not in spec:
        if spec not in available_levels():
            raise ValueError(f"Unknown level {spec!r}. Choose from {available_levels()}.")
        return spec, None
    mode, _, names = spec.partition(":")
    levels = [n.strip() for n in names.split(",") if n.strip()]
    schedule = LevelSchedule(mode, levels, seed)  # validates the names before we index them
    return levels[-1], schedule
