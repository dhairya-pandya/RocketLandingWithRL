"""Confidence intervals and robust aggregates for reporting results honestly."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a success rate; well-behaved near 0% and 100%."""
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def iqm(values: np.ndarray) -> float:
    """Interquartile mean: the mean of the middle 50% (robust to a few outlier runs)."""
    v = np.sort(np.asarray(values, dtype=float))
    cut = int(np.floor(len(v) * 0.25))
    return float(v[cut : len(v) - cut].mean())


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile-bootstrap interval of `statistic` over resampled values."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    stats = [statistic(rng.choice(v, size=len(v), replace=True)) for _ in range(samples)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
