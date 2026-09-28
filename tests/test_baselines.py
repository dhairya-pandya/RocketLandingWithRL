import numpy as np
import pytest

from rocketlander.agents.pid import PIDAgent
from rocketlander.agents.random_agent import RandomAgent
from rocketlander.evaluation import evaluate

SEEDS = list(range(50))


@pytest.mark.parametrize("level", ["L0", "L1"])
def test_pid_lands_reliably_on_easy_levels(level):
    summary = evaluate(PIDAgent(), level, SEEDS)
    assert summary.success_rate >= 0.9, summary.reasons


def test_random_agent_never_lands():
    summary = evaluate(RandomAgent(seed=0), "L0", SEEDS[:20])
    assert summary.success_rate == 0.0
    assert summary.episodes == 20 and sum(summary.reasons.values()) == 20


class HoverAgent:
    """Holds altitude and never tries to land."""

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        throttle = np.clip(0.55 - 0.5 * obs[3], 0.0, 1.0)  # obs[3] = vertical speed / 20
        return np.array([2.0 * throttle - 1.0, 0.0, 0.0], dtype=np.float32)


def test_task_return_ranks_landing_above_hovering():
    hover = evaluate(HoverAgent(), "L0", SEEDS[:20])
    pid = evaluate(PIDAgent(), "L0", SEEDS[:20])
    assert hover.success_rate == 0.0 and pid.success_rate == 1.0
    assert pid.mean_task_return > hover.mean_task_return


@pytest.mark.parametrize("level", ["L0", "L1", "L2", "L3", "L4"])
def test_hovering_never_outlasts_the_time_limit(level):
    """The tank runs dry (or wind blows the rocket away) before the unpenalized time limit."""
    summary = evaluate(HoverAgent(), level, SEEDS[:10])
    assert "time limit" not in summary.reasons
    assert set(summary.reasons) <= {"out of fuel", "out of bounds"}
    assert summary.mean_task_return < evaluate(PIDAgent(), level, SEEDS[:10]).mean_task_return
