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
