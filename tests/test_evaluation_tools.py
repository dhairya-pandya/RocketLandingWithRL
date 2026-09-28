import numpy as np
import pytest

from rocketlander.agents.pid import PIDAgent
from rocketlander.cli import evaluate as evaluate_cli
from rocketlander.cli import robustness as robustness_cli
from rocketlander.envs.levels import get_level
from rocketlander.envs.physics import RocketParams
from rocketlander.envs.rocket_env import RocketLanderEnv
from rocketlander.evaluation import EVAL_SEEDS, evaluate
from rocketlander.robustness import perturbed, sweep, sweep_values
from rocketlander.stats import bootstrap_ci, iqm, wilson_interval


def test_wilson_interval_stays_inside_zero_one_and_contains_the_rate():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(100, 100)
    assert 0.95 < low < high == pytest.approx(1.0)
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high and high - low == pytest.approx(0.19, abs=0.01)


def test_iqm_ignores_the_outer_quartiles():
    assert iqm(np.array([0.0, 0.5, 0.5, 1.0])) == 0.5
    assert iqm(np.array([0.2, 0.4])) == pytest.approx(0.3)  # too few values to trim


def test_bootstrap_ci_brackets_the_mean_and_collapses_for_constants():
    values = np.random.default_rng(0).normal(1.0, 0.1, 200)
    low, high = bootstrap_ci(values, samples=2000)
    assert low < values.mean() < high
    assert bootstrap_ci(np.ones(5), samples=100) == (1.0, 1.0)


def test_perturbed_pins_one_parameter_and_renames():
    level = perturbed(get_level("L2"), "engine_lag", 0.3)
    assert level.engine_lag == (0.3, 0.3)
    assert level.name == "L2 engine_lag=0.3"
    assert level.wind_mean == get_level("L2").wind_mean


def test_env_accepts_a_level_config():
    env = RocketLanderEnv(level=perturbed(get_level("L1"), "mass_scale", 1.3))
    env.reset(seed=0)
    assert env.frame().params.dry_mass == pytest.approx(1.3 * RocketParams().dry_mass)
    assert env.level.name == "L1 mass_scale=1.3"


def test_every_sweep_is_a_level_field():
    for param in sweep_values():
        perturbed(get_level("L0"), param, 1.0)


def test_extreme_engine_lag_hurts_the_pid():
    results = dict(sweep(PIDAgent(), "L2", "engine_lag", EVAL_SEEDS[:10]))
    assert results[0.0].success_rate > results[0.4].success_rate


def test_unknown_sweep_raises():
    with pytest.raises(ValueError):
        sweep(PIDAgent(), "L0", "gravity", EVAL_SEEDS[:1])


def test_summary_carries_a_confidence_interval():
    summary = evaluate(PIDAgent(), "L0", EVAL_SEEDS[:5])
    low, high = summary.success_ci
    assert low <= summary.success_rate <= high


def test_cli_prints_intervals_and_iqm(capsys):
    evaluate_cli.main(["--agent", "pid", "pid", "--level", "L0", "--episodes", "3"])
    robustness_cli.main(
        ["--agent", "pid", "--level", "L0", "--param", "wind_mean", "--episodes", "2"]
    )
    out = capsys.readouterr().out
    assert "95% CI" in out and "IQM success over 2 agents" in out and "sweeping wind_mean" in out
