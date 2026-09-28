import numpy as np
import pytest

from rocketlander.envs.ship import ShipMotion, deck_state
from rocketlander.envs.wind import WindModel, WindState, step_wind

MOVING = ShipMotion(
    sway_amplitude=8.0,
    sway_frequency=0.05,
    sway_phase=0.3,
    drift_speed=0.5,
    heave_amplitude=1.0,
    heave_frequency=0.1,
    heave_phase=1.0,
    roll_amplitude=0.07,
    roll_frequency=0.1,
    roll_phase=2.0,
)


def test_static_ship_stays_at_origin():
    deck = deck_state(ShipMotion(), t=12.3)
    assert (deck.x, deck.y, deck.vx, deck.vy, deck.angle) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_deck_velocity_matches_finite_difference():
    t, h = 7.0, 1e-5
    before, now, after = (deck_state(MOVING, t + k * h) for k in (-1, 0, 1))
    assert now.vx == pytest.approx((after.x - before.x) / (2 * h), rel=1e-6)
    assert now.vy == pytest.approx((after.y - before.y) / (2 * h), rel=1e-6)
    assert now.angular_velocity == pytest.approx((after.angle - before.angle) / (2 * h), rel=1e-6)


def test_tilted_deck_surface_and_extent():
    deck = deck_state(ShipMotion(roll_amplitude=0.1, roll_frequency=0.0, roll_phase=np.pi / 2), 0.0)
    assert deck.angle == pytest.approx(0.1)
    assert deck.surface_height(10.0) == pytest.approx(10.0 * np.tan(0.1))
    assert deck.contains(14.0) and not deck.contains(15.5)


def test_wind_without_gusts_decays_to_mean():
    model = WindModel(mean=5.0, reversion=0.5, volatility=0.0)
    state = WindState(speed=0.0)
    rng = np.random.default_rng(0)
    for _ in range(60 * 30):
        state = step_wind(state, model, 1.0 / 60.0, rng)
    assert state.speed == pytest.approx(5.0, abs=1e-3)


def test_wind_gusts_are_reproducible_with_the_same_seed():
    model = WindModel(mean=0.0, volatility=2.0)
    runs = []
    for _ in range(2):
        rng, state = np.random.default_rng(42), WindState(0.0)
        for _ in range(100):
            state = step_wind(state, model, 1.0 / 60.0, rng)
        runs.append(state.speed)
    assert runs[0] == runs[1] != 0.0
