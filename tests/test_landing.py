import numpy as np
import pytest

from rocketlander.envs.landing import Bounds, Outcome, judge
from rocketlander.envs.physics import SEA_LEVEL, RocketParams, RocketState
from rocketlander.envs.ship import ShipMotion, deck_state

PARAMS = RocketParams()
DECK = deck_state(ShipMotion(), 0.0)  # flat deck, top surface at y = 0
TOUCHING = PARAMS.length / 2 + PARAMS.leg_drop - 0.01  # centre height where legs just touch


def rocket(**kw) -> RocketState:
    base = dict(x=0.0, y=TOUCHING, vx=0.0, vy=-1.0, theta=0.0, omega=0.0, fuel=1_000.0)
    base.update(kw)
    return RocketState(**base)


def verdict(state: RocketState):
    return judge(state, PARAMS, DECK, Bounds())


def test_gentle_upright_touchdown_lands():
    result = verdict(rocket())
    assert result.outcome is Outcome.LANDED
    assert result.touchdown_speed == pytest.approx(1.0)


def test_still_in_the_air_is_in_flight():
    assert verdict(rocket(y=50.0)).outcome is Outcome.IN_FLIGHT


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (dict(vy=-3.0), "too fast vertically"),
        (dict(vx=2.0), "too fast sideways"),
        (dict(theta=np.radians(12.0)), "tilted"),
        (dict(omega=0.5), "spinning"),
        (dict(x=13.0), "leg off the deck edge"),
    ],
)
def test_bad_touchdowns_crash_with_reason(overrides, reason):
    result = verdict(rocket(**overrides))
    assert result.outcome is Outcome.CRASHED
    assert result.reason == reason


def test_speed_limits_are_relative_to_a_moving_deck():
    moving = deck_state(ShipMotion(drift_speed=3.0), 0.0)
    result = judge(rocket(vx=3.0), PARAMS, moving, Bounds())
    assert result.outcome is Outcome.LANDED


def test_hitting_the_sea_crashes():
    result = verdict(rocket(x=100.0, y=SEA_LEVEL + TOUCHING))
    assert (result.outcome, result.reason) == (Outcome.CRASHED, "missed the ship")


def test_leaving_the_flight_box_or_running_out_of_fuel_fails():
    assert verdict(rocket(y=50.0, x=500.0)).reason == "out of bounds"
    assert verdict(rocket(y=5_000.0)).reason == "out of bounds"
    result = verdict(rocket(y=50.0, fuel=0.0))
    assert (result.outcome, result.reason) == (Outcome.FAILED, "out of fuel")
