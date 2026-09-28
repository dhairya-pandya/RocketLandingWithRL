import numpy as np
import pytest

from rocketlander.envs.levels import available_levels, get_level, level_from_dict, sample_setup


def test_all_five_levels_ship_as_yaml():
    assert available_levels() == ["L0", "L1", "L2", "L3", "L4"]


@pytest.mark.parametrize("name", ["L0", "L1", "L2", "L3", "L4"])
def test_samples_stay_inside_level_ranges(name):
    level = get_level(name)
    assert level.name == name
    rng = np.random.default_rng(0)
    for _ in range(50):
        setup = sample_setup(level, rng)
        assert level.start_altitude[0] <= setup.rocket.y <= level.start_altitude[1]
        assert level.start_offset_x[0] <= setup.rocket.x <= level.start_offset_x[1]
        assert level.wind_mean[0] <= setup.wind_model.mean <= level.wind_mean[1]
        assert setup.rocket.fuel == setup.params.initial_fuel
        assert level.fuel[0] <= setup.params.initial_fuel <= level.fuel[1]


def test_l0_is_a_static_calm_pad():
    setup = sample_setup(get_level("L0"), np.random.default_rng(0))
    ship = setup.ship
    motion = (ship.sway_amplitude, ship.drift_speed, ship.heave_amplitude, ship.roll_amplitude)
    assert motion == (0.0, 0.0, 0.0, 0.0)
    assert setup.wind_model.mean == setup.wind_model.volatility == 0.0
    assert setup.params.engine_lag == 0.0


def test_degree_fields_are_converted_to_radians():
    level = get_level("L2")
    assert level.roll_amplitude == pytest.approx((0.0, np.radians(4.0)))
    assert level.start_theta == pytest.approx((np.radians(-6.0), np.radians(6.0)))


def test_unknown_level_or_field_raises():
    with pytest.raises(ValueError, match="Unknown level 'L9'"):
        get_level("L9")
    raw = {"name": "X", "description": "", "start_offset_x": [0, 1], "typo_field": [0, 1]}
    with pytest.raises(ValueError, match="Unknown level field 'typo_field'"):
        level_from_dict(raw)
