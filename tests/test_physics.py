from dataclasses import replace

import numpy as np
import pytest

from rocketlander.envs.physics import (
    GRAVITY,
    Controls,
    RocketParams,
    RocketState,
    body_to_world,
    compute_forces,
    decode_action,
    step_rocket,
)

DT = 1.0 / 60.0
NO_DRAG = RocketParams(drag_x=0.0, drag_y=0.0)
ENGINE_OFF = Controls(throttle=0.0, gimbal=0.0, rcs=0.0)


def upright(**overrides) -> RocketState:
    state = RocketState(x=0.0, y=100.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0, fuel=5_000.0)
    return replace(state, **overrides)


def test_free_fall_matches_semi_implicit_euler_solution():
    state = upright()
    steps = 120
    for _ in range(steps):
        state, _ = step_rocket(state, ENGINE_OFF, NO_DRAG, wind_speed=0.0, dt=DT)
    # Semi-implicit Euler: v_n = -g n dt and y_n = y_0 - g dt^2 n (n + 1) / 2.
    assert state.vy == pytest.approx(-GRAVITY * steps * DT)
    assert state.y == pytest.approx(100.0 - GRAVITY * DT**2 * steps * (steps + 1) / 2)
    assert state.x == 0.0 and state.theta == 0.0


def test_zero_gimbal_gives_zero_torque_and_thrust_along_body_axis():
    state = upright(theta=0.3, throttle=1.0)
    forces = compute_forces(state, Controls(1.0, 0.0, 0.0), NO_DRAG, wind_speed=0.0)
    assert forces.torque == pytest.approx(0.0)
    body_axis = np.array([-np.sin(0.3), np.cos(0.3)])
    assert np.allclose(forces.thrust / np.linalg.norm(forces.thrust), body_axis)


def test_positive_gimbal_gives_clockwise_torque():
    params = RocketParams()
    state = upright(throttle=1.0)
    gimbal = np.radians(10.0)
    forces = compute_forces(state, Controls(1.0, gimbal, 0.0), params, wind_speed=0.0)
    expected = -params.max_thrust * np.sin(gimbal) * params.length / 2.0
    assert forces.torque == pytest.approx(expected)
    assert forces.torque < 0.0


def test_hover_throttle_holds_altitude():
    params = NO_DRAG
    mass = params.dry_mass + 5_000.0
    hover = mass * GRAVITY / params.max_thrust
    state, _ = step_rocket(upright(), Controls(hover, 0.0, 0.0), params, 0.0, DT)
    assert state.vy == pytest.approx(0.0, abs=1e-9)


def test_fuel_burns_in_proportion_to_throttle_and_empty_tank_gives_no_thrust():
    params = RocketParams()
    state, _ = step_rocket(upright(), Controls(0.5, 0.0, 0.0), params, 0.0, 1.0)
    assert state.fuel == pytest.approx(5_000.0 - 0.5 * params.max_burn_rate)

    empty = upright(fuel=0.0, throttle=1.0)
    forces = compute_forces(empty, Controls(1.0, 0.0, 1.0), params, wind_speed=0.0)
    assert np.allclose(forces.thrust, 0.0) and forces.torque == 0.0


def test_engine_lag_makes_throttle_respond_gradually():
    params = RocketParams(engine_lag=0.5)
    state, _ = step_rocket(upright(), Controls(1.0, 0.0, 0.0), params, 0.0, DT)
    assert 0.0 < state.throttle < 0.1


def test_wind_pushes_a_stationary_rocket_downwind():
    state, forces = step_rocket(upright(), ENGINE_OFF, RocketParams(), wind_speed=10.0, dt=DT)
    assert forces.drag[0] > 0.0 and state.vx > 0.0


def test_decode_action_maps_unit_box_to_physical_controls():
    params = RocketParams()
    low = decode_action(np.array([-1.0, -1.0, -1.0]), params)
    high = decode_action(np.array([1.0, 1.0, 1.0]), params)
    assert (low.throttle, low.gimbal, low.rcs) == (0.0, -params.max_gimbal, -1.0)
    assert (high.throttle, high.gimbal, high.rcs) == (1.0, params.max_gimbal, 1.0)
    clipped = decode_action(np.array([5.0, 0.0, 0.0]), params)
    assert clipped.throttle == 1.0


def test_body_to_world_rotates_counter_clockwise():
    state = upright(x=10.0, y=20.0, theta=np.pi / 2)
    world = body_to_world(state, np.array([[0.0, 1.0]]))  # the nose direction
    assert np.allclose(world, [[9.0, 20.0]])


@pytest.mark.parametrize("bad", [np.zeros(2), np.zeros((1, 3)), np.array([np.nan, 0.0, 0.0])])
def test_decode_action_rejects_malformed_actions(bad):
    with pytest.raises(ValueError, match="action must"):
        decode_action(bad, RocketParams())
