"""2D rigid-body rocket physics (SI units, theta counter-clockwise from vertical)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

GRAVITY = 9.81
SEA_LEVEL = -4.0


@dataclass(frozen=True)
class RocketParams:
    length: float = 20.0
    width: float = 2.0
    leg_half_span: float = 3.0
    leg_drop: float = 1.0  # leg tips below the body base
    dry_mass: float = 20_000.0
    initial_fuel: float = 5_000.0
    max_thrust: float = 490_000.0  # ~2x full weight
    max_burn_rate: float = 80.0  # kg/s
    max_gimbal: float = np.radians(15.0)
    max_rcs_torque: float = 150_000.0
    drag_x: float = 400.0
    drag_y: float = 50.0
    engine_lag: float = 0.0  # throttle time constant (s)

    @property
    def leg_clearance(self) -> float:
        """Distance from the centre of mass down to the leg tips."""
        return self.length / 2.0 + self.leg_drop

    def moment_of_inertia(self, mass: float) -> float:
        return mass * self.length**2 / 12.0

    def leg_tips_body(self) -> np.ndarray:
        """Left and right leg tips in the body frame, shape (2, 2)."""
        y = -self.length / 2.0 - self.leg_drop
        return np.array([[-self.leg_half_span, y], [self.leg_half_span, y]])


@dataclass(frozen=True)
class RocketState:
    x: float
    y: float
    vx: float
    vy: float
    theta: float
    omega: float
    fuel: float
    throttle: float = 0.0  # actual throttle after engine lag


@dataclass(frozen=True)
class Controls:
    throttle: float  # [0, 1]
    gimbal: float  # rad
    rcs: float  # [-1, 1], positive = counter-clockwise


@dataclass(frozen=True)
class Forces:
    thrust: np.ndarray
    gravity: np.ndarray
    drag: np.ndarray
    torque: float


def decode_action(action: np.ndarray, params: RocketParams) -> Controls:
    """Map an action in [-1, 1]^3 to physical controls."""
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (3,):
        raise ValueError(f"action must have shape (3,), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"action must be finite, got {a}")
    a = np.clip(a, -1.0, 1.0)
    return Controls(
        throttle=float((a[0] + 1.0) / 2.0),
        gimbal=float(a[1] * params.max_gimbal),
        rcs=float(a[2]),
    )


def compute_forces(
    state: RocketState, controls: Controls, params: RocketParams, wind_speed: float
) -> Forces:
    mass = params.dry_mass + state.fuel
    thrust_mag = state.throttle * params.max_thrust if state.fuel > 0.0 else 0.0
    direction = state.theta + controls.gimbal
    thrust = thrust_mag * np.array([-np.sin(direction), np.cos(direction)])
    gravity = np.array([0.0, -mass * GRAVITY])

    rel_vx = state.vx - wind_speed
    drag = np.array(
        [-params.drag_x * rel_vx * abs(rel_vx), -params.drag_y * state.vy * abs(state.vy)]
    )

    # Thrust acts at the base, so gimbal torque is -T sin(delta) L/2.
    gimbal_torque = -thrust_mag * np.sin(controls.gimbal) * params.length / 2.0
    rcs_torque = controls.rcs * params.max_rcs_torque if state.fuel > 0.0 else 0.0
    return Forces(thrust=thrust, gravity=gravity, drag=drag, torque=gimbal_torque + rcs_torque)


def step_rocket(
    state: RocketState, controls: Controls, params: RocketParams, wind_speed: float, dt: float
) -> tuple[RocketState, Forces]:
    """Advance one physics step with semi-implicit Euler."""
    if params.engine_lag > 0.0:
        blend = min(1.0, dt / params.engine_lag)
        throttle = state.throttle + blend * (controls.throttle - state.throttle)
    else:
        throttle = controls.throttle
    state = replace(state, throttle=throttle)

    forces = compute_forces(state, controls, params, wind_speed)
    mass = params.dry_mass + state.fuel
    accel = (forces.thrust + forces.gravity + forces.drag) / mass
    alpha = forces.torque / params.moment_of_inertia(mass)

    vx = state.vx + accel[0] * dt
    vy = state.vy + accel[1] * dt
    omega = state.omega + alpha * dt
    new_state = RocketState(
        x=state.x + vx * dt,
        y=state.y + vy * dt,
        vx=vx,
        vy=vy,
        theta=state.theta + omega * dt,
        omega=omega,
        fuel=max(0.0, state.fuel - throttle * params.max_burn_rate * dt),
        throttle=throttle,
    )
    return new_state, forces


def body_to_world(state: RocketState, points_body: np.ndarray) -> np.ndarray:
    c, s = np.cos(state.theta), np.sin(state.theta)
    rotation = np.array([[c, -s], [s, c]])
    return points_body @ rotation.T + np.array([state.x, state.y])
