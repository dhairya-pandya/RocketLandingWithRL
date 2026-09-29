"""Cascaded PID controller: the hand-written baseline the RL agents must beat."""

from __future__ import annotations

import numpy as np

from rocketlander.envs.physics import GRAVITY, RocketParams
from rocketlander.envs.rocket_env import FUEL_SCALE


class PIDAgent:
    def __init__(self, params: RocketParams | None = None) -> None:
        self.params = params or RocketParams()

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        p = self.params
        dx, height = obs[0] * 100.0, obs[1] * 100.0
        dvx, dvy = obs[2] * 20.0, obs[3] * 20.0
        theta = float(np.arctan2(obs[4], obs[5]))
        omega = float(obs[6])
        deck_angle = float(obs[7]) * 0.1
        mass = p.dry_mass + float(obs[9]) * FUEL_SCALE
        altitude = max(0.0, height - p.leg_clearance)

        # Vertical: constant-deceleration descent profile with feedforward.
        decel = 2.0
        target_vy = -(0.5 + np.sqrt(2.0 * decel * altitude))
        misaligned = abs(dx) > 3.0 or abs(dvx) > 1.0
        if altitude < 15.0 and misaligned:
            target_vy = max(target_vy, -1.0)  # hover down slowly until lined up
        desired_accel_y = decel + 1.5 * (target_vy - dvy)
        max_accel = p.max_thrust * max(np.cos(theta), 0.5) / mass
        throttle = np.clip((GRAVITY + desired_accel_y) / max_accel, 0.0, 1.0)

        # Horizontal: tilt toward the deck (theta < 0 pushes right).
        target_vx = np.clip(-0.2 * dx, -10.0, 10.0)
        desired_accel = 1.0 * (target_vx - dvx)
        target_theta = np.clip(-desired_accel / GRAVITY, -0.25, 0.25)
        if altitude < 8.0:
            target_theta = np.clip(target_theta, deck_angle - 0.04, deck_angle + 0.04)

        # Attitude: PD -> desired torque -> gimbal angle and RCS.
        desired_alpha = 4.0 * (target_theta - theta) - 3.0 * omega
        torque = p.moment_of_inertia(mass) * desired_alpha
        thrust = max(throttle * p.max_thrust, 1.0)
        sin_gimbal = np.clip(-torque / (thrust * p.length / 2.0), -1.0, 1.0)
        gimbal_action = np.clip(np.arcsin(sin_gimbal) / p.max_gimbal, -1.0, 1.0)
        rcs_action = np.clip(torque / p.max_rcs_torque, -1.0, 1.0)

        return np.array([2.0 * throttle - 1.0, gimbal_action, rcs_action], dtype=np.float32)
