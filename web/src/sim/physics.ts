// 2D rigid-body rocket physics, ported line by line from rocketlander/envs/physics.py.
// The operation order matches the Python so both give the same numbers.

export const GRAVITY = 9.81;
export const SEA_LEVEL = -4.0;
export const PHYSICS_DT = 1.0 / 60.0;
export const PHYSICS_STEPS_PER_ACTION = 2; // the pilot acts at 30 Hz

export interface RocketParams {
  length: number;
  width: number;
  leg_half_span: number;
  leg_drop: number;
  dry_mass: number;
  initial_fuel: number;
  max_thrust: number;
  max_burn_rate: number;
  max_gimbal: number;
  max_rcs_torque: number;
  drag_x: number;
  drag_y: number;
  engine_lag: number;
}

export interface RocketState {
  x: number;
  y: number;
  vx: number;
  vy: number;
  theta: number; // counter-clockwise from vertical
  omega: number;
  fuel: number;
  throttle: number; // actual throttle after engine lag
}

export interface Controls {
  throttle: number; // [0, 1]
  gimbal: number; // rad
  rcs: number; // [-1, 1], positive = counter-clockwise
}

const clip = (v: number) => Math.min(1.0, Math.max(-1.0, v));

/** Map an action in [-1, 1]^3 (throttle, gimbal, RCS) to physical controls. */
export function decodeAction(action: readonly number[], params: RocketParams): Controls {
  const a = action.map(clip);
  return { throttle: (a[0] + 1.0) / 2.0, gimbal: a[1] * params.max_gimbal, rcs: a[2] };
}

export function momentOfInertia(params: RocketParams, mass: number): number {
  return (mass * params.length ** 2) / 12.0;
}

/** Advance one physics step with semi-implicit Euler. */
export function stepRocket(
  state: RocketState,
  controls: Controls,
  params: RocketParams,
  windSpeed: number,
  dt: number,
): RocketState {
  let throttle = controls.throttle;
  if (params.engine_lag > 0.0) {
    const blend = Math.min(1.0, dt / params.engine_lag);
    throttle = state.throttle + blend * (controls.throttle - state.throttle);
  }
  const fuel = state.fuel;
  const mass = params.dry_mass + fuel;
  const thrustMag = fuel > 0.0 ? throttle * params.max_thrust : 0.0;
  const direction = state.theta + controls.gimbal;
  const thrustX = thrustMag * -Math.sin(direction);
  const thrustY = thrustMag * Math.cos(direction);
  const gravityY = -mass * GRAVITY;
  const relVx = state.vx - windSpeed;
  const dragX = -params.drag_x * relVx * Math.abs(relVx);
  const dragY = -params.drag_y * state.vy * Math.abs(state.vy);
  const gimbalTorque = (-thrustMag * Math.sin(controls.gimbal) * params.length) / 2.0;
  const rcsTorque = fuel > 0.0 ? controls.rcs * params.max_rcs_torque : 0.0;

  const accelX = (thrustX + 0.0 + dragX) / mass;
  const accelY = (thrustY + gravityY + dragY) / mass;
  const alpha = (gimbalTorque + rcsTorque) / momentOfInertia(params, mass);

  const vx = state.vx + accelX * dt;
  const vy = state.vy + accelY * dt;
  const omega = state.omega + alpha * dt;
  return {
    x: state.x + vx * dt,
    y: state.y + vy * dt,
    vx,
    vy,
    theta: state.theta + omega * dt,
    omega,
    fuel: Math.max(0.0, state.fuel - throttle * params.max_burn_rate * dt),
    throttle,
  };
}

/** Rocket-frame points [[x, y], ...] to world coordinates. */
export function bodyToWorld(state: RocketState, points: readonly (readonly number[])[]) {
  const c = Math.cos(state.theta);
  const s = Math.sin(state.theta);
  return points.map(([px, py]) => [px * c + py * -s + state.x, px * s + py * c + state.y]);
}

export function legTipsBody(params: RocketParams): number[][] {
  const y = -params.length / 2.0 - params.leg_drop;
  return [
    [-params.leg_half_span, y],
    [params.leg_half_span, y],
  ];
}
