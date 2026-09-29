// The player's controls, as rl-play's KeyboardPilot: throttle is sticky, gimbal and RCS spring back.

export const THROTTLE_RATE = 0.8; // throttle change per second while W/S is held

export type Control = "up" | "down" | "left" | "right" | "rcsLeft" | "rcsRight";

export class Pilot {
  throttle = 0.0; // [0, 1]
  readonly held = new Set<Control>();

  /** The action for the next pilot step of length dt seconds, in [-1, 1]^3. */
  action(dt: number): number[] {
    const change = Number(this.held.has("up")) - Number(this.held.has("down"));
    this.throttle = Math.min(1.0, Math.max(0.0, this.throttle + change * THROTTLE_RATE * dt));
    const gimbal = Number(this.held.has("right")) - Number(this.held.has("left"));
    const rcs = Number(this.held.has("rcsLeft")) - Number(this.held.has("rcsRight"));
    return [2.0 * this.throttle - 1.0, gimbal, rcs];
  }

  /** Touch slider: set the throttle directly. */
  setThrottle(value: number): void {
    this.throttle = Math.min(1.0, Math.max(0.0, value));
  }
}

const KEYS: Record<string, Control> = {
  KeyW: "up",
  ArrowUp: "up",
  KeyS: "down",
  ArrowDown: "down",
  KeyA: "left",
  ArrowLeft: "left",
  KeyD: "right",
  ArrowRight: "right",
  KeyQ: "rcsLeft",
  KeyE: "rcsRight",
};

export function controlForKey(code: string): Control | undefined {
  return KEYS[code];
}
