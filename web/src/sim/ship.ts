// Drone ship motion as closed-form functions of time, ported from rocketlander/envs/ship.py.

export interface ShipMotion {
  sway_amplitude: number;
  sway_frequency: number;
  sway_phase: number;
  drift_speed: number;
  heave_amplitude: number;
  heave_frequency: number;
  heave_phase: number;
  roll_amplitude: number;
  roll_frequency: number;
  roll_phase: number;
  deck_half_width: number;
}

export interface DeckState {
  x: number;
  y: number; // deck top at the centre
  vx: number;
  vy: number;
  angle: number;
  angularVelocity: number;
  halfWidth: number;
}

const TWO_PI = 2.0 * Math.PI;

function wave(amplitude: number, frequency: number, phase: number, t: number): [number, number] {
  const arg = TWO_PI * frequency * t + phase;
  return [amplitude * Math.sin(arg), amplitude * TWO_PI * frequency * Math.cos(arg)];
}

export function deckState(m: ShipMotion, t: number): DeckState {
  const [sway, swayRate] = wave(m.sway_amplitude, m.sway_frequency, m.sway_phase, t);
  const [heave, heaveRate] = wave(m.heave_amplitude, m.heave_frequency, m.heave_phase, t);
  const [roll, rollRate] = wave(m.roll_amplitude, m.roll_frequency, m.roll_phase, t);
  return {
    x: sway + m.drift_speed * t,
    y: heave,
    vx: swayRate + m.drift_speed,
    vy: heaveRate,
    angle: roll,
    angularVelocity: rollRate,
    halfWidth: m.deck_half_width,
  };
}

export function surfaceHeight(deck: DeckState, x: number): number {
  return deck.y + (x - deck.x) * Math.tan(deck.angle);
}

export function onDeck(deck: DeckState, x: number): boolean {
  return Math.abs(x - deck.x) <= deck.halfWidth * Math.cos(deck.angle);
}
