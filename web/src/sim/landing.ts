// Touchdown judge, ported from rocketlander/envs/landing.py.
import {
  bodyToWorld,
  legTipsBody,
  SEA_LEVEL,
  type RocketParams,
  type RocketState,
} from "./physics";
import { onDeck, surfaceHeight, type DeckState } from "./ship";

export const MAX_VERTICAL_SPEED = 2.0; // relative to the deck
export const MAX_HORIZONTAL_SPEED = 1.5;
export const MAX_TILT = (10.0 * Math.PI) / 180.0;
export const MAX_SPIN = 0.3;
const MAX_HORIZONTAL_OFFSET = 400.0;
const MAX_ALTITUDE = 1_500.0;

export type Outcome = "in_flight" | "landed" | "crashed" | "failed";

export interface Judgement {
  outcome: Outcome;
  reason: string;
  touchdownSpeed: number;
}

const inFlight: Judgement = { outcome: "in_flight", reason: "", touchdownSpeed: 0.0 };

export function judge(rocket: RocketState, params: RocketParams, deck: DeckState): Judgement {
  const tips = bodyToWorld(rocket, legTipsBody(params));
  if (tips.some(([x, y]) => onDeck(deck, x) && y <= surfaceHeight(deck, x))) {
    return judgeTouchdown(rocket, deck, tips);
  }
  if (tips.some(([, y]) => y <= SEA_LEVEL)) {
    return { outcome: "crashed", reason: "missed the ship", touchdownSpeed: 0.0 };
  }
  if (Math.abs(rocket.x - deck.x) > MAX_HORIZONTAL_OFFSET || rocket.y > MAX_ALTITUDE) {
    return { outcome: "failed", reason: "out of bounds", touchdownSpeed: 0.0 };
  }
  if (rocket.fuel <= 0.0) return { outcome: "failed", reason: "out of fuel", touchdownSpeed: 0.0 };
  return inFlight;
}

function judgeTouchdown(rocket: RocketState, deck: DeckState, tips: number[][]): Judgement {
  const verticalSpeed = rocket.vy - deck.vy;
  const horizontalSpeed = rocket.vx - deck.vx;
  const tilt = rocket.theta - deck.angle;
  let reason = "";
  if (!tips.every(([x]) => onDeck(deck, x))) reason = "leg off the deck edge";
  else if (Math.abs(verticalSpeed) > MAX_VERTICAL_SPEED) reason = "too fast vertically";
  else if (Math.abs(horizontalSpeed) > MAX_HORIZONTAL_SPEED) reason = "too fast sideways";
  else if (Math.abs(tilt) > MAX_TILT) reason = "tilted";
  else if (Math.abs(rocket.omega) > MAX_SPIN) reason = "spinning";
  const touchdownSpeed = Math.abs(verticalSpeed);
  if (reason === "") return { outcome: "landed", reason: "landed", touchdownSpeed };
  return { outcome: "crashed", reason, touchdownSpeed };
}
