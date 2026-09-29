// One flight through a mission: RocketLanderEnv.step's physics, judge and task reward in TS.
import { judge, type Judgement } from "./landing";
import type { Mission } from "./mission";
import {
  decodeAction,
  PHYSICS_DT,
  PHYSICS_STEPS_PER_ACTION,
  stepRocket,
  type Controls,
  type RocketState,
} from "./physics";
import { deckState, type DeckState } from "./ship";

export const FUEL_COST = 0.05;
export const TIME_COST = 0.1;
const LANDING_REWARD = 100.0;
const SOFTNESS_BONUS = 50.0;
const CRASH_PENALTY = -100.0;
const DECK_CRASH_BASE = 20.0;
const DECK_CRASH_PER_MS = 8.0;

export function terminalReward(j: Judgement): number {
  if (j.outcome === "landed")
    return LANDING_REWARD + SOFTNESS_BONUS * (1.0 - j.touchdownSpeed / 2.0);
  if (j.outcome === "crashed" && j.reason !== "missed the ship") {
    return Math.max(CRASH_PENALTY, -(DECK_CRASH_BASE + DECK_CRASH_PER_MS * j.touchdownSpeed));
  }
  if (j.outcome === "crashed" || j.outcome === "failed") return CRASH_PENALTY;
  return 0.0;
}

export class Episode {
  rocket: RocketState;
  controls: Controls = { throttle: 0, gimbal: 0, rcs: 0 };
  judgement: Judgement = { outcome: "in_flight", reason: "", touchdownSpeed: 0 };
  time = 0.0;
  steps = 0;
  score = 0.0; // task reward so far, as the Python env's info["task_reward"] sums up
  timeLimitReached = false;

  constructor(
    readonly mission: Mission,
    readonly maxSteps: number,
  ) {
    this.rocket = { ...mission.start };
  }

  get done(): boolean {
    return this.judgement.outcome !== "in_flight" || this.timeLimitReached;
  }

  deck(): DeckState {
    return deckState(this.mission.ship, this.time);
  }

  /** Advance one pilot step (1/30 s) with an action in [-1, 1]^3; no-op once done. */
  step(action: readonly number[]): void {
    if (this.done) return;
    const params = this.mission.params;
    this.controls = decodeAction(action, params);
    const wind = this.mission.wind;
    for (let sub = 0; sub < PHYSICS_STEPS_PER_ACTION; sub++) {
      const k = Math.min(this.steps * PHYSICS_STEPS_PER_ACTION + sub, wind.length - 1);
      this.rocket = stepRocket(this.rocket, this.controls, params, wind[k] / 1000, PHYSICS_DT);
      this.time += PHYSICS_DT;
      this.judgement = judge(this.rocket, params, this.deck());
      if (this.judgement.outcome !== "in_flight") break;
    }
    this.steps += 1;
    this.timeLimitReached = this.judgement.outcome === "in_flight" && this.steps >= this.maxSteps;
    this.score += terminalReward(this.judgement) - FUEL_COST * this.controls.throttle - TIME_COST;
  }

  get fuelUsed(): number {
    return this.mission.params.initial_fuel - this.rocket.fuel;
  }
}
