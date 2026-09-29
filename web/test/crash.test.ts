import { describe, expect, it } from "vitest";
import { Crash } from "../src/render/crash";
import { SEA_LEVEL, type RocketParams, type RocketState } from "../src/sim/physics";
import { surfaceHeight, type DeckState } from "../src/sim/ship";

const params = { length: 20, width: 2, leg_half_span: 3, leg_drop: 1 } as RocketParams;
const deck: DeckState = { x: 0, y: 0, vx: 0, vy: 0, angle: 0, angularVelocity: 0, halfWidth: 15 };

function seeded(seed: number): () => number {
  return () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;
}

function rocket(x: number, y: number, vy: number): RocketState {
  return { x, y, vx: 0, vy, theta: 0.2, omega: 0, fuel: 100, throttle: 0 };
}

function run(crash: Crash, seconds: number, d = deck): [number, number][] {
  const splashes: [number, number][] = [];
  for (let t = 0; t < seconds; t += 1 / 60) splashes.push(...crash.update(1 / 60, d).splashes);
  return splashes;
}

describe("crash", () => {
  it("breaks the rocket into pieces that come to rest on the deck, never inside it", () => {
    const crash = new Crash(rocket(0, 11, -6), params, deck, seeded(1));
    expect(crash.debris.length).toBeGreaterThanOrEqual(6);
    run(crash, 6);
    const onDeck = crash.debris.filter((p) => p.resting);
    expect(onDeck.length).toBeGreaterThan(0);
    for (const p of onDeck) expect(p.y).toBeGreaterThanOrEqual(surfaceHeight(deck, p.x));
  });

  it("keeps resting pieces on a moving deck", () => {
    const crash = new Crash(rocket(0, 11, -6), params, deck, seeded(2));
    run(crash, 6);
    const moved = { ...deck, x: 3, y: 0.5, angle: 0.05 };
    crash.update(1 / 60, moved);
    for (const p of crash.debris.filter((q) => q.resting)) {
      expect(p.y).toBeGreaterThanOrEqual(surfaceHeight(moved, p.x));
      expect(Math.abs(p.x - moved.x)).toBeLessThanOrEqual(moved.halfWidth);
    }
  });

  it("sinks every piece and splashes when the rocket misses the ship", () => {
    const crash = new Crash(rocket(80, SEA_LEVEL + 10, -8), params, deck, seeded(3));
    const splashes = run(crash, 8);
    expect(splashes.length).toBeGreaterThan(0);
    expect(crash.debris.every((p) => p.sunk >= 1)).toBe(true);
    expect(crash.finished).toBe(true);
  });

  it("flashes a fireball that grows and then fades", () => {
    const crash = new Crash(rocket(0, 11, -6), params, deck, seeded(4));
    run(crash, 0.1);
    const early = crash.fireball();
    run(crash, 0.2);
    const later = crash.fireball();
    expect(early.alpha).toBeGreaterThan(0.5);
    expect(later.radius).toBeGreaterThan(early.radius);
    run(crash, 1);
    expect(crash.fireball().alpha).toBe(0);
  });
});
