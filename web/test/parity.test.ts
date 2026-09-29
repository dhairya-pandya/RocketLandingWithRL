// The TypeScript simulation must reproduce the Python environment (rocketlander/envs).
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Episode } from "../src/sim/episode";
import { flightAction, parseLevelFile, type Flight } from "../src/sim/mission";

const LEVELS = ["L0", "L1", "L2", "L3", "L4"];
const load = (path: string) => JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8"));

describe.each(LEVELS)("level %s", (level) => {
  const file = parseLevelFile(load(`../public/missions/${level}.json`));

  it("follows the Python reference trajectory step by step", () => {
    const reference: Flight & { states: Record<string, number>[] } = load(
      `./fixtures/${level}-reference.json`,
    );
    const episode = new Episode(file.missions[0], file.maxSteps);
    reference.states.forEach((expected, step) => {
      episode.step(flightAction(reference, step));
      for (const [key, value] of Object.entries(expected)) {
        expect(episode.rocket[key as keyof typeof episode.rocket]).toBeCloseTo(value, 6);
      }
    });
    expect(episode.done).toBe(true);
    expect(episode.judgement.outcome).toBe(reference.outcome);
    expect(episode.score).toBeCloseTo(reference.score, 6);
  });

  it("replays every recorded agent flight to its recorded result", () => {
    for (const agent of file.agents) {
      agent.flights.forEach((flight, m) => {
        const episode = new Episode(file.missions[m], file.maxSteps);
        for (let step = 0; step < flight.steps; step++) episode.step(flightAction(flight, step));
        const label = `${agent.id} mission ${m + 1}`;
        expect(episode.done, label).toBe(true);
        expect(
          episode.judgement.outcome === "in_flight" ? "in_flight" : episode.judgement.outcome,
          label,
        ).toBe(flight.outcome);
        expect(episode.score, label).toBeCloseTo(flight.score, 6);
      });
    }
  });
});
