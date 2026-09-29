import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { drawOpponents } from "../src/game/opponents";
import { Pilot, THROTTLE_RATE, controlForKey } from "../src/game/pilot";
import {
  emptyProgress,
  loadProgress,
  recordRace,
  saveProgress,
  STORAGE_KEY,
} from "../src/game/progress";
import { Race } from "../src/game/race";
import { beats, standings } from "../src/game/scoring";
import { MissionFileError, parseLevelFile, type LevelFile } from "../src/sim/mission";

const level = (name: string): LevelFile =>
  parseLevelFile(
    JSON.parse(readFileSync(new URL(`../public/missions/${name}.json`, import.meta.url), "utf8")),
  );

class MemoryStorage {
  data = new Map<string, string>();
  getItem = (k: string) => this.data.get(k) ?? null;
  setItem = (k: string, v: string) => void this.data.set(k, v);
}

describe("pilot", () => {
  it("keeps the throttle where it was left and springs the gimbal back", () => {
    const pilot = new Pilot();
    pilot.held.add("up");
    expect(pilot.action(0.5)[0]).toBeCloseTo(2 * THROTTLE_RATE * 0.5 - 1);
    pilot.held.clear();
    const [throttle, gimbal, rcs] = pilot.action(0.5);
    expect(throttle).toBeCloseTo(2 * THROTTLE_RATE * 0.5 - 1);
    expect([gimbal, rcs]).toEqual([0, 0]);
  });

  it("clamps the throttle and maps the keys like rl-play", () => {
    const pilot = new Pilot();
    pilot.held.add("down");
    expect(pilot.action(1)[0]).toBe(-1);
    pilot.setThrottle(2);
    expect(pilot.throttle).toBe(1);
    expect([controlForKey("KeyA"), controlForKey("ArrowRight"), controlForKey("KeyQ")]).toEqual([
      "left",
      "right",
      "rcsLeft",
    ]);
    expect(controlForKey("KeyZ")).toBeUndefined();
  });
});

describe("race", () => {
  it("replays each ghost to the result recorded in Python", () => {
    const file = level("L2");
    const race = new Race(file, 3, file.agents);
    while (!race.ghosts.every((g) => g.episode.done)) race.step(null);
    for (const ghost of race.ghosts) {
      const recorded = ghost.agent!.flights[3];
      expect(ghost.episode.judgement.outcome, ghost.label).toBe(recorded.outcome);
      expect(ghost.episode.score).toBeCloseTo(recorded.score, 6);
    }
    expect(race.player.episode.steps).toBe(0); // the player only moves with an action
  });

  it("keeps a landed rocket on the moving deck", () => {
    const file = level("L2");
    const lander = file.agents.find((a) => a.flights[0].outcome === "landed")!;
    const race = new Race(file, 0, [lander]);
    const hover = [0.3, 0, 0]; // the player stays in the air while the ghost lands
    while (!race.ghosts[0].episode.done) race.step(hover);
    const ghost = race.ghosts[0];
    const touchdown = race.pose(ghost);
    for (let i = 0; i < 90; i++) race.step(hover); // 3 more seconds of the ship moving
    expect(race.player.episode.done).toBe(false);
    const later = race.pose(ghost);
    const deck = race.deck();
    expect(later.x).not.toBeCloseTo(touchdown.x, 3);
    expect(Math.abs(later.x - deck.x)).toBeLessThan(deck.halfWidth);
  });
});

describe("scoring", () => {
  it("needs a strictly higher score to beat an agent", () => {
    expect(beats(10, 9.9)).toBe(true);
    expect(beats(10, 10)).toBe(false);
  });

  it("ranks everyone by score and marks the agents the player beat", () => {
    const file = level("L0");
    const race = new Race(file, 0, file.agents.slice(-2)); // the two weakest agents
    while (!race.finished) race.step([-1, 0, 0]); // the player free-falls and crashes
    const rows = standings(race);
    expect(rows.map((r) => r.score)).toEqual([...rows.map((r) => r.score)].sort((a, b) => b - a));
    const you = rows.find((r) => r.beaten === null)!;
    expect(you.outcome).toBe("crashed");
    for (const r of rows) if (r.beaten !== null) expect(r.beaten).toBe(you.score > r.score);
  });
});

describe("progress", () => {
  it("survives corrupt or foreign data by starting afresh", () => {
    const storage = new MemoryStorage();
    storage.setItem(STORAGE_KEY, "{not json");
    expect(loadProgress(storage)).toEqual(emptyProgress());
    storage.setItem(STORAGE_KEY, JSON.stringify({ version: 2, beaten: {} }));
    expect(loadProgress(storage)).toEqual(emptyProgress());
  });

  it("records new wins once and keeps the best score per mission", () => {
    const storage = new MemoryStorage();
    const progress = loadProgress(storage);
    expect(recordRace(progress, "L1", 3, 50, ["pid", "dqn"])).toEqual(["pid", "dqn"]);
    expect(recordRace(progress, "L1", 3, 20, ["pid", "sac"])).toEqual(["sac"]);
    saveProgress(storage, progress);
    const loaded = loadProgress(storage);
    expect(loaded.beaten.L1.sort()).toEqual(["dqn", "pid", "sac"]);
    expect(loaded.best.L1["3"]).toBe(50);
  });
});

describe("mission files", () => {
  it("refuses unknown versions and empty files", () => {
    expect(() => parseLevelFile({ schemaVersion: 99 })).toThrow(MissionFileError);
    expect(() => parseLevelFile({ schemaVersion: 1, missions: [], agents: [] })).toThrow(
      MissionFileError,
    );
    expect(() => parseLevelFile(null)).toThrow(MissionFileError);
  });
});

describe("opponents", () => {
  const seeded = (seed: number) => () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;

  it("draws four distinct agents, different draws for different luck", () => {
    const agents = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"];
    const first = drawOpponents(agents, 4, seeded(1));
    expect(first).toHaveLength(4);
    expect(new Set(first).size).toBe(4);
    expect(first.every((a) => agents.includes(a))).toBe(true);
    const draws = new Set(
      Array.from({ length: 20 }, (_, s) => drawOpponents(agents, 4, seeded(s + 2)).join()),
    );
    expect(draws.size).toBeGreaterThan(10);
    expect(agents).toHaveLength(12); // the level's list is not shuffled in place
  });

  it("returns everyone when a level has fewer agents than seats", () => {
    expect(drawOpponents(["a", "b"], 4).sort()).toEqual(["a", "b"]);
  });
});
