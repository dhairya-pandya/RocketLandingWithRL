// The per-level mission file written by `rl-export-web` (see web_export.py).
import type { RocketParams, RocketState } from "./physics";
import type { ShipMotion } from "./ship";

export const SCHEMA_VERSION = 1;

export interface Mission {
  seed: number;
  start: RocketState;
  params: RocketParams;
  ship: ShipMotion;
  wind: number[]; // mm/s per physics step
}

export interface Flight {
  actions: number[]; // thousandths, three per pilot step: throttle, gimbal, RCS
  outcome: "landed" | "crashed" | "failed" | "in_flight";
  reason: string;
  score: number;
  touchdownSpeed: number;
  fuelUsed: number;
  steps: number;
}

export interface AgentRecord {
  id: string;
  name: string;
  flights: Flight[]; // one per mission, same order
}

export interface LevelFile {
  schemaVersion: number;
  level: string;
  description: string;
  maxSteps: number;
  gitCommit: string;
  missions: Mission[];
  agents: AgentRecord[]; // best average score first
}

export class MissionFileError extends Error {}

/** Check a parsed mission file before the game trusts it. */
export function parseLevelFile(raw: unknown): LevelFile {
  const file = raw as Partial<LevelFile> | null;
  if (!file || typeof file !== "object") throw new MissionFileError("not a mission file");
  if (file.schemaVersion !== SCHEMA_VERSION) {
    throw new MissionFileError(`unsupported mission file version ${String(file.schemaVersion)}`);
  }
  if (!Array.isArray(file.missions) || !Array.isArray(file.agents) || file.missions.length === 0) {
    throw new MissionFileError("mission file has no missions");
  }
  return file as LevelFile;
}

/** An agent's action at a pilot step, as the numbers the env received. */
export function flightAction(flight: Flight, step: number): number[] {
  const i = step * 3;
  return [flight.actions[i] / 1000, flight.actions[i + 1] / 1000, flight.actions[i + 2] / 1000];
}
