// Loading the exported mission files, with a friendly error instead of a blank page.
import {
  MissionFileError,
  parseLevelFile,
  SCHEMA_VERSION,
  type AgentRecord,
  type LevelFile,
} from "../sim/mission";

export interface LevelSummary {
  level: string;
  description: string;
  missions: number;
  agents: Pick<AgentRecord, "id" | "name">[];
}

const cache = new Map<string, Promise<LevelFile>>();

async function getJson(path: string): Promise<unknown> {
  const response = await fetch(path);
  if (!response.ok) throw new MissionFileError(`could not load ${path} (${response.status})`);
  return response.json();
}

export async function loadIndex(): Promise<LevelSummary[]> {
  const raw = (await getJson("/missions/index.json")) as {
    schemaVersion?: number;
    levels?: LevelSummary[];
  };
  if (raw.schemaVersion !== SCHEMA_VERSION || !Array.isArray(raw.levels)) {
    throw new MissionFileError("the game data is out of date; please reload the page");
  }
  return raw.levels;
}

export function loadLevel(level: string): Promise<LevelFile> {
  let pending = cache.get(level);
  if (!pending) {
    pending = getJson(`/missions/${level}.json`).then(parseLevelFile);
    pending.catch(() => cache.delete(level)); // allow a retry after a failure
    cache.set(level, pending);
  }
  return pending;
}
