// The player's record, kept in localStorage: which agents they have beaten on which level.

export const STORAGE_KEY = "rocketlander.progress.v1";

export interface Progress {
  version: 1;
  beaten: Record<string, string[]>; // level -> agent ids
  best: Record<string, Record<string, number>>; // level -> mission number -> best score
}

export const emptyProgress = (): Progress => ({ version: 1, beaten: {}, best: {} });

/** Read the record; anything unreadable or from another version starts afresh. */
export function loadProgress(storage: Pick<Storage, "getItem">): Progress {
  try {
    const raw = JSON.parse(storage.getItem(STORAGE_KEY) ?? "null");
    if (raw?.version === 1 && typeof raw.beaten === "object" && typeof raw.best === "object") {
      return raw as Progress;
    }
  } catch {
    // corrupt JSON: fall through to an empty record
  }
  return emptyProgress();
}

export function saveProgress(storage: Pick<Storage, "setItem">, progress: Progress): void {
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(progress));
  } catch {
    // storage full or blocked (private mode): the game still works, it just forgets
  }
}

/** Record a race; returns the agent ids beaten for the first time on this level. */
export function recordRace(
  progress: Progress,
  level: string,
  mission: number,
  score: number,
  beatenIds: string[],
): string[] {
  const already = new Set(progress.beaten[level] ?? []);
  const fresh = beatenIds.filter((id) => !already.has(id));
  progress.beaten[level] = [...already, ...fresh];
  const best = (progress.best[level] ??= {});
  best[String(mission)] = Math.max(best[String(mission)] ?? -Infinity, score);
  return fresh;
}
