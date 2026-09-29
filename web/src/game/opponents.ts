// Every race draws its opponents at random from the level's agents.

export const OPPONENTS = 4;

/** `count` distinct items in random order (Fisher-Yates on a copy); `random` returns [0, 1). */
export function drawOpponents<T>(
  agents: readonly T[],
  count = OPPONENTS,
  random = Math.random,
): T[] {
  const pool = [...agents];
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, Math.min(count, pool.length));
}
