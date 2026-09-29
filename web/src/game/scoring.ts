// Who beat whom: task scores on the same mission, higher wins (ties are not wins).
import type { Race, Racer } from "./race";

export interface Standing {
  racer: Racer;
  score: number;
  outcome: string;
  reason: string;
  touchdownSpeed: number;
  fuelUsed: number;
  beaten: boolean | null; // for agents: did the player beat them? null for the player
}

export function beats(playerScore: number, agentScore: number): boolean {
  return playerScore > agentScore;
}

/** Everyone in the race ranked by score, best first. */
export function standings(race: Race): Standing[] {
  const player = race.player.episode.score;
  return race.racers
    .map((racer) => {
      const e = racer.episode;
      const outcome = e.judgement.outcome === "in_flight" ? "time limit" : e.judgement.outcome;
      return {
        racer,
        score: e.score,
        outcome,
        reason: e.judgement.reason,
        touchdownSpeed: e.judgement.touchdownSpeed,
        fuelUsed: e.fuelUsed,
        beaten: racer.agent === null ? null : beats(player, e.score),
      };
    })
    .sort((a, b) => b.score - a.score);
}
