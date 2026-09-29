// A race: the player and the chosen agents fly the same mission, one pilot step at a time.
import { Episode } from "../sim/episode";
import { flightAction, type AgentRecord, type LevelFile } from "../sim/mission";
import type { RocketState } from "../sim/physics";
import { deckState, type DeckState } from "../sim/ship";

export interface Racer {
  label: string;
  agent: AgentRecord | null; // null for the player
  episode: Episode;
  trail: [number, number][];
  poseOnDeck: [number, number, number] | null; // after landing: x, y, theta in the deck's frame
}

export class Race {
  readonly racers: Racer[];

  constructor(
    readonly file: LevelFile,
    readonly missionIndex: number,
    opponents: AgentRecord[],
  ) {
    const mission = file.missions[missionIndex];
    const racer = (label: string, agent: AgentRecord | null): Racer => ({
      label,
      agent,
      episode: new Episode(mission, file.maxSteps),
      trail: [],
      poseOnDeck: null,
    });
    this.racers = [racer("You", null), ...opponents.map((a) => racer(a.name, a))];
  }

  get player(): Racer {
    return this.racers[0];
  }

  get ghosts(): Racer[] {
    return this.racers.slice(1);
  }

  get finished(): boolean {
    return this.racers.every((r) => r.episode.done);
  }

  /** The latest simulated time of anyone still flying (the ship keeps moving for everyone). */
  get time(): number {
    return Math.max(...this.racers.map((r) => r.episode.time));
  }

  /** One pilot step: the player with `action`, each ghost with its recorded action. */
  step(action: readonly number[] | null): void {
    for (const r of this.racers) {
      if (r.episode.done) continue;
      if (r.agent === null) {
        if (action === null) continue;
        r.episode.step(action);
      } else {
        r.episode.step(flightAction(r.agent.flights[this.missionIndex], r.episode.steps));
      }
      r.trail.push([r.episode.rocket.x, r.episode.rocket.y]);
      if (r.episode.judgement.outcome === "landed") {
        r.poseOnDeck = toDeck(r.episode.rocket, r.episode.deck());
      }
    }
  }

  deck(): DeckState {
    return deckState(this.file.missions[this.missionIndex].ship, this.time);
  }

  /** Where to draw a racer now: a landed rocket rides along with the moving deck. */
  pose(r: Racer): RocketState {
    if (r.poseOnDeck === null) return r.episode.rocket;
    const [x, y, theta] = fromDeck(r.poseOnDeck, this.deck());
    return { ...r.episode.rocket, x, y, theta };
  }
}

function toDeck(r: RocketState, d: DeckState): [number, number, number] {
  const c = Math.cos(-d.angle);
  const s = Math.sin(-d.angle);
  const dx = r.x - d.x;
  const dy = r.y - d.y;
  return [c * dx - s * dy, s * dx + c * dy, r.theta - d.angle];
}

function fromDeck([lx, ly, lt]: [number, number, number], d: DeckState): [number, number, number] {
  const c = Math.cos(d.angle);
  const s = Math.sin(d.angle);
  return [c * lx - s * ly + d.x, s * lx + c * ly + d.y, lt + d.angle];
}
