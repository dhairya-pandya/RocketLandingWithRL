// The results screen: everyone ranked by score, and which pilots the player beat.
import { recordRace, type Progress } from "../game/progress";
import { standings } from "../game/scoring";
import type { RaceEnd } from "./raceScreen";
import { button, h } from "./dom";
import { modeOf } from "./modes";

export interface ResultActions {
  again: () => void; // a new mission and new opponents
  retry: () => void; // the same mission and opponents
  menu: () => void;
}

export function resultsScreen(
  end: RaceEnd,
  progress: Progress,
  actions: ResultActions,
): HTMLElement {
  const { race } = end;
  const rows = standings(race);
  const level = race.file.level;
  const mode = modeOf(level);
  const beaten = rows.filter((r) => r.beaten).map((r) => r.racer.agent!.id);
  const fresh = new Set(
    recordRace(progress, level, race.missionIndex + 1, race.player.episode.score, beaten),
  );
  const total = rows.length - 1;
  const place = rows.findIndex((r) => r.beaten === null) + 1;
  const headline =
    place === 1
      ? "You won the race"
      : beaten.length === 0
        ? "The AI pilots win this time"
        : `You beat ${beaten.length} of ${total} pilots`;
  const list = h(
    "ol",
    { className: "standings" },
    ...rows.map((r, i) => {
      const you = r.racer.agent === null;
      const result =
        r.outcome === "landed"
          ? `Landed at ${r.touchdownSpeed.toFixed(1)} m/s`
          : r.reason
            ? r.reason[0].toUpperCase() + r.reason.slice(1)
            : "Ran out of time";
      const isNew = !you && fresh.has(r.racer.agent!.id);
      return h(
        "li",
        { className: `${you ? "you" : ""}${r.outcome === "landed" ? " landed" : ""}` },
        h("span", { className: "place" }, String(i + 1)),
        h(
          "span",
          { className: "pilot" },
          you ? "You" : r.racer.label,
          isNew ? h("span", { className: "badge" }, "First win") : null,
        ),
        h("span", { className: "result" }, result),
        h("span", { className: "score" }, r.score.toFixed(0)),
      );
    }),
  );
  return h(
    "div",
    { className: "panel results", style: `--mode: ${mode.color}` },
    h("p", { className: "context" }, `${mode.name}, mission ${race.missionIndex + 1}`),
    h("h2", {}, headline),
    list,
    h(
      "p",
      { className: "muted small" },
      "Everyone flew the same start, ship and wind. The higher score wins.",
    ),
    h(
      "div",
      { className: "actions" },
      button("Fly a new mission", actions.again, "btn primary"),
      button("Retry this one", actions.retry, "btn"),
      button("Menu", actions.menu, "btn"),
    ),
  );
}
