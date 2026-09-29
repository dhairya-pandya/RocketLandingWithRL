// The app: landing page, race, results. Choosing a difficulty starts a race right away,
// on a random mission against four randomly drawn agents.
import "./style.css";
import { drawOpponents } from "./game/opponents";
import { loadProgress, saveProgress, type Progress } from "./game/progress";
import type { AgentRecord, LevelFile } from "./sim/mission";
import { loadIndex, loadLevel, type LevelSummary } from "./ui/data";
import { button, h, show } from "./ui/dom";
import { landingPage } from "./ui/landing";
import { modeOf } from "./ui/modes";
import { RaceScreen, type RaceEnd } from "./ui/raceScreen";
import { resultsScreen } from "./ui/results";

const canvas = document.querySelector<HTMLCanvasElement>("#game")!;
const ui = document.querySelector<HTMLDivElement>("#ui")!;
const race = new RaceScreen(canvas);
const progress: Progress = loadProgress(localStorage);
let levels: LevelSummary[] = [];

function errorScreen(message: string, retry: () => void): void {
  ui.className = "centered";
  show(
    ui,
    h(
      "div",
      { className: "panel", role: "alert" },
      h("h2", {}, "The mission files did not load"),
      h("p", { className: "muted" }, `${message}. Check your connection and try again.`),
      h(
        "div",
        { className: "actions" },
        button("Try again", retry, "btn primary"),
        button("Back to menu", () => void home(), "btn"),
      ),
    ),
  );
}

async function home(): Promise<void> {
  try {
    levels = levels.length ? levels : await loadIndex();
  } catch (error) {
    return errorScreen((error as Error).message, () => void home());
  }
  race.showcase();
  ui.className = "scroll";
  show(
    ui,
    landingPage(
      levels,
      progress,
      (level) => void play(level),
      ui,
      (v) => (race.showcaseVisible = v),
    ),
  );
  ui.scrollTop = 0;
  frameShowcase();
}

/** Frame the replay beside the headline when there is room, else between it and the row. */
function frameShowcase(): void {
  const copy = ui.querySelector(".hero-copy .sub")?.getBoundingClientRect();
  const row = ui.querySelector(".hero-play")?.getBoundingClientRect();
  if (!copy || !row) return;
  const right = copy.left + copy.width + 32;
  const band = row.top - copy.bottom - 16;
  race.showcaseFrame =
    innerWidth - right >= 360
      ? { x: right, y: 88, w: innerWidth - right - 24, h: row.top - 104 }
      : band >= 160
        ? { x: 0, y: copy.bottom + 8, w: innerWidth, h: band }
        : row.top >= 264
          ? { x: 0, y: 88, w: innerWidth, h: row.top - 104 } // above the row, behind the text
          : null; // a short landscape screen: the row is below the fold anyway
}

addEventListener("resize", frameShowcase);

race.onShowcase = (names, level) => {
  const caption = document.querySelector("#now-playing");
  if (caption) caption.textContent = `Replaying now: ${names.join(", ")} on ${modeOf(level).name}.`;
};

/** Start a race at once: a random mission and four random opponents. */
async function play(level: string): Promise<void> {
  const mode = modeOf(level);
  ui.className = "centered";
  show(
    ui,
    h(
      "div",
      { className: "panel loading", style: `--mode: ${mode.color}`, ariaBusy: "true" },
      h("h2", {}, `Loading your ${mode.name} mission`),
      h("div", { className: "skeleton" }),
      h("div", { className: "skeleton short" }),
    ),
  );
  let file: LevelFile;
  try {
    file = await loadLevel(level);
  } catch (error) {
    return errorScreen((error as Error).message, () => void play(level));
  }
  const mission = Math.floor(Math.random() * file.missions.length);
  startRace(file, mission, drawOpponents(file.agents));
}

function startRace(file: LevelFile, mission: number, opponents: AgentRecord[]): void {
  ui.hidden = true;
  race.start(file, mission, opponents, (end: RaceEnd) => results(file, mission, opponents, end));
}

function results(file: LevelFile, mission: number, opponents: AgentRecord[], end: RaceEnd): void {
  ui.className = "centered";
  show(
    ui,
    resultsScreen(end, progress, {
      again: () => void play(file.level),
      retry: () => startRace(file, mission, opponents),
      menu: () => void home(),
    }),
  );
  saveProgress(localStorage, progress);
}

race.onQuit = () => void home();
void home();
