// The landing page: a live replay of recorded AI flights behind the headline, the five
// difficulties in one row (a click starts a race), then how it works and common questions.
import type { Progress } from "../game/progress";
import type { LevelSummary } from "./data";
import { h } from "./dom";
import { icon } from "./icons";
import { modeOf, seaGlyph } from "./modes";

export const SOURCE_URL = "https://github.com/dhairya-pandya/RocketLandingWithRL";

const STORY =
  "Most of these pilots started out knowing nothing about rockets. " +
  "They crashed thousands of times before they landed one. " +
  "You get the same rocket, the same sea, and no practice.";

const STEPS: [string, string][] = [
  [
    "Pick a sea",
    "Five difficulties, from a calm sea to a fast fall from high up. Each race uses a mission drawn at random.",
  ],
  [
    "Fly with four AI pilots",
    "They fly beside you as translucent rockets, replaying flights recorded in exactly your wind and waves.",
  ],
  [
    "Land softly to beat them",
    "An upright, gentle touchdown on the deck scores most. Beat a pilot once and it stays beaten on this device.",
  ],
];

const KEYS: [string, string][] = [
  ["W S", "Throttle up and down"],
  ["A D", "Lean left and right"],
  ["Q E", "Side thrusters, to stop a spin"],
  ["Esc R", "Pause and restart"],
];

const QUESTIONS: [string, string][] = [
  [
    "Are the AI pilots flying live?",
    "No. Every flight was recorded ahead of time, one per mission, in exactly the wind and waves " +
      "you fly in. Your rocket runs on the same physics, so the comparison is exact.",
  ],
  [
    "How is my score worked out?",
    "An upright, slow touchdown on the deck earns the most. Every second and every bit of fuel " +
      "costs a little, and a crash costs a lot. The higher score wins.",
  ],
  [
    "What counts as a crash?",
    "Coming down faster than 2 m/s, sliding sideways faster than 1.5 m/s, leaning more than " +
      "10 degrees, or missing the deck. The readouts in the top left turn green when you are " +
      "within the limits.",
  ],
  [
    "Can I play on my phone?",
    "Yes. Turn it sideways, slide the throttle on the left and use the lean and thruster buttons " +
      "on the right.",
  ],
  [
    "Is my progress saved?",
    "On this device only. There are no accounts, no cookies, and nothing about you leaves your browser.",
  ],
  [
    "Who are the pilots?",
    "Twelve pilots, each named after a seabird. Two are autopilots written by hand; the other ten " +
      "taught themselves to fly with reinforcement learning. The code is open source.",
  ],
];

/** The five difficulties in one row; each tile starts a race. */
export function modeRow(
  levels: LevelSummary[],
  progress: Progress,
  play: (level: string) => void,
): HTMLElement {
  return h(
    "div",
    { className: "modes", role: "group", ariaLabel: "Choose a difficulty to start a race" },
    ...levels.map((l) => {
      const mode = modeOf(l.level);
      const beaten = (progress.beaten[l.level] ?? []).length;
      return h(
        "button",
        {
          className: "mode",
          type: "button",
          onclick: () => play(l.level),
          style: `--mode: ${mode.color}`,
          ariaLabel: `${mode.name}: ${mode.tagline}. ${beaten} of ${l.agents.length} pilots beaten.`,
        },
        seaGlyph(mode.rank),
        h("strong", {}, mode.name),
        h("span", { className: "tagline" }, mode.tagline),
        h(
          "span",
          { className: "beaten" },
          `${beaten}/${l.agents.length}`,
          h("span", { className: "wide" }, " beaten"),
        ),
      );
    }),
  );
}

function hero(levels: LevelSummary[], progress: Progress, play: (l: string) => void): HTMLElement {
  return h(
    "header",
    { className: "hero", id: "top" },
    h(
      "div",
      { className: "hero-copy" },
      h("h1", {}, "Can you land a rocket", h("br"), "better than the AI?"),
      h(
        "p",
        { className: "sub" },
        "Bring a falling rocket down onto a drone ship at sea while four AI pilots fly the same " +
          "mission beside you. Same wind, same waves, higher score wins.",
      ),
    ),
    h(
      "div",
      { className: "hero-play" },
      modeRow(levels, progress, play),
      h("p", { className: "caption", id: "now-playing" }, "Loading the replays…"),
    ),
  );
}

function nav(): HTMLElement {
  return h(
    "nav",
    { className: "nav", ariaLabel: "Page" },
    h("a", { className: "brand", href: "#top" }, "Rocket Landing"),
    h("a", { className: "section-link", href: "#how" }, "How it works"),
    h("a", { className: "section-link", href: "#questions" }, "Questions"),
    h(
      "a",
      { href: SOURCE_URL, target: "_blank", rel: "noopener", ariaLabel: "Source code on GitHub" },
      icon("github"),
    ),
  );
}

function story(): HTMLElement {
  const words = STORY.split(" ").map((w) => h("span", { className: "word" }, w + " "));
  return h("section", { className: "story" }, h("p", {}, ...words));
}

function how(): HTMLElement {
  return h(
    "section",
    { className: "how reveal", id: "how" },
    h("h2", {}, "How a race works"),
    h(
      "ol",
      { className: "steps" },
      ...STEPS.map(([title, text]) => h("li", {}, h("h3", {}, title), h("p", {}, text))),
    ),
    h(
      "div",
      { className: "controls" },
      h("h3", {}, "Controls"),
      h(
        "dl",
        {},
        ...KEYS.flatMap(([keys, what]) => [
          h("dt", {}, ...keys.split(" ").map((k) => h("kbd", {}, k))),
          h("dd", {}, what),
        ]),
      ),
    ),
  );
}

function questions(): HTMLElement {
  return h(
    "section",
    { className: "questions reveal", id: "questions" },
    h("h2", {}, "Questions"),
    ...QUESTIONS.map(([q, a]) => h("details", {}, h("summary", {}, q), h("p", {}, a))),
  );
}

function finale(
  levels: LevelSummary[],
  progress: Progress,
  play: (l: string) => void,
): HTMLElement {
  return h(
    "section",
    { className: "finale reveal" },
    h("h2", {}, "The drone ship is waiting"),
    modeRow(levels, progress, play),
  );
}

function footer(): HTMLElement {
  return h(
    "footer",
    { className: "footer" },
    h("p", {}, "Built on the open source RocketLandingWithRL project. No cookies, no tracking."),
    h("a", { href: SOURCE_URL, target: "_blank", rel: "noopener" }, icon("github"), "Source"),
  );
}

let observers: IntersectionObserver[] = [];

/** Builds the page. `scroller` is the element that scrolls; `heroVisible` pauses the replay. */
export function landingPage(
  levels: LevelSummary[],
  progress: Progress,
  play: (level: string) => void,
  scroller: HTMLElement,
  heroVisible: (visible: boolean) => void,
): HTMLElement {
  observers.forEach((o) => o.disconnect());
  const page = h(
    "div",
    { className: "landing" },
    h("a", { className: "skip", href: "#how" }, "Skip to how it works"),
    nav(),
    hero(levels, progress, play),
    h("main", {}, story(), how(), questions(), finale(levels, progress, play)),
    footer(),
  );
  const heroObserver = new IntersectionObserver(([entry]) => heroVisible(entry.isIntersecting), {
    root: scroller,
  });
  heroObserver.observe(page.querySelector(".hero")!);
  // Sections rise into view once; words of the story light up as they cross a line 60% down.
  const reveal = new IntersectionObserver(
    (entries) =>
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        e.target.classList.add("shown");
        reveal.unobserve(e.target);
      }),
    { root: scroller, threshold: 0.15 },
  );
  page.querySelectorAll(".reveal").forEach((el) => reveal.observe(el));
  const words = new IntersectionObserver(
    (entries) => {
      const lit = entries.filter((e) => e.isIntersecting).map((e) => e.target as HTMLElement);
      lit.forEach((el, i) => {
        el.style.transitionDelay = `${i * 60}ms`; // one at a time, in reading order
        el.classList.add("lit");
        words.unobserve(el);
      });
    },
    { root: scroller, rootMargin: "0px 0px -40% 0px", threshold: 1 },
  );
  page.querySelectorAll(".word").forEach((el) => words.observe(el));
  observers = [heroObserver, reveal, words];
  return page;
}
