// The race: a fixed-timestep loop (the pilot acts at 30 Hz, as in Python), keyboard and touch
// input, countdown, pause, and the ghosts-only loop behind the title screen.
import { drawOpponents } from "../game/opponents";
import { controlForKey, Pilot, type Control } from "../game/pilot";
import { Race } from "../game/race";
import type { Frame } from "../render/camera";
import { RaceView } from "../render/view";
import type { AgentRecord, LevelFile } from "../sim/mission";
import { loadLevel } from "./data";
import { button, h } from "./dom";
import { icon, type IconName } from "./icons";

export const STEP = 1 / 30;
const COUNTDOWN = 3; // seconds
const FAST_FORWARD = 4; // ghost steps per pilot step once the player has finished
const HOLD = 2.5; // seconds to show the finished race (and any crash) before the results

export interface RaceEnd {
  race: Race;
}

type Mode = "idle" | "showcase" | "race";

export class RaceScreen {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly view: RaceView;
  private readonly overlay = h("div", { className: "overlay" });
  private mode: Mode = "idle";
  private race: Race | null = null;
  private pilot = new Pilot();
  private onEnd: (end: RaceEnd) => void = () => {};
  onQuit: () => void = () => {};
  /** Called with the pilots and level each time the landing page's replay starts. */
  onShowcase: (names: string[], level: string) => void = () => {};
  /** False while the landing page's replay is scrolled out of view: nothing to draw. */
  showcaseVisible = true;

  /** The free space on the landing page to frame the replay in. */
  set showcaseFrame(frame: Frame | null) {
    this.view.showcaseFrame = frame;
  }
  private args: [LevelFile, number, AgentRecord[]] | null = null;
  private clock = 0; // real seconds since the race screen started
  private acc = 0;
  private hold = 0;
  private paused = false;
  private last = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    this.view = new RaceView(this.ctx);
    document.body.append(this.overlay);
    addEventListener("resize", () => this.resize());
    addEventListener("keydown", (e) => this.key(e, true));
    addEventListener("keyup", (e) => this.key(e, false));
    document.addEventListener("visibilitychange", () => {
      if (document.hidden && this.mode === "race") this.setPaused(true);
    });
    this.resize();
    requestAnimationFrame((t) => this.frame(t));
  }

  /** The landing page's background: three random agents replaying L1 missions, forever. */
  showcase(): void {
    this.mode = "showcase";
    this.overlay.replaceChildren();
    this.hold = 0;
    this.acc = 0;
    loadLevel("L1").then(
      (file) => {
        if (this.mode !== "showcase") return;
        const ghosts = drawOpponents(file.agents, 3);
        this.race = new Race(file, Math.floor(Math.random() * file.missions.length), ghosts);
        this.view.reset();
        this.onShowcase(
          ghosts.map((g) => g.name),
          file.level,
        );
      },
      () => (this.race = null),
    );
  }

  start(
    file: LevelFile,
    mission: number,
    opponents: AgentRecord[],
    onEnd: (end: RaceEnd) => void,
  ): void {
    this.args = [file, mission, opponents];
    this.onEnd = onEnd;
    this.race = new Race(file, mission, opponents);
    this.pilot = new Pilot();
    this.mode = "race";
    this.clock = 0;
    this.acc = 0;
    this.hold = 0;
    this.view.reset();
    this.setPaused(false);
    this.touchControls();
  }

  private resize(): void {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(innerWidth * dpr);
    this.canvas.height = Math.round(innerHeight * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.view.resize(innerWidth, innerHeight);
  }

  private key(e: KeyboardEvent, down: boolean): void {
    if (this.mode !== "race") return;
    if (down && e.code === "Escape") return this.setPaused(!this.paused);
    if (down && e.code === "KeyR" && this.args) return this.start(...this.args, this.onEnd);
    const control = controlForKey(e.code);
    if (!control) return;
    e.preventDefault();
    if (down) this.pilot.held.add(control);
    else this.pilot.held.delete(control);
  }

  private setPaused(paused: boolean): void {
    this.paused = paused;
    this.pilot.held.clear();
    this.overlay.querySelector(".pause")?.remove();
    if (!paused || !this.args) return;
    const args = this.args;
    this.overlay.append(
      h(
        "div",
        { className: "pause panel", role: "dialog", ariaLabel: "Paused" },
        h("h2", {}, "Paused"),
        h("p", { className: "muted" }, "The pilots wait for you."),
        h(
          "div",
          { className: "actions" },
          button("Resume", () => this.setPaused(false), "btn primary"),
          button("Restart", () => this.start(...args, this.onEnd), "btn"),
          button(
            "Quit to menu",
            () => {
              this.mode = "idle";
              this.overlay.replaceChildren();
              this.onQuit();
            },
            "btn",
          ),
        ),
      ),
    );
  }

  private touchControls(): void {
    this.overlay.replaceChildren();
    const touch = matchMedia("(pointer: coarse)").matches;
    this.view.hint = touch
      ? "Slide to throttle, arrows to lean, round arrows to stop a spin"
      : "W and S throttle, A and D lean, Q and E stop a spin";
    if (!touch) return;
    const hold = (name: IconName, label: string, control: Control, className: string) => {
      const b = h(
        "button",
        { className: `touch ${className}`, type: "button", ariaLabel: label },
        icon(name),
      );
      const on = (e: Event) => {
        e.preventDefault();
        this.pilot.held.add(control);
      };
      const off = (e: Event) => {
        e.preventDefault();
        this.pilot.held.delete(control);
      };
      b.addEventListener("pointerdown", on);
      for (const ev of ["pointerup", "pointercancel", "pointerleave"]) b.addEventListener(ev, off);
      return b;
    };
    const slider = h("input", {
      type: "range",
      min: "0",
      max: "1000",
      value: "0",
      className: "throttle",
    });
    slider.addEventListener("input", () => this.pilot.setThrottle(Number(slider.value) / 1000));
    this.overlay.append(
      h("label", { className: "touch-left" }, h("span", {}, "Throttle"), slider),
      h(
        "div",
        { className: "touch-right" },
        h(
          "div",
          { className: "row" },
          hold("counterClockwise", "Thruster left", "rcsLeft", "small"),
          hold("clockwise", "Thruster right", "rcsRight", "small"),
        ),
        h(
          "div",
          { className: "row" },
          hold("left", "Lean left", "left", ""),
          hold("right", "Lean right", "right", ""),
        ),
      ),
      h(
        "button",
        {
          className: "touch pause-btn",
          type: "button",
          ariaLabel: "Pause",
          onclick: () => this.setPaused(true),
        },
        icon("pause"),
      ),
      h("div", { className: "rotate" }, "Turn your phone sideways to fly"),
    );
  }

  private frame(now: number): void {
    const realDt = Math.min(0.1, (now - (this.last || now)) / 1000);
    this.last = now;
    const race = this.race;
    if (race && this.mode === "showcase" && this.showcaseVisible) {
      this.acc += realDt;
      while (this.acc >= STEP) {
        this.acc -= STEP;
        race.step(null);
      }
      if (race.ghosts.every((g) => g.episode.done)) this.hold += realDt;
      if (this.hold > 2) this.showcase();
      this.view.draw(race, realDt, null, true);
    } else if (race && this.mode === "race") {
      if (!this.paused) this.advance(race, realDt);
      const countdown =
        this.clock < COUNTDOWN
          ? Math.ceil(COUNTDOWN - this.clock)
          : this.clock < COUNTDOWN + 0.6
            ? 0
            : null;
      this.view.draw(race, this.paused ? 0 : realDt, countdown);
    }
    requestAnimationFrame((t) => this.frame(t));
  }

  private advance(race: Race, realDt: number): void {
    this.clock += realDt;
    if (this.clock < COUNTDOWN) return;
    this.acc += realDt;
    while (this.acc >= STEP) {
      this.acc -= STEP;
      if (!race.player.episode.done) {
        race.step(this.pilot.action(STEP));
      } else {
        for (let i = 0; i < FAST_FORWARD; i++) race.step(null);
      }
    }
    if (race.finished) {
      this.hold += realDt;
      if (this.hold >= HOLD) {
        this.mode = "idle";
        this.overlay.replaceChildren();
        this.onEnd({ race });
      }
    }
  }
}
