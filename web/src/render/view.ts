// Draws a race: the player's rocket, colour-coded ghost rockets with trails, effects and HUD.
import type { Race } from "../game/race";
import { bodyToWorld } from "../sim/physics";
import { Camera, type Frame } from "./camera";
import { Crash } from "./crash";
import { Particles } from "./effects";
import { drawBanner, drawLegend, drawPlayerPanel, drawTopBar, GOOD, BAD } from "./hud";
import { drawRocket, drawSea, drawShip, drawSky, nozzleGeometry } from "./scene";

export const GHOST_COLORS = ["#62c6f2", "#7bd88f", "#f279b8", "#b79cff", "#f5e36b", "#f2766b"];
const PLAYER = "#f5f5f5";
const SHAKE = 12; // pixels, when the player crashes
const calm = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

export class RaceView {
  readonly camera: Camera;
  private readonly particles = new Particles();
  private crashes = new Map<number, Crash>(); // by racer
  private shake = 0; // 1 right after the player crashes, fading to 0
  private lastTime = 0;
  /** Shown under the countdown; the race screen switches it to the touch controls on phones. */
  hint = "W and S throttle, A and D lean, Q and E stop a spin";
  /** Where the landing page leaves room for its replay. */
  showcaseFrame: Frame | null = null;

  constructor(private readonly ctx: CanvasRenderingContext2D) {
    this.camera = new Camera(ctx.canvas.width, ctx.canvas.height);
  }

  resize(width: number, height: number): void {
    this.camera.width = width;
    this.camera.height = height;
  }

  reset(): void {
    this.camera.reset();
    this.particles.reset();
    this.crashes = new Map();
    this.shake = 0;
    this.lastTime = 0;
  }

  colors(race: Race): string[] {
    return race.racers.map((_, i) =>
      i === 0 ? PLAYER : GHOST_COLORS[(i - 1) % GHOST_COLORS.length],
    );
  }

  /** `realDt`: seconds since the last frame, for effects that keep moving while paused. */
  draw(race: Race, realDt: number, countdown: number | null, showcase = false): void {
    const { ctx, camera } = this;
    const t = race.time;
    const dt = t > this.lastTime ? t - this.lastTime : realDt;
    this.lastTime = t;
    const deck = race.deck();
    const player = race.player.episode;
    const small = camera.width < 700;
    const focus = showcase ? race.ghosts[0].episode.rocket : player.rocket; // the landing page
    camera.frame = showcase ? this.showcaseFrame : null;
    camera.follow([[focus.x, focus.y]], [deck.x, deck.y], dt, small ? 0.75 : 1);

    race.racers.forEach((r, i) => {
      const e = r.episode;
      if (e.judgement.outcome === "crashed" && !this.crashes.has(i)) {
        const crash = new Crash(race.pose(r), e.mission.params, deck);
        const scale = i === 0 ? 1 : 0.5;
        if (crash.inWater) this.particles.splash([crash.x, crash.y], 3 * scale);
        this.particles.explode(crash.x, crash.y, crash.inWater ? 0.4 * scale : scale);
        this.crashes.set(i, crash);
        if (i === 0 && !showcase && !calm()) this.shake = 1;
      }
      if (e.done || e.rocket.fuel <= 0) return;
      if (e.rocket.throttle > 0.02) {
        const { exit, dir } = nozzleGeometry(e.rocket, e.mission.params, e.controls.gimbal);
        this.particles.flame(exit, dir, e.rocket.throttle, dt);
      }
      if (Math.abs(e.controls.rcs) > 0.1) {
        const side = (Math.sign(e.controls.rcs) * e.mission.params.width) / 2;
        const y = e.mission.params.length / 2 - 1;
        const [point, out] = bodyToWorld(e.rocket, [
          [side, y],
          [side * 2, y],
        ]);
        const norm = Math.hypot(out[0] - point[0], out[1] - point[1]);
        this.particles.puff(
          [point[0], point[1]],
          [(out[0] - point[0]) / norm, (out[1] - point[1]) / norm],
          dt,
        );
      }
    });
    const wind =
      player.mission.wind[Math.min(player.steps * 2, player.mission.wind.length - 1)] / 1000;
    this.particles.wind(camera, wind, dt);
    // Crashes and particles run in real time, so a crash plays out at normal speed even while
    // the ghosts fast-forward.
    for (const crash of this.crashes.values()) {
      const { splashes, trails } = crash.update(realDt, deck);
      splashes.forEach((p) => this.particles.splash(p, 0.6));
      trails.forEach((p) => Math.random() < 0.4 && this.particles.smoke(p, 0.5));
    }
    this.particles.update(realDt);
    this.particles.hideInsideHull(deck);

    const colors = this.colors(race);
    const alphaOf = (i: number) => (i === 0 ? 1 : 0.55);
    ctx.save();
    if (this.shake > 0) {
      const k = SHAKE * this.shake * this.shake;
      ctx.translate((Math.random() - 0.5) * 2 * k, (Math.random() - 0.5) * 2 * k);
      this.shake = Math.max(0, this.shake - realDt / 0.7);
    }
    drawSky(ctx, camera);
    drawSea(ctx, camera, t);
    this.particles.draw(ctx, camera);
    drawShip(ctx, camera, deck);
    for (const [i, crash] of this.crashes)
      if (!showcase || i > 0) crash.drawDebris(ctx, camera, colors[i], alphaOf(i));
    const first = showcase ? 1 : 0; // the title screen shows ghosts only
    for (let i = race.racers.length - 1; i >= first; i--) {
      // the player on top
      const r = race.racers[i];
      if (r.trail.length > 1) {
        ctx.strokeStyle = colors[i];
        ctx.globalAlpha = i === 0 ? 0.6 : 0.45;
        ctx.lineWidth = 1;
        ctx.beginPath();
        r.trail.forEach(([x, y], k) => {
          const [px, py] = camera.toScreen(x, y);
          if (k === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      if (this.crashes.has(i)) continue;
      const e = r.episode;
      drawRocket(
        ctx,
        camera,
        race.pose(r),
        e.mission.params,
        e.done ? null : e.controls,
        colors[i],
        alphaOf(i),
      );
    }
    for (const [i, crash] of this.crashes)
      if (!showcase || i > 0) crash.drawFire(ctx, camera, alphaOf(i));
    ctx.restore();

    if (showcase) return;
    const s = Math.min(1, camera.width / 900, camera.height / 560) * (small ? 1.1 : 1);
    drawTopBar(ctx, race, camera.width, s);
    drawPlayerPanel(ctx, race, s);
    drawLegend(ctx, race, colors, camera.width, s);
    if (countdown !== null) {
      drawBanner(
        ctx,
        countdown > 0 ? String(countdown) : "Go",
        this.hint,
        "#f5f5f5",
        camera.width,
        s,
      );
    } else if (player.done) {
      const j = player.judgement;
      const title =
        j.outcome === "landed"
          ? "Landed"
          : player.timeLimitReached
            ? "Out of time"
            : j.outcome === "crashed"
              ? "Crashed"
              : "Failed";
      const detail =
        j.outcome === "landed"
          ? `Touchdown at ${j.touchdownSpeed.toFixed(2)} m/s, score ${player.score.toFixed(0)}`
          : j.reason
            ? j.reason[0].toUpperCase() + j.reason.slice(1)
            : "The clock ran out";
      drawBanner(ctx, title, detail, j.outcome === "landed" ? GOOD : BAD, camera.width, s);
    }
  }
}
