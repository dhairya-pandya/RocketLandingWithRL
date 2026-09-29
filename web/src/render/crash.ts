// A crash: a fireball and shockwave, and the rocket breaking into pieces that tumble, bounce and
// come to rest on the deck or sink into the sea. Runs in real time, independent of the race.
import {
  bodyToWorld,
  GRAVITY,
  SEA_LEVEL,
  type RocketParams,
  type RocketState,
} from "../sim/physics";
import { onDeck, surfaceHeight, type DeckState } from "../sim/ship";
import type { Camera } from "./camera";

const METAL = "#8a8f99";
const FIRE_TIME = 0.9; // seconds the fireball lasts
const SINK_TIME = 1.5; // seconds a piece takes to disappear under water
const REST_SPEED = 2.5; // m/s: slower than this against the deck, a piece stops bouncing

export interface Debris {
  x: number;
  y: number;
  vx: number;
  vy: number;
  angle: number; // the piece's long axis, counter-clockwise from vertical
  spin: number;
  length: number;
  width: number;
  metal: boolean; // legs and nozzle are grey; body pieces take the rocket's colour
  bounces: number;
  resting: boolean;
  along: number; // while resting: distance from the deck centre along the deck
  sunk: number; // 0 afloat or flying, rising to 1 as it disappears
}

export interface CrashEvents {
  splashes: [number, number][]; // where pieces hit the water this frame
  trails: [number, number][]; // burning pieces still in the air
}

export class Crash {
  readonly debris: Debris[];
  readonly x: number;
  readonly y: number;
  readonly inWater: boolean;
  private readonly size: number;
  age = 0;

  constructor(r: RocketState, p: RocketParams, deck: DeckState, random = Math.random) {
    const hl = p.length / 2;
    const hw = p.width / 2;
    const uniform = (a: number, b: number) => a + random() * (b - a);
    [this.x, this.y] = [r.x, r.y];
    this.inWater = !onDeck(deck, r.x);
    this.size = 3 + 0.3 * p.length;
    // Pieces in the rocket's frame: [x, y, length, width, metal].
    const segment = p.length / 4;
    const legLength = Math.hypot(p.leg_half_span - hw, p.leg_drop + 3);
    const pieces: [number, number, number, number, boolean][] = [
      ...[0, 1, 2, 3].map((i): [number, number, number, number, boolean] => [
        0,
        -hl + segment * (i + 0.5),
        segment,
        p.width,
        false,
      ]),
      [0, hl + 1, 2.5, p.width * 0.8, false], // nose
      [-(p.leg_half_span + hw) / 2, -hl - p.leg_drop / 2 + 1, legLength, 0.4, true],
      [(p.leg_half_span + hw) / 2, -hl - p.leg_drop / 2 + 1, legLength, 0.4, true],
      [0, -hl - 0.4, 1.2, 1.4, true], // nozzle
    ];
    const world = bodyToWorld(
      r,
      pieces.map(([x, y]) => [x, y]),
    );
    this.debris = pieces.map(([, , length, width, metal], i) => {
      const [x, y] = world[i];
      const dx = x - r.x;
      const dy = y - r.y;
      const d = Math.hypot(dx, dy) || 1;
      const kick = uniform(6, 16);
      return {
        x,
        y,
        vx: 0.25 * r.vx + (dx / d) * kick + uniform(-4, 4),
        vy: 0.25 * r.vy + Math.abs(dy / d) * kick * 0.5 + uniform(4, 12),
        angle: r.theta + uniform(-0.3, 0.3),
        spin: uniform(-8, 8),
        length,
        width,
        metal,
        bounces: 0,
        resting: false,
        along: 0,
        sunk: 0,
      };
    });
  }

  get finished(): boolean {
    return this.age > FIRE_TIME && this.debris.every((p) => p.resting || p.sunk >= 1);
  }

  /** The fireball's radius (metres) and opacity. */
  fireball(): { radius: number; alpha: number } {
    const size = this.inWater ? this.size * 0.6 : this.size;
    const alpha = this.age < FIRE_TIME ? Math.pow(1 - this.age / FIRE_TIME, 1.2) : 0;
    return { radius: size * (1 - Math.exp(-7 * this.age)), alpha };
  }

  update(dt: number, deck: DeckState): CrashEvents {
    this.age += dt;
    const events: CrashEvents = { splashes: [], trails: [] };
    const c = Math.cos(deck.angle);
    const s = Math.sin(deck.angle);
    for (const p of this.debris) {
      const r = p.width / 2;
      if (p.sunk > 0) {
        p.sunk = Math.min(1, p.sunk + dt / SINK_TIME);
        p.x += p.vx * dt;
        p.vx *= Math.exp(-2 * dt);
        p.y -= 1.2 * dt;
        continue;
      }
      if (p.resting) {
        // Ride the deck: lie along it, a piece's half-width above the surface.
        p.x = deck.x + p.along * c - r * s;
        p.y = deck.y + p.along * s + r * c;
        const lying = deck.angle + (p.angle - deck.angle > 0 ? Math.PI / 2 : -Math.PI / 2);
        p.angle += (lying - p.angle) * Math.min(1, 10 * dt);
        continue;
      }
      p.vy -= GRAVITY * dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.angle += p.spin * dt;
      const surface = surfaceHeight(deck, p.x);
      if (onDeck(deck, p.x) && p.y - r < surface && p.y > surface - 3) {
        p.y = surface + r;
        const relative = p.vy - deck.vy;
        if (Math.abs(relative) < REST_SPEED || p.bounces >= 2) {
          p.resting = true;
          p.along = (p.x - deck.x) / c;
          p.angle = deck.angle + ((((p.angle - deck.angle) % Math.PI) + Math.PI) % Math.PI);
        } else {
          p.vy = deck.vy - 0.35 * relative;
          p.vx *= 0.6;
          p.spin *= -0.5;
          p.bounces += 1;
        }
      } else if (p.y < SEA_LEVEL) {
        events.splashes.push([p.x, SEA_LEVEL]);
        p.sunk = 1e-3;
      } else if (this.age < 2) {
        events.trails.push([p.x, p.y]);
      }
    }
    return events;
  }

  drawDebris(ctx: CanvasRenderingContext2D, cam: Camera, color: string, alpha: number): void {
    for (const p of this.debris) {
      if (p.sunk >= 1) continue;
      const [sx, sy] = cam.toScreen(p.x, p.y);
      ctx.save();
      ctx.globalAlpha = alpha * (1 - p.sunk);
      ctx.translate(sx, sy);
      ctx.rotate(-p.angle);
      const w = Math.max(1.5, p.width * cam.scale);
      const h = Math.max(2, p.length * cam.scale);
      ctx.fillStyle = p.metal ? METAL : color;
      ctx.fillRect(-w / 2, -h / 2, w, h);
      ctx.fillStyle = "rgba(0,0,0,0.35)"; // scorched
      ctx.fillRect(-w / 2, -h / 2, w, h * 0.4);
      ctx.restore();
    }
  }

  drawFire(ctx: CanvasRenderingContext2D, cam: Camera, alpha: number): void {
    const { radius, alpha: a } = this.fireball();
    if (a <= 0) return;
    const [sx, sy] = cam.toScreen(this.x, this.y);
    const r = Math.max(4, radius * cam.scale);
    const g = ctx.createRadialGradient(sx, sy, 0, sx, sy, r);
    g.addColorStop(0, `rgba(255,248,230,${a * alpha})`);
    g.addColorStop(0.35, `rgba(255,181,71,${0.9 * a * alpha})`);
    g.addColorStop(1, "rgba(242,118,107,0)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, 2 * Math.PI);
    ctx.fill();
    if (this.inWater || this.age > 0.5) return;
    ctx.strokeStyle = `rgba(255,255,255,${0.6 * (1 - this.age / 0.5) * alpha})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(sx, sy, r * (0.5 + 5 * this.age), 0, 2 * Math.PI);
    ctx.stroke();
  }
}
