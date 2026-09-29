// Flame, thruster puffs, wind streaks, explosions, smoke and splashes (render/effects.py).
import type { DeckState } from "../sim/ship";
import type { Camera } from "./camera";

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  size: number; // metres
  grow: number; // metres per second (smoke billows)
  gravity: number; // share of gravity it feels; negative rises
  rgb: [number, number, number];
  streak: boolean;
}

type Rgb = [number, number, number];
const FLAME: Rgb = [255, 190, 90];
const PUFF: Rgb = [220, 220, 230];
const WIND: Rgb = [200, 220, 240];
const SPARK: Rgb = [255, 200, 110];
const EMBER: Rgb = [255, 120, 60];
const SOOT: Rgb = [70, 70, 80];
const WATER: Rgb = [200, 225, 240];
const MAX_PARTICLES = 4000;

const uniform = (a: number, b: number) => a + Math.random() * (b - a);
const normal = (sd: number) =>
  sd * Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());

export class Particles {
  private list: Particle[] = [];

  reset(): void {
    this.list = [];
  }

  private add(p: Partial<Particle> & Pick<Particle, "x" | "y" | "vx" | "vy" | "life">): void {
    this.list.push({
      maxLife: p.life,
      size: 0.5,
      grow: 0,
      gravity: 0.3, // hot gas rises less than debris falls
      rgb: PUFF,
      streak: false,
      ...p,
    });
    if (this.list.length > MAX_PARTICLES) this.list.shift();
  }

  flame(exit: [number, number], dir: [number, number], throttle: number, dt: number): void {
    const n = Math.floor(throttle * 900 * dt);
    const base = Math.atan2(-dir[1], -dir[0]);
    for (let i = 0; i < n; i++) {
      const angle = base + normal(0.12);
      const speed = uniform(25, 45) * (0.5 + throttle);
      this.add({
        x: exit[0] + normal(0.3),
        y: exit[1] + normal(0.3),
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: uniform(0.15, 0.35),
        size: uniform(0.6, 1.4),
        rgb: FLAME,
      });
    }
  }

  puff(point: [number, number], outward: [number, number], dt: number): void {
    const n = Math.max(1, Math.floor(120 * dt));
    for (let i = 0; i < n; i++) {
      const s = uniform(8, 14);
      this.add({
        x: point[0],
        y: point[1],
        vx: outward[0] * s + normal(1.5),
        vy: outward[1] * s + normal(1.5),
        life: uniform(0.2, 0.4),
      });
    }
  }

  wind(cam: Camera, wind: number, dt: number): void {
    let expected = Math.abs(wind) * 6 * dt;
    while (expected > 0) {
      if (Math.random() < Math.min(1, expected)) {
        const [x, y] = cam.toWorld(uniform(0, cam.width), uniform(0, cam.height));
        this.add({ x, y, vx: wind * 6, vy: 0, life: 0.6, size: 0.3, rgb: WIND, streak: true });
      }
      expected -= 1;
    }
  }

  /** Sparks, embers and a column of rising smoke; `scale` shrinks it for ghosts. */
  explode(x: number, y: number, scale = 1): void {
    for (let i = 0; i < 160 * scale; i++) {
      const angle = uniform(0, 2 * Math.PI);
      const speed = uniform(15, 55);
      this.add({
        x: x + normal(1),
        y: y + normal(1),
        vx: Math.cos(angle) * speed,
        vy: Math.abs(Math.sin(angle)) * speed,
        life: uniform(0.3, 0.9),
        size: uniform(0.3, 0.6),
        gravity: 1,
        rgb: SPARK,
        streak: true,
      });
    }
    for (let i = 0; i < 90 * scale; i++) {
      const angle = uniform(0, 2 * Math.PI);
      const speed = uniform(4, 18);
      this.add({
        x: x + normal(2),
        y: y + normal(2),
        vx: Math.cos(angle) * speed,
        vy: Math.abs(Math.sin(angle)) * speed + 3,
        life: uniform(0.8, 2),
        size: uniform(0.4, 0.9),
        gravity: 0.6,
        rgb: EMBER,
      });
    }
    for (let i = 0; i < 50 * scale; i++) this.smoke([x + normal(3), y + normal(2)], 1.5);
  }

  /** A puff of dark smoke that rises and billows out. */
  smoke(point: [number, number], size = 0.8): void {
    this.add({
      x: point[0],
      y: point[1],
      vx: normal(1.5),
      vy: uniform(1, 4),
      life: uniform(1.5, 3),
      size: size * uniform(0.8, 1.6),
      grow: uniform(1.5, 3),
      gravity: -0.05,
      rgb: SOOT,
    });
  }

  /** Spray thrown up where something hits the water. */
  splash(point: [number, number], scale = 1): void {
    for (let i = 0; i < 40 * scale; i++) {
      const angle = Math.PI / 2 + normal(0.35);
      const speed = uniform(4, 14) * Math.sqrt(scale);
      this.add({
        x: point[0] + normal(0.8),
        y: point[1],
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: uniform(0.5, 1.1),
        size: uniform(0.3, 0.7),
        gravity: 1,
        rgb: WATER,
      });
    }
  }

  /** Drop particles that have drifted inside the ship's hull (below the deck, over the ship). */
  hideInsideHull(deck: DeckState): void {
    const tan = Math.tan(deck.angle);
    this.list = this.list.filter(
      (p) => Math.abs(p.x - deck.x) > deck.halfWidth || p.y >= deck.y + (p.x - deck.x) * tan,
    );
  }

  update(dt: number): void {
    for (const p of this.list) {
      p.vy -= 9.81 * p.gravity * dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.size += p.grow * dt;
      p.life -= dt;
    }
    this.list = this.list.filter((p) => p.life > 0);
  }

  draw(ctx: CanvasRenderingContext2D, cam: Camera): void {
    for (const p of this.list) {
      const fade = Math.min(1, Math.max(0, p.life / p.maxLife));
      const color = `rgb(${p.rgb[0]},${p.rgb[1]},${p.rgb[2]})`;
      const [sx, sy] = cam.toScreen(p.x, p.y);
      ctx.globalAlpha = p.grow > 0 ? 0.55 * fade : fade;
      if (p.streak) {
        const [tx, ty] = cam.toScreen(p.x - p.vx * 0.08, p.y - p.vy * 0.08);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(tx, ty);
        ctx.stroke();
      } else {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(sx, sy, Math.max(1, p.size * cam.scale * (0.4 + 0.6 * fade)), 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }
}
