// The world: sky, sea, drone ship and rockets (render/scene.py).
import {
  bodyToWorld,
  legTipsBody,
  SEA_LEVEL,
  type Controls,
  type RocketParams,
  type RocketState,
} from "../sim/physics";
import type { DeckState } from "../sim/ship";
import type { Camera } from "./camera";

// Dusk: a deep sky warming towards the horizon, a dark sea catching the last light.
const SKY: [number, string][] = [
  [0, "#05070f"],
  [0.55, "#16203a"],
  [0.85, "#3d3552"],
  [1, "#b7765a"],
];
const SEA = "#081320";
const WAVE = "#e0a077";
const HULL = "#1c1f26";
const DECK = "#c9ccd3";
const TARGET = "#ffb547";
const STARS = Array.from({ length: 70 }, (_, i) => [
  (Math.sin(i * 12.9898) * 43758.5453) % 1,
  (Math.sin(i * 78.233) * 12345.6789) % 1,
]).map(([a, b]) => [Math.abs(a), Math.abs(b) * 0.5, 0.3 + 0.5 * Math.abs(a * b)]);
const NOZZLE = "#3c3c46";
const HULL_DEPTH = 6.0;
const SHIP_NAME = "OCISLY"; // Of Course I Still Love You, painted on the hull

type Ctx = CanvasRenderingContext2D;

export function drawSky(ctx: Ctx, cam: Camera): void {
  const horizon = Math.min(cam.height, Math.max(cam.height * 0.3, cam.toScreen(0, SEA_LEVEL)[1]));
  const g = ctx.createLinearGradient(0, 0, 0, horizon);
  for (const [stop, color] of SKY) g.addColorStop(stop, color);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, cam.width, cam.height);
  for (const [x, y, alpha] of STARS) {
    ctx.fillStyle = `rgba(255,255,255,${alpha * (1 - 2 * y)})`;
    ctx.fillRect(x * cam.width, y * cam.height, 1.5, 1.5);
  }
}

export function drawSea(ctx: Ctx, cam: Camera, t: number): void {
  const [left] = cam.toWorld(0, 0);
  const [right] = cam.toWorld(cam.width, 0);
  ctx.beginPath();
  for (let i = 0; i <= 120; i++) {
    const x = left + ((right - left) * i) / 120;
    const y = SEA_LEVEL + 0.6 * Math.sin(0.15 * x + 1.3 * t) + 0.3 * Math.sin(0.41 * x - 2.1 * t);
    const [px, py] = cam.toScreen(x, y);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.lineTo(cam.width, cam.height);
  ctx.lineTo(0, cam.height);
  ctx.closePath();
  ctx.fillStyle = SEA;
  ctx.fill();
  ctx.strokeStyle = WAVE;
  ctx.globalAlpha = 0.7;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function deckToScreen(cam: Camera, d: DeckState, px: number, py: number): [number, number] {
  const c = Math.cos(d.angle);
  const s = Math.sin(d.angle);
  return cam.toScreen(px * c - py * s + d.x, px * s + py * c + d.y);
}

function polygon(ctx: Ctx, points: [number, number][], fill: string): void {
  ctx.beginPath();
  points.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
}

export function drawShip(ctx: Ctx, cam: Camera, d: DeckState): void {
  const w = d.halfWidth;
  const hull: [number, number][] = [
    [-w, 0],
    [w, 0],
    [w * 0.85, -HULL_DEPTH],
    [-w * 0.85, -HULL_DEPTH],
  ];
  polygon(
    ctx,
    hull.map(([x, y]) => deckToScreen(cam, d, x, y)),
    HULL,
  );
  const [a, b] = [deckToScreen(cam, d, -w, 0), deckToScreen(cam, d, w, 0)];
  ctx.strokeStyle = DECK;
  ctx.lineWidth = Math.max(2, 0.6 * cam.scale);
  ctx.beginPath();
  ctx.moveTo(...a);
  ctx.lineTo(...b);
  ctx.stroke();
  const size = 0.45 * HULL_DEPTH * cam.scale;
  if (size >= 6) {
    const [nx, ny] = deckToScreen(cam, d, 0, -HULL_DEPTH / 2);
    ctx.save();
    ctx.translate(nx, ny);
    ctx.rotate(-d.angle);
    ctx.font = `700 ${size}px Geist, system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "rgba(245,245,245,0.85)";
    ctx.fillText(SHIP_NAME, 0, 0);
    ctx.restore();
  }
  const [cx, cy] = deckToScreen(cam, d, 0, 0);
  ctx.strokeStyle = TARGET;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.ellipse(cx, cy, Math.max(3, 4 * cam.scale), 2, 0, 0, 2 * Math.PI);
  ctx.stroke();
}

/** Nozzle exit point (world) and unit thrust direction. */
export function nozzleGeometry(r: RocketState, p: RocketParams, gimbal: number) {
  const dir: [number, number] = [-Math.sin(r.theta + gimbal), Math.cos(r.theta + gimbal)];
  const [base] = bodyToWorld(r, [[0, -p.length / 2]]);
  return { exit: [base[0] - dir[0] * 0.8, base[1] - dir[1] * 0.8] as [number, number], dir, base };
}

export function drawRocket(
  ctx: Ctx,
  cam: Camera,
  r: RocketState,
  p: RocketParams,
  controls: Controls | null,
  body: string,
  alpha = 1,
): void {
  const hl = p.length / 2;
  const hw = p.width / 2;
  const px = (pts: number[][]) => bodyToWorld(r, pts).map(([x, y]) => cam.toScreen(x, y));
  ctx.save();
  ctx.globalAlpha = alpha;
  const { exit, dir, base } = nozzleGeometry(r, p, controls?.gimbal ?? 0);
  const side = [dir[1] * 0.7, -dir[0] * 0.7];
  const nozzle = [
    [base[0] + side[0] * 0.6, base[1] + side[1] * 0.6],
    [base[0] - side[0] * 0.6, base[1] - side[1] * 0.6],
    [exit[0] - side[0], exit[1] - side[1]],
    [exit[0] + side[0], exit[1] + side[1]],
  ].map(([x, y]) => cam.toScreen(x, y));
  polygon(ctx, nozzle, NOZZLE);
  const hips = px([
    [-hw, -hl + 3],
    [hw, -hl + 3],
  ]);
  const tips = px(legTipsBody(p));
  ctx.strokeStyle = "#afafb9";
  ctx.lineWidth = Math.max(2, 0.4 * cam.scale);
  for (let i = 0; i < 2; i++) {
    ctx.beginPath();
    ctx.moveTo(...hips[i]);
    ctx.lineTo(...tips[i]);
    ctx.stroke();
  }
  polygon(
    ctx,
    px([
      [-hw, -hl],
      [hw, -hl],
      [hw, hl],
      [-hw, hl],
    ]),
    body,
  );
  polygon(
    ctx,
    px([
      [-hw, hl],
      [hw, hl],
      [0, hl + 2.5],
    ]),
    "#9696a0",
  );
  polygon(
    ctx,
    px([
      [-hw, hl - 4],
      [hw, hl - 4],
      [hw, hl - 3],
      [-hw, hl - 3],
    ]),
    "#9696a0",
  );
  ctx.restore();
}
