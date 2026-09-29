// Heads-up display: the player's readouts, the racers' legend and the outcome banner.
import { MAX_HORIZONTAL_SPEED, MAX_SPIN, MAX_TILT, MAX_VERTICAL_SPEED } from "../sim/landing";
import type { Race, Racer } from "../game/race";
import { modeOf } from "../ui/modes";

export const TEXT = "#f5f5f5";
export const GOOD = "#7bd88f";
export const BAD = "#f2766b";
const MUTED = "#9b9b9b";
const PLUME = "#ffb547";
const TRACK = "#272727";
const SANS = "Geist, system-ui, sans-serif";
const MONO = "'Geist Mono', ui-monospace, monospace";

type Ctx = CanvasRenderingContext2D;

function panel(ctx: Ctx, x: number, y: number, w: number, h: number, s: number): void {
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.strokeStyle = "#313131";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 12 * s);
  ctx.fill();
  ctx.stroke();
}

const capital = (text: string) => text[0].toUpperCase() + text.slice(1);

export function status(r: Racer): { text: string; color: string } {
  const e = r.episode;
  if (e.judgement.outcome === "landed")
    return { text: `Landed ${e.judgement.touchdownSpeed.toFixed(1)} m/s`, color: GOOD };
  if (e.judgement.outcome !== "in_flight") return { text: capital(e.judgement.reason), color: BAD };
  if (e.timeLimitReached) return { text: "Out of time", color: BAD };
  const altitude = e.rocket.y - e.deck().y;
  return { text: `${altitude.toFixed(0)} m up`, color: MUTED };
}

export function drawPlayerPanel(ctx: Ctx, race: Race, s: number): void {
  const e = race.player.episode;
  const r = e.rocket;
  const d = e.deck();
  const p = e.mission.params;
  const altitude = r.y - d.y - p.length / 2 - p.leg_drop;
  const rows: [string, string, boolean | null][] = [
    ["Altitude", `${altitude.toFixed(1)} m`, null],
    ["Descent", `${(r.vy - d.vy).toFixed(1)} m/s`, Math.abs(r.vy - d.vy) <= MAX_VERTICAL_SPEED],
    ["Drift", `${(r.vx - d.vx).toFixed(1)} m/s`, Math.abs(r.vx - d.vx) <= MAX_HORIZONTAL_SPEED],
    [
      "Tilt",
      `${(((r.theta - d.angle) * 180) / Math.PI).toFixed(1)}°`,
      Math.abs(r.theta - d.angle) <= MAX_TILT,
    ],
    ["Spin", `${((r.omega * 180) / Math.PI).toFixed(1)}°/s`, Math.abs(r.omega) <= MAX_SPIN],
  ];
  const x = 16;
  const y = 16;
  const w = 200 * s;
  panel(ctx, x, y, w, 188 * s, s);
  rows.forEach(([label, value, ok], i) => {
    const ry = y + (28 + 22 * i) * s;
    ctx.font = `500 ${12 * s}px ${SANS}`;
    ctx.fillStyle = MUTED;
    ctx.fillText(label, x + 16 * s, ry);
    ctx.font = `500 ${14 * s}px ${MONO}`;
    ctx.fillStyle = ok === null ? TEXT : ok ? GOOD : BAD;
    ctx.textAlign = "right";
    ctx.fillText(value, x + w - 16 * s, ry);
    ctx.textAlign = "left";
  });
  const bar = (label: string, fraction: number, color: string, row: number) => {
    const by = y + (140 + 24 * row) * s;
    ctx.font = `500 ${12 * s}px ${SANS}`;
    ctx.fillStyle = MUTED;
    ctx.fillText(label, x + 16 * s, by);
    const bx = x + 80 * s;
    const bw = w - 96 * s;
    ctx.fillStyle = TRACK;
    ctx.beginPath();
    ctx.roundRect(bx, by - 6 * s, bw, 4 * s, 2 * s);
    ctx.fill();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(bx, by - 6 * s, bw * Math.min(1, Math.max(0, fraction)), 4 * s, 2 * s);
    ctx.fill();
  };
  const fuel = r.fuel / p.initial_fuel;
  bar("Fuel", fuel, fuel > 0.2 ? GOOD : BAD, 0);
  bar("Throttle", r.throttle, PLUME, 1);
}

export function drawTopBar(ctx: Ctx, race: Race, width: number, s: number): void {
  const e = race.player.episode;
  const wind = e.mission.wind[Math.min(e.steps * 2, e.mission.wind.length - 1)] / 1000;
  const mode = modeOf(race.file.level);
  const parts: [string, string, string][] = [
    [mode.name, mode.color, SANS],
    [`Mission ${race.missionIndex + 1}`, TEXT, SANS],
    [`${e.time.toFixed(1)} s`, TEXT, MONO],
    [`Wind ${wind >= 0 ? "→" : "←"} ${Math.abs(wind).toFixed(1)} m/s`, TEXT, MONO],
  ];
  const gap = 20 * s;
  const widths = parts.map(([text, , font]) => {
    ctx.font = `600 ${14 * s}px ${font}`;
    return ctx.measureText(text).width;
  });
  const total = widths.reduce((a, b) => a + b, 0) + gap * (parts.length - 1) + 32 * s;
  let cx = (width - total) / 2;
  panel(ctx, cx, 16, total, 36 * s, s);
  cx += 16 * s;
  parts.forEach(([text, color, font], i) => {
    ctx.font = `600 ${14 * s}px ${font}`;
    ctx.fillStyle = color;
    ctx.fillText(text, cx, 16 + 23 * s);
    cx += widths[i] + gap;
  });
}

export function drawLegend(ctx: Ctx, race: Race, colors: string[], width: number, s: number): void {
  const w = 240 * s;
  const x = width - w - 16;
  panel(ctx, x, 16, w, (16 + 24 * race.racers.length) * s, s);
  race.racers.forEach((r, i) => {
    const y = 16 + (28 + 24 * i) * s;
    ctx.fillStyle = colors[i];
    ctx.beginPath();
    ctx.arc(x + 20 * s, y - 4 * s, 4 * s, 0, 2 * Math.PI);
    ctx.fill();
    ctx.font = `600 ${13 * s}px ${SANS}`;
    ctx.fillStyle = TEXT;
    ctx.fillText(r.label, x + 32 * s, y);
    const st = status(r);
    ctx.font = `500 ${12 * s}px ${SANS}`;
    ctx.fillStyle = st.color;
    ctx.textAlign = "right";
    ctx.fillText(st.text, x + w - 16 * s, y);
    ctx.textAlign = "left";
  });
}

export function drawBanner(
  ctx: Ctx,
  title: string,
  detail: string,
  color: string,
  width: number,
  s: number,
): void {
  const w = 400 * s;
  const top = 72 * s;
  panel(ctx, (width - w) / 2, top, w, 104 * s, s);
  ctx.textAlign = "center";
  ctx.font = `600 ${40 * s}px ${SANS}`;
  ctx.fillStyle = color;
  ctx.fillText(title, width / 2, top + 52 * s);
  ctx.font = `500 ${14 * s}px ${SANS}`;
  ctx.fillStyle = MUTED;
  ctx.fillText(detail, width / 2, top + 82 * s);
  ctx.textAlign = "left";
}
