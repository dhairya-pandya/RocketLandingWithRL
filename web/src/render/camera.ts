// Keeps the rocket and the ship in view, zooming in near touchdown (render/camera.py).
const MIN_SCALE = 0.4; // pixels per metre far away
const MAX_SCALE = 9.0; // at touchdown
const MARGIN = 1.6;
const SMOOTHING = 4.0; // 1/s

/** The part of the screen the action is framed in, in pixels. */
export interface Frame {
  x: number;
  y: number;
  w: number;
  h: number;
}

export class Camera {
  cx = 0;
  cy = 0;
  scale = MAX_SCALE;
  /** Frame the action in this part of the screen (the landing page's free space); null: all of it. */
  frame: Frame | null = null;
  private ready = false;

  constructor(
    public width: number,
    public height: number,
  ) {}

  /** Frame the deck and the given rocket positions; `zoom` lets small screens see more. */
  follow(points: [number, number][], deck: [number, number], dt: number, zoom = 1): void {
    const xs = [...points.map((p) => p[0]), deck[0]];
    const ys = [...points.map((p) => p[1]), deck[1]];
    const [lowX, highX, lowY, highY] = [
      Math.min(...xs),
      Math.max(...xs),
      Math.min(...ys),
      Math.max(...ys),
    ];
    const cx = (lowX + highX) / 2;
    const cy = (lowY + highY) / 2;
    const spanX = (highX - lowX) * MARGIN + 40;
    const spanY = (highY - lowY) * MARGIN + 40;
    const f = this.view();
    const fit = Math.min(f.w / spanX, f.h / spanY) * zoom;
    const scale = Math.min(MAX_SCALE * zoom, Math.max(MIN_SCALE, fit));
    if (!this.ready) {
      [this.cx, this.cy, this.scale, this.ready] = [cx, cy, scale, true];
      return;
    }
    const blend = 1 - Math.exp(-SMOOTHING * dt);
    this.cx += blend * (cx - this.cx);
    this.cy += blend * (cy - this.cy);
    this.scale += blend * (scale - this.scale);
  }

  reset(): void {
    this.ready = false;
  }

  private view(): Frame {
    return this.frame ?? { x: 0, y: 0, w: this.width, h: this.height };
  }

  /** World metres to screen pixels (screen y points down). */
  toScreen(x: number, y: number): [number, number] {
    const f = this.view();
    return [f.x + f.w / 2 + (x - this.cx) * this.scale, f.y + f.h / 2 - (y - this.cy) * this.scale];
  }

  toWorld(px: number, py: number): [number, number] {
    const f = this.view();
    return [
      (px - f.x - f.w / 2) / this.scale + this.cx,
      (f.y + f.h / 2 - py) / this.scale + this.cy,
    ];
  }
}
