// How the levels appear to players: difficulty modes with a plain description and a sea glyph.

export interface Mode {
  name: string;
  tagline: string;
  color: string;
  rank: number; // 1 (easiest) to 5
}

export const MODES: Record<string, Mode> = {
  L0: { name: "Easy", tagline: "Calm sea, the ship holds still", color: "#7bd88f", rank: 1 },
  L1: { name: "Normal", tagline: "The ship drifts and sways", color: "#62c6f2", rank: 2 },
  L2: { name: "Hard", tagline: "A rolling, heaving deck", color: "#f5b94a", rank: 3 },
  L3: { name: "Expert", tagline: "Gusty wind and a fickle engine", color: "#f2766b", rank: 4 },
  L4: { name: "Extreme", tagline: "Return from high up, falling fast", color: "#b79cff", rank: 5 },
};

export function modeOf(level: string): Mode {
  return MODES[level] ?? { name: level, tagline: "", color: "#f5f5f5", rank: 3 };
}

/** A small sea: the rougher the water, the harder the mode. Gust marks from Expert up. */
export function seaGlyph(rank: number): SVGSVGElement {
  const amplitude = [0, 2, 4.5, 6.5, 8.5][rank - 1] ?? 4;
  const wavelength = [40, 34, 28, 22, 18][rank - 1] ?? 28;
  const points: string[] = [];
  for (let x = 0; x <= 120; x += 2) {
    const y = 22 + amplitude * Math.sin((2 * Math.PI * x) / wavelength);
    points.push(`${x},${y.toFixed(1)}`);
  }
  const gusts =
    rank >= 4
      ? '<path class="gust" d="M72 1h22M88 5h20" />' +
        (rank === 5 ? '<path class="gust" d="M60 3h16" />' : "")
      : "";
  const template = document.createElement("template");
  template.innerHTML =
    `<svg class="sea" viewBox="0 0 120 32" preserveAspectRatio="none" aria-hidden="true">` +
    `<polyline class="wave" points="${points.join(" ")}" />${gusts}</svg>`;
  return template.content.firstChild as SVGSVGElement;
}
