// Phosphor icons (MIT), inlined at build time so they inherit the text colour.
import back from "@phosphor-icons/core/assets/bold/arrow-u-up-left-bold.svg?raw";
import clockwise from "@phosphor-icons/core/assets/bold/arrow-clockwise-bold.svg?raw";
import counterClockwise from "@phosphor-icons/core/assets/bold/arrow-counter-clockwise-bold.svg?raw";
import left from "@phosphor-icons/core/assets/bold/caret-left-bold.svg?raw";
import right from "@phosphor-icons/core/assets/bold/caret-right-bold.svg?raw";
import github from "@phosphor-icons/core/assets/bold/github-logo-bold.svg?raw";
import pause from "@phosphor-icons/core/assets/bold/pause-bold.svg?raw";
import play from "@phosphor-icons/core/assets/bold/play-bold.svg?raw";

const ICONS = { back, clockwise, counterClockwise, left, right, github, pause, play };

export type IconName = keyof typeof ICONS;

export function icon(name: IconName): HTMLSpanElement {
  const span = document.createElement("span");
  span.className = "icon";
  span.setAttribute("aria-hidden", "true");
  span.innerHTML = ICONS[name];
  return span;
}
