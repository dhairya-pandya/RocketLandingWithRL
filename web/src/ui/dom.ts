// Tiny helpers for building screens without a framework.

type Props<K extends keyof HTMLElementTagNameMap> = Partial<
  Omit<HTMLElementTagNameMap[K], "style">
> & {
  className?: string;
  style?: string; // inline CSS, e.g. "--mode: #5adc78"
};

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Props<K> = {},
  ...children: (Node | string | null | false)[]
): HTMLElementTagNameMap[K] {
  const { style, ...rest } = props;
  const el = Object.assign(document.createElement(tag), rest);
  if (style) el.style.cssText = style;
  for (const child of children) if (child) el.append(child);
  return el;
}

export function button(label: string, onClick: () => void, className = "btn"): HTMLButtonElement {
  return h("button", { className, onclick: onClick, type: "button" }, label);
}

export function show(root: HTMLElement, ...screen: Node[]): void {
  root.replaceChildren(...screen);
  root.hidden = false;
}
