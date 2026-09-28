"""Build RESULTS.md: a results table with intervals, learning curves, robustness plots and races.

What goes into the report is listed in a YAML file (see results.yaml at the repository root).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from matplotlib.figure import Figure  # the object API needs no window or backend

from rocketlander.cli.common import make_agent
from rocketlander.common.config import config_from_dict
from rocketlander.evaluation import EVAL_SEEDS, evaluate
from rocketlander.robustness import sweep, sweep_values
from rocketlander.stats import wilson_interval

LESSONS = "## Lessons learned"  # this section of an existing report is kept when rebuilding
ALGO_COLORS = {
    "reinforce": "tab:blue",
    "ppo": "tab:orange",
    "sac": "tab:green",
    "td3": "tab:red",
    "grpo": "tab:purple",
    "es": "tab:brown",
}


def line_style(agent: str) -> dict:
    """One colour per algorithm (from the checkpoint name); mixes dashed, curricula dotted."""
    algo, _, variant = Path(agent).stem.partition("_")
    dashes = {"mix": "--", "curriculum": ":"}.get(variant, "-")
    return {"color": ALGO_COLORS.get(algo, "tab:gray"), "linestyle": dashes}


@dataclass
class AgentEntry:
    name: str
    agent: str  # "pid", "random" or a checkpoint path
    run: str | None = None  # training run directory with metrics.csv, for learning curves
    trained_on: str | None = None  # panel of the learning-curve figure


@dataclass
class Race:
    level: str
    seed: int
    agents: list[str]


@dataclass
class ReportConfig:
    out: str
    figures: str
    levels: list[str]
    episodes: int
    agents: list[AgentEntry]
    robustness_level: str = "L2"
    robustness_episodes: int = 50
    robustness_agents: list[str] = field(default_factory=list)
    races: list[Race] = field(default_factory=list)
    workers: int = 8


def load_report_config(raw: dict) -> ReportConfig:
    raw = dict(raw)
    raw["agents"] = [config_from_dict(AgentEntry, a) for a in raw.get("agents", [])]
    raw["races"] = [config_from_dict(Race, r) for r in raw.get("races", [])]
    config = config_from_dict(ReportConfig, raw)
    names = {a.name for a in config.agents}
    wanted = set(config.robustness_agents) | {n for r in config.races for n in r.agents}
    if wanted - names:
        raise ValueError(f"Unknown agent names: {sorted(wanted - names)}")
    return config


def _success(job: tuple[str, str, int]) -> tuple[float, int]:
    agent, level, episodes = job
    summary = evaluate(make_agent(agent), level, EVAL_SEEDS[:episodes])
    return summary.success_rate, summary.episodes


def success_table(config: ReportConfig) -> dict[tuple[str, str], tuple[float, int]]:
    """(agent name, level) -> (success rate, episodes), evaluated in parallel processes."""
    keys = [(a.name, level) for a in config.agents for level in config.levels]
    paths = {a.name: a.agent for a in config.agents}
    jobs = [(paths[name], level, config.episodes) for name, level in keys]
    with ProcessPoolExecutor(config.workers) as pool:
        return dict(zip(keys, pool.map(_success, jobs), strict=True))


def read_curve(run: Path) -> tuple[list[float], list[float], list[float]]:
    """Env steps, wall-clock minutes and held-out success from a run's metrics.csv."""
    steps, minutes, success = [], [], []
    speed = None
    with open(run / "metrics.csv") as f:
        for row in csv.DictReader(f):
            if row["key"] == "train/steps_per_second":
                speed = float(row["value"])
            elif row["key"] == "eval/success_rate" and speed:
                step = int(row["step"])
                steps.append(step)
                minutes.append(step / speed / 60)
                success.append(float(row["value"]))
    return steps, minutes, success


def plot_learning_curves(config: ReportConfig, path: Path) -> bool:
    panels: dict[str, list[AgentEntry]] = defaultdict(list)
    for a in config.agents:
        if a.run and a.trained_on and (Path(a.run) / "metrics.csv").exists():
            panels[a.trained_on].append(a)
    if not panels:
        return False
    levels = [lv for lv in config.levels if lv in panels]
    fig = Figure(figsize=(4 * len(levels), 6.5))
    axes = fig.subplots(2, len(levels), squeeze=False)
    for col, level in enumerate(levels):
        for a in panels[level]:
            steps, minutes, success = read_curve(Path(a.run))
            style = line_style(a.agent)
            axes[0, col].plot([s / 1e6 for s in steps], success, label=a.name, **style)
            axes[1, col].plot(minutes, success, label=a.name, **style)
        axes[0, col].set_title(f"trained on {level}")
        axes[0, col].set_xlabel("environment steps (millions)")
        axes[1, col].set_xlabel("wall-clock minutes")
        axes[0, col].legend(fontsize=7)
    for ax in axes.flat:
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("held-out success")
    axes[1, 0].set_ylabel("held-out success")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return True


def plot_robustness(config: ReportConfig, path: Path) -> None:
    params = sorted(sweep_values())
    fig = Figure(figsize=(4 * len(params), 3.4))
    axes = fig.subplots(1, len(params), squeeze=False)
    entries = {a.name: a for a in config.agents}
    seeds = EVAL_SEEDS[: config.robustness_episodes]
    for name in config.robustness_agents:
        agent = make_agent(entries[name].agent)
        for ax, param in zip(axes[0], params, strict=True):
            results = sweep(agent, config.robustness_level, param, seeds)
            ax.plot([v for v, _ in results], [s.success_rate for _, s in results], "o-", label=name)
    for ax, param in zip(axes[0], params, strict=True):
        ax.set_title(f"{param} (on {config.robustness_level})")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("success")
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def cell(rate: float, episodes: int) -> str:
    low, high = wilson_interval(round(rate * episodes), episodes)
    return f"{100 * rate:.0f}% ({100 * low:.0f}–{100 * high:.0f})"


def markdown(config: ReportConfig, table, figures: dict[str, str], kept: str) -> str:
    lines = [
        "# Results",
        "",
        "Generated by `uv run rl-compare --report results.yaml`; edit that file to change what is "
        "compared. Success rates are over held-out start states (seeds 10000 and up, never used "
        "in training), with 95% Wilson intervals.",
        "",
        "## Success rate by level",
        "",
        "| Agent | " + " | ".join(config.levels) + " |",
        "|---|" + "---|" * len(config.levels),
    ]
    best = {lv: max(table[a.name, lv][0] for a in config.agents) for lv in config.levels}
    for a in config.agents:
        cells = []
        for lv in config.levels:
            rate, n = table[a.name, lv]
            text = cell(rate, n)
            cells.append(f"**{text}**" if rate == best[lv] and rate > 0 else text)
        lines.append(f"| {a.name} | " + " | ".join(cells) + " |")
    lines += ["", "Bold: best on that level (ties included)."]
    if "curves" in figures:
        lines += [
            "",
            "## Learning curves",
            "",
            "Held-out success during training (20 episodes per point), against environment steps "
            "(sample efficiency) and against wall-clock time on one CPU core. Runs trained several "
            "at a time, so the times are approximate.",
            "",
            f"![Learning curves]({figures['curves']})",
        ]
    if "robustness" in figures:
        lines += [
            "",
            "## Robustness",
            "",
            f"One physical parameter pinned per point, the rest sampled as on "
            f"{config.robustness_level}; {config.robustness_episodes} episodes per point.",
            "",
            f"![Robustness sweeps]({figures['robustness']})",
        ]
    races = [k for k in figures if k.startswith("race")]
    if races:
        lines += [
            "",
            "## Races",
            "",
            "Every agent flies the same start state, ship motion and wind.",
        ]
        for key in races:
            lines += ["", f"![{key}]({figures[key]})"]
    return "\n".join(lines) + "\n\n" + kept


def build_report(config: ReportConfig, root: Path = Path(".")) -> Path:
    from rocketlander.cli.compare import record_race  # pygame is only needed for races

    figure_dir = root / config.figures
    figure_dir.mkdir(parents=True, exist_ok=True)
    table = success_table(config)
    figures = {}
    if plot_learning_curves(config, figure_dir / "learning_curves.png"):
        figures["curves"] = f"{config.figures}/learning_curves.png"
    if config.robustness_agents:
        plot_robustness(config, figure_dir / "robustness.png")
        figures["robustness"] = f"{config.figures}/robustness.png"
    entries = {a.name: a.agent for a in config.agents}
    for race in config.races:
        name = f"race_{race.level}_seed{race.seed}.gif"
        record_race(
            [entries[n] for n in race.agents],
            race.level,
            race.seed,
            figure_dir / name,
            every=4,
            scale=0.4,
            labels=race.agents,
        )
        figures[f"race {race.level}, seed {race.seed}"] = f"{config.figures}/{name}"
    out = root / config.out
    old = out.read_text() if out.exists() else ""
    kept = old[old.index(LESSONS) :] if LESSONS in old else f"{LESSONS}\n\n(Your notes go here.)\n"
    out.write_text(markdown(config, table, figures, kept))
    return out
