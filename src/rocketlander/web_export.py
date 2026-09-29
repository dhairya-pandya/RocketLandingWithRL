"""Export missions and every agent's flights for the web game (web/public/missions/<level>.json).

A mission is a held-out seed of a level: its start state, rocket and ship parameters, and its wind,
rounded to mm/s per physics step. Each agent flies the mission with that exact rounded wind and
with its actions rounded to thousandths, so the browser can re-simulate every flight exactly from
the stored integers. What should be exported is listed in web-export.yaml.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from rocketlander.agents.base import start_episode
from rocketlander.cli.common import make_agent
from rocketlander.common.checkpoint import git_commit
from rocketlander.common.config import config_from_dict
from rocketlander.envs.levels import available_levels, get_level
from rocketlander.envs.rocket_env import PHYSICS_DT, PHYSICS_STEPS_PER_ACTION, RocketLanderEnv
from rocketlander.envs.wind import step_wind
from rocketlander.evaluation import EVAL_SEEDS

SCHEMA_VERSION = 1


@dataclass
class ExportAgent:
    id: str
    name: str
    agent: str  # "pid", "mpc" or a checkpoint path; "{level}" is replaced by the level name
    levels: list[str] = field(default_factory=available_levels)


@dataclass
class ExportConfig:
    out: str  # directory for <level>.json
    fixtures: str  # directory for the parity-test references
    missions: int  # held-out seeds per level, from EVAL_SEEDS
    levels: list[str]
    agents: list[ExportAgent]
    workers: int = 5


def load_export_config(raw: dict) -> ExportConfig:
    raw = dict(raw)
    raw["agents"] = [config_from_dict(ExportAgent, a) for a in raw.get("agents", [])]
    config = config_from_dict(ExportConfig, raw)
    unknown = set(config.levels) - set(available_levels())
    if unknown:
        raise ValueError(f"Unknown levels: {sorted(unknown)}")
    return config


def mission_wind(level: str, seed: int) -> list[int]:
    """The level's natural wind for this seed, in mm/s, for every physics step of the time limit."""
    env = RocketLanderEnv(level=level)
    env.reset(seed=seed)
    state = env.get_state()
    wind, winds = state.wind, []
    for _ in range(env.max_steps * PHYSICS_STEPS_PER_ACTION):
        wind = step_wind(wind, state.setup.wind_model, PHYSICS_DT, env.np_random)
        winds.append(round(wind.speed * 1000))
    return winds


def mission(level: str, seed: int) -> dict:
    env = RocketLanderEnv(level=level)
    env.reset(seed=seed)
    state = env.get_state()
    return {
        "seed": seed,
        "start": asdict(state.rocket),
        "params": asdict(state.setup.params),
        "ship": asdict(state.setup.ship),
        "wind": mission_wind(level, seed),
    }


def fly(agent, level: str, seed: int, wind_mm: list[int], trace: bool = False) -> dict:
    """One flight with the mission's rounded wind and thousandth-rounded actions."""
    env = RocketLanderEnv(level=level)
    obs, _ = env.reset(seed=seed, options={"wind": [w / 1000 for w in wind_mm]})
    start_episode(agent)
    actions, states, score, done = [], [], 0.0, False
    while not done:
        ints = np.round(np.clip(np.asarray(agent.act(obs), dtype=np.float64), -1, 1) * 1000)
        obs, _, terminated, truncated, info = env.step(ints / 1000)
        actions += [int(v) for v in ints]
        score += info["task_reward"]
        done = terminated or truncated
        if trace:
            states.append(asdict(env.get_state().rocket))
    flight = {
        "actions": actions,
        "outcome": info["outcome"],
        "reason": info["reason"],
        "score": score,
        "touchdownSpeed": info["touchdown_speed"],
        "fuelUsed": info["fuel_used"],
        "steps": len(actions) // 3,
    }
    return flight | ({"states": states} if trace else {})


def export_level(config: ExportConfig, level: str) -> tuple[dict, dict]:
    """The level's mission file and its parity-test reference (the PID on the first mission)."""
    torch.set_num_threads(1)
    seeds = EVAL_SEEDS[: config.missions]
    missions = [mission(level, seed) for seed in seeds]
    agents = []
    for entry in config.agents:
        if level not in entry.levels:
            continue
        agent = make_agent(entry.agent.format(level=level))
        flights = [fly(agent, level, m["seed"], m["wind"]) for m in missions]
        agents.append(
            {
                "id": entry.id,
                "name": entry.name,
                "flights": flights,
            }
        )
    agents.sort(key=lambda a: -float(np.mean([f["score"] for f in a["flights"]])))  # best first
    info = get_level(level)
    data = {
        "schemaVersion": SCHEMA_VERSION,
        "level": level,
        "description": info.description,
        "maxSteps": RocketLanderEnv(level=level).max_steps,
        "gitCommit": git_commit(),
        "missions": missions,
        "agents": agents,
    }
    reference = fly(make_agent("pid"), level, seeds[0], missions[0]["wind"], trace=True)
    return data, {"level": level, "mission": 0} | reference


def export(config: ExportConfig) -> list[Path]:
    out, fixtures = Path(config.out), Path(config.fixtures)
    out.mkdir(parents=True, exist_ok=True)
    fixtures.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(config.workers) as pool:
        results = list(pool.map(export_level, [config] * len(config.levels), config.levels))
    written, index = [], []
    for level, (data, reference) in zip(config.levels, results, strict=True):
        path = out / f"{level}.json"
        path.write_text(json.dumps(data, separators=(",", ":")))
        (fixtures / f"{level}-reference.json").write_text(json.dumps(reference))
        written.append(path)
        index.append(
            {
                "level": level,
                "description": data["description"],
                "missions": len(data["missions"]),
                "agents": [{"id": a["id"], "name": a["name"]} for a in data["agents"]],
            }
        )
    path = out / "index.json"  # what the level-select screen shows before a level is loaded
    path.write_text(json.dumps({"schemaVersion": SCHEMA_VERSION, "levels": index}, indent=1))
    return [*written, path]
