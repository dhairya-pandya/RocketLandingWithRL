import json
from pathlib import Path

import numpy as np
import pytest

from rocketlander.cli import export_web
from rocketlander.envs.rocket_env import PHYSICS_STEPS_PER_ACTION, RocketLanderEnv
from rocketlander.evaluation import EVAL_SEEDS
from rocketlander.web_export import (
    SCHEMA_VERSION,
    export,
    load_export_config,
    mission_wind,
)

CHECKPOINT = str(Path(__file__).resolve().parents[1] / "checkpoints" / "ppo_L3.pt")
TINY = {
    "out": "missions",
    "fixtures": "fixtures",
    "missions": 1,
    "levels": ["L3"],
    "workers": 1,
    "agents": [
        {
            "id": "pid",
            "name": "PID",
            "agent": "pid",
        },
        {
            "id": "ppo",
            "name": "PPO",
            "agent": CHECKPOINT,
        },
    ],
}


def test_a_replayed_wind_is_what_the_physics_feels():
    env = RocketLanderEnv(level="L3")
    track = [0.5 * k for k in range(env.max_steps * PHYSICS_STEPS_PER_ACTION)]
    env.reset(seed=1, options={"wind": track})
    for step in range(3):
        env.step(np.zeros(3))
        assert env.get_state().wind.speed == track[(step + 1) * PHYSICS_STEPS_PER_ACTION - 1]


def test_mission_wind_matches_the_natural_wind_to_the_millimetre():
    level, seed = "L3", EVAL_SEEDS[0]
    env = RocketLanderEnv(level=level)
    env.reset(seed=seed)
    natural = []
    for _ in range(5):
        env.step(np.zeros(3))
        natural.append(env.get_state().wind.speed)
    exported = mission_wind(level, seed)
    assert len(exported) == env.max_steps * PHYSICS_STEPS_PER_ACTION
    assert [w / 1000 for w in exported[1:10:2]] == pytest.approx(natural, abs=5e-4)


@pytest.fixture
def exported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    export(load_export_config(TINY))
    return tmp_path


def test_export_writes_the_documented_schema(exported):
    data = json.loads((exported / "missions" / "L3.json").read_text())
    assert data["schemaVersion"] == SCHEMA_VERSION and data["level"] == "L3"
    mission = data["missions"][0]
    assert set(mission) == {"seed", "start", "params", "ship", "wind"}
    assert all(isinstance(w, int) for w in mission["wind"])
    flight = data["agents"][0]["flights"][0]
    assert len(flight["actions"]) == 3 * flight["steps"]
    assert all(-1000 <= a <= 1000 for a in flight["actions"])
    scores = [np.mean([f["score"] for f in a["flights"]]) for a in data["agents"]]
    assert scores == sorted(scores, reverse=True)  # best average score first
    index = json.loads((exported / "missions" / "index.json").read_text())
    assert {a["id"] for a in index["levels"][0]["agents"]} == {"pid", "ppo"}
    reference = json.loads((exported / "fixtures" / "L3-reference.json").read_text())
    assert len(reference["states"]) == reference["steps"]


def test_stored_actions_reproduce_every_stored_result(exported):
    data = json.loads((exported / "missions" / "L3.json").read_text())
    mission = data["missions"][0]
    for agent in data["agents"]:
        flight = agent["flights"][0]
        env = RocketLanderEnv(level="L3")
        env.reset(seed=mission["seed"], options={"wind": [w / 1000 for w in mission["wind"]]})
        score, actions = 0.0, np.array(flight["actions"]).reshape(-1, 3) / 1000
        for action in actions:
            *_, info = env.step(action)
            score += info["task_reward"]
        assert info["outcome"] == flight["outcome"] and score == pytest.approx(flight["score"])


def test_export_is_deterministic(exported, tmp_path_factory, monkeypatch):
    first = (exported / "missions" / "L3.json").read_bytes()
    again = tmp_path_factory.mktemp("again")
    monkeypatch.chdir(again)
    export(load_export_config(TINY))
    assert (again / "missions" / "L3.json").read_bytes() == first


@pytest.mark.parametrize("change", [{"levels": ["L9"]}, {"agents": [{"id": "x"}]}])
def test_bad_configs_are_rejected(change):
    with pytest.raises((ValueError, TypeError)):
        load_export_config(TINY | change)


def test_cli_reports_a_bad_config_as_a_usage_error(tmp_path, capsys):
    (tmp_path / "bad.yaml").write_text(
        "out: o\nfixtures: f\nmissions: 1\nlevels: [L9]\nagents: []\n"
    )
    with pytest.raises(SystemExit):
        export_web.main(["--config", str(tmp_path / "bad.yaml")])
    assert "L9" in capsys.readouterr().err
