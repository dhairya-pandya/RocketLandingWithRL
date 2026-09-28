import numpy as np
import pytest

from rocketlander.common.curriculum import LevelSchedule, parse_level
from rocketlander.common.vec_env import VecEnv
from rocketlander.envs.rocket_env import RocketLanderEnv


def test_plain_level_has_no_schedule():
    assert parse_level("L2", seed=0) == ("L2", None)


def test_mix_evaluates_on_the_last_level_and_samples_all():
    level, schedule = parse_level("mix:L0,L3", seed=0)
    assert level == "L3"
    assert {schedule.sample() for _ in range(50)} == {"L0", "L3"}


@pytest.mark.parametrize("spec", ["L9", "mix:L0,L9", "mix:", "ladder:L0,L1"])
def test_bad_specs_raise_value_error(spec):
    with pytest.raises(ValueError):
        parse_level(spec, seed=0)


def test_curriculum_unlocks_only_after_a_full_successful_window():
    schedule = LevelSchedule("curriculum", ["L0", "L1", "L2"], seed=0, threshold=0.8, window=10)
    assert {schedule.sample() for _ in range(20)} == {"L0"}
    for _ in range(9):
        schedule.record("L0", True)
    assert schedule.unlocked == 1  # window not full yet
    schedule.record("L0", True)
    assert schedule.unlocked == 2
    for _ in range(10):
        schedule.record("L1", False)
    assert schedule.unlocked == 2
    assert schedule.summary() == {
        "curriculum/unlocked": 2.0,
        "curriculum/success_L0": 1.0,
        "curriculum/success_L1": 0.0,
    }


def test_reset_option_switches_level_and_time_limit():
    env = RocketLanderEnv(level="L0")
    short = env.max_steps
    env.reset(seed=0, options={"level": "L4"})
    assert env.level.name == "L4" and env.max_steps > short
    info = env.step(np.zeros(3, dtype=np.float32))[4]
    assert info["level"] == "L4"


def test_vec_env_records_finished_episodes_to_the_schedule():
    schedule = LevelSchedule("mix", ["L0", "L1"], seed=0)
    envs = VecEnv(lambda: RocketLanderEnv(level="L0"), n=2, seed=0, schedule=schedule)
    envs.reset()
    for _ in range(400):  # free fall ends every episode well within 400 steps
        envs.step(np.full((2, 3), -1.0, dtype=np.float32))  # engine off
    assert sum(len(r) for r in schedule.results.values()) > 0


@pytest.mark.parametrize("algo", ["grpo", "es"])
def test_train_cli_rejects_level_mixes_for_single_level_algorithms(algo, tmp_path):
    from rocketlander.cli import train

    with pytest.raises(SystemExit):
        train.main(["--algo", algo, "--level", "mix:L0,L1", "--run-dir", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_train_cli_runs_a_curriculum_and_evaluates_on_the_last_level(algo, tmp_path, capsys):
    from rocketlander.cli import train

    steps = "4096" if algo == "ppo" else "2000"
    train.main(
        [
            "--algo",
            algo,
            "--level",
            "curriculum:L0,L2",
            "--total-steps",
            steps,
            "--run-dir",
            str(tmp_path / "run"),
            "--quiet",
        ]
    )
    assert "Held-out success on L2" in capsys.readouterr().out
    assert (tmp_path / "run" / "model.pt").exists()
