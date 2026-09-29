import numpy as np
import pytest
import yaml

from rocketlander.cli import evaluate, record, train, watch
from rocketlander.cli.common import make_agent
from rocketlander.common.actor_critic import ActorCritic


def test_train_then_evaluate_watch_and_record_a_checkpoint(tmp_path, capsys):
    run_dir = tmp_path / "run"
    train.main(
        [
            "--algo",
            "ppo",
            "--level",
            "L0",
            "--total-steps",
            "4096",
            "--run-dir",
            str(run_dir),
            "--quiet",
        ]
    )
    model = run_dir / "model.pt"
    assert model.exists() and (run_dir / "config.yaml").exists()
    assert "Held-out success" in capsys.readouterr().out
    assert isinstance(make_agent(str(model)), ActorCritic)

    evaluate.main(["--agent", str(model), "--level", "L0", "--episodes", "3"])
    assert "over 3 episodes" in capsys.readouterr().out
    watch.main(["--agent", str(model), "--headless", "--max-frames", "10"])
    frames = record.record(
        str(model),
        "L0",
        0,
        tmp_path / "clip.gif",
        every=10,
        scale=0.2,
        hold_seconds=0.0,
        max_steps=20,
    )
    assert frames == 2


def test_train_an_off_policy_agent_from_the_command_line(tmp_path):
    run_dir = tmp_path / "sac"
    train.main(
        [
            "--algo",
            "sac",
            "--level",
            "L0",
            "--total-steps",
            "1000",
            "--run-dir",
            str(run_dir),
            "--quiet",
        ]
    )
    assert (run_dir / "model.pt").exists()
    assert make_agent(str(run_dir / "model.pt")).act(np.zeros(11, dtype=np.float32)).shape == (3,)


def test_train_cli_overrides_config_values(tmp_path):
    run = tmp_path / "run"
    train.main(
        [
            "--algo",
            "ppo",
            "--level",
            "L0",
            "--total-steps",
            "2048",
            "--run-dir",
            str(run),
            "--set",
            "ent_coef=0.003",
            "--set",
            "num_envs=2",
            "--set",
            "num_steps=64",
            "--set",
            "num_minibatches=4",
            "--set",
            "hidden=[16, 16]",
            "--quiet",
        ]
    )
    written = yaml.safe_load((run / "config.yaml").read_text())
    assert written["ent_coef"] == 0.003 and written["hidden"] == [16, 16]


@pytest.mark.parametrize("bad", ["entropy=0.1", "ent_coef"])
def test_train_cli_rejects_unknown_or_malformed_overrides(bad, tmp_path, capsys):
    with pytest.raises(SystemExit):
        train.main(["--set", bad, "--run-dir", str(tmp_path / "run")])
    err = capsys.readouterr().err
    assert "--set" in err and "KEY=VALUE" in err and not (tmp_path / "run").exists()
