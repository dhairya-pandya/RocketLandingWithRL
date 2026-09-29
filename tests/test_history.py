import numpy as np
import pytest

from rocketlander.agents.ppo import PPOConfig, train_ppo
from rocketlander.common.actor_critic import ActorCritic
from rocketlander.common.checkpoint import load_checkpoint, save_checkpoint
from rocketlander.common.history import ENTRY_SIZE, History, HistoryVecEnv, history_size
from rocketlander.common.logger import Logger
from rocketlander.common.vec_env import VecEnv
from rocketlander.envs.rocket_env import OBS_SIZE, RocketLanderEnv
from rocketlander.evaluation import EVAL_SEEDS, evaluate


def test_history_starts_filled_with_the_first_observation_and_rolls_forward():
    history = History(length=3, n=1)
    first, second = np.full(OBS_SIZE, 1.0), np.full(OBS_SIZE, 2.0)
    history.reset(0, first)
    assert np.all(history.entries[0, :, :OBS_SIZE] == 1.0)
    assert np.all(history.entries[0, :, OBS_SIZE:] == 0.0)
    history.push(second[None], np.array([[0.1, 0.2, 0.3]]))
    newest = history.entries[0, -1]
    assert np.all(newest[:OBS_SIZE] == 2.0) and np.allclose(newest[OBS_SIZE:], [0.1, 0.2, 0.3])
    assert history.stacked().shape == (1, 3 * ENTRY_SIZE) == (1, history_size(3))
    assert history_size(1) == OBS_SIZE


def test_history_vec_env_bootstraps_from_the_final_history_and_restarts_it():
    envs = HistoryVecEnv(VecEnv(lambda: RocketLanderEnv(level="L0"), n=2, seed=0), length=4)
    obs = envs.reset()
    assert obs.shape == (2, history_size(4))
    engine_off = np.full((2, 3), -1.0, dtype=np.float32)
    for _ in range(600):  # free fall ends every episode
        step = envs.step(engine_off)
        if step.terminated.any():
            break
    i = int(np.flatnonzero(step.terminated)[0])
    final = step.final_obs[i].reshape(4, ENTRY_SIZE)
    assert np.allclose(final[-1, OBS_SIZE:], -1.0)  # the last action is in the final history
    fresh = step.obs[i].reshape(4, ENTRY_SIZE)
    assert np.all(fresh[:, OBS_SIZE:] == 0.0)  # a new episode starts without actions
    assert np.all(fresh == fresh[0])


def test_actor_critic_remembers_within_an_episode_and_forgets_on_reset():
    agent = ActorCritic([16], -0.5, history=4)
    obs = np.random.default_rng(0).normal(size=(3, OBS_SIZE)).astype(np.float32)
    first = agent.act(obs[0])
    agent.act(obs[1])
    assert np.isfinite(agent.value(obs[1]))
    agent.reset()
    assert np.array_equal(agent.act(obs[0]), first)  # same start, same action


def test_history_is_saved_with_the_checkpoint(tmp_path):
    agent = ActorCritic([16], -0.5, history=4)
    save_checkpoint(tmp_path / "m.pt", agent, "ppo", PPOConfig(history=4), level="L0", seed=0)
    loaded, _ = load_checkpoint(tmp_path / "m.pt")
    assert loaded.history == 4
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    assert np.array_equal(agent.act(obs), loaded.act(obs))


def test_ppo_trains_with_a_history_and_evaluates_through_act(tmp_path):
    config = PPOConfig(
        total_steps=2_048,
        num_envs=2,
        num_steps=64,
        num_minibatches=4,
        hidden=[16, 16],
        eval_interval=10**9,
        eval_episodes=1,
        history=4,
    )
    agent = train_ppo(config, "L0", seed=0, logger=Logger(tmp_path, verbose=False))
    assert agent.history == 4
    assert 0.0 <= evaluate(agent, "L0", EVAL_SEEDS[:2]).success_rate <= 1.0


def test_train_cli_can_switch_the_history_on(tmp_path):
    from rocketlander.cli import train

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
            "history=4",
            "--set",
            "num_envs=2",
            "--set",
            "num_steps=64",
            "--set",
            "num_minibatches=4",
            "--quiet",
        ]
    )
    assert load_checkpoint(run / "model.pt")[0].history == 4


def test_train_cli_refuses_to_fine_tune_with_a_different_history(tmp_path, capsys):
    from rocketlander.cli import train

    save_checkpoint(
        tmp_path / "h1.pt", ActorCritic([16], -0.5), "ppo", PPOConfig(), level="L0", seed=0
    )
    with pytest.raises(SystemExit):
        train.main(
            [
                "--init",
                str(tmp_path / "h1.pt"),
                "--set",
                "history=4",
                "--run-dir",
                str(tmp_path / "run"),
            ]
        )
    assert "history" in capsys.readouterr().err and not (tmp_path / "run").exists()
