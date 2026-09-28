import numpy as np
import pytest
import torch

from rocketlander.agents.ppo import PPOConfig, train_ppo
from rocketlander.agents.reinforce import ReinforceConfig, discounted_returns, train_reinforce
from rocketlander.common.checkpoint import load_checkpoint, save_checkpoint
from rocketlander.common.config import config_from_dict, load_config
from rocketlander.common.logger import Logger
from rocketlander.evaluation import EVAL_SEEDS, evaluate

TINY = dict(total_steps=2_048, num_envs=2, hidden=[16, 16], eval_interval=10**9, eval_episodes=2)


@pytest.mark.parametrize(
    ("train", "config"),
    [
        (train_ppo, PPOConfig(num_steps=64, num_minibatches=4, **TINY)),
        (train_reinforce, ReinforceConfig(episodes_per_update=2, **TINY)),
    ],
)
def test_training_runs_and_returns_a_working_agent(tmp_path, train, config):
    logger = Logger(tmp_path, verbose=False)
    agent = train(config, "L0", seed=0, logger=logger)
    logger.close()
    obs = np.zeros(11, dtype=np.float32)
    action = agent.act(obs)
    assert action.shape == (3,) and np.all(np.abs(action) <= 1.0)
    assert np.isfinite(agent.value(obs))
    assert "eval/success_rate" in (tmp_path / "metrics.csv").read_text()


def test_discounted_returns_bootstrap_on_truncation():
    returns = discounted_returns(np.array([1.0, 1.0]), gamma=0.5, bootstrap=4.0)
    assert returns.tolist() == [1.0 + 0.5 * (1.0 + 0.5 * 4.0), 1.0 + 0.5 * 4.0]


def test_checkpoint_round_trip_gives_identical_actions(tmp_path):
    torch.manual_seed(0)
    logger = Logger(tmp_path / "run", verbose=False)
    config = PPOConfig(num_steps=64, num_minibatches=4, **TINY)
    agent = train_ppo(config, "L0", seed=0, logger=logger)
    save_checkpoint(tmp_path / "model.pt", agent, "ppo", config, level="L0", seed=0)
    loaded, meta = load_checkpoint(tmp_path / "model.pt")
    obs = np.random.default_rng(0).normal(size=(5, 11)).astype(np.float32)
    assert np.array_equal(agent.act(obs), loaded.act(obs))
    assert meta["algo"] == "ppo" and meta["level"] == "L0" and meta["config"]["gamma"] == 0.999


def test_eval_seeds_are_held_out_from_training_seeds():
    assert min(EVAL_SEEDS) >= 10_000 and len(EVAL_SEEDS) == 100


def test_configs_load_from_yaml_and_reject_unknown_keys():
    config = load_config(PPOConfig, "ppo", {"total_steps": 1_000})
    assert config.total_steps == 1_000 and config.gamma == 0.999 and config.hidden == [64, 64]
    with pytest.raises(ValueError, match="Unknown PPOConfig fields"):
        config_from_dict(PPOConfig, {"learning_rat": 1e-3})


@pytest.mark.slow
def test_ppo_learns_to_land_on_l0(tmp_path):
    torch.set_num_threads(1)
    logger = Logger(tmp_path, verbose=False)
    agent = train_ppo(PPOConfig(total_steps=3_000_000), "L0", seed=1, logger=logger)
    assert evaluate(agent, "L0", EVAL_SEEDS[:50]).success_rate >= 0.5
