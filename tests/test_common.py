import numpy as np
import pytest
import torch

from rocketlander.common.networks import GaussianPolicy, mlp
from rocketlander.common.normalization import RewardScaler, RunningMeanStd
from rocketlander.common.vec_env import VecEnv
from rocketlander.envs.rocket_env import OBS_SIZE, RocketLanderEnv


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, size=(1_000, 4))
    rms = RunningMeanStd((4,))
    for batch in np.split(data, 10):
        rms.update(batch)
    assert rms.mean == pytest.approx(data.mean(axis=0), abs=1e-3)
    assert rms.var == pytest.approx(data.var(axis=0), rel=1e-3)
    assert np.all(np.abs(rms.normalize(data * 100)) <= 10.0)


def test_reward_scaler_restarts_returns_at_episode_end():
    scaler = RewardScaler(num_envs=2, gamma=0.9)
    scaler.scale(np.array([1.0, 1.0]), np.array([True, False]))
    assert scaler.returns.tolist() == [0.0, 1.0]


def test_vec_env_resets_finished_episodes_and_keeps_the_final_observation():
    envs = VecEnv(lambda: RocketLanderEnv(level="L0"), n=2, seed=0)
    obs = envs.reset()
    assert obs.shape == (2, OBS_SIZE)
    falling = np.tile(np.array([-1.0, 0.0, 0.0], dtype=np.float32), (2, 1))
    for _ in range(2_000):
        step = envs.step(falling)
        if step.terminated.any():
            break
    i = int(np.flatnonzero(step.terminated)[0])
    assert step.infos[i]["outcome"] == "crashed"
    assert not np.array_equal(step.obs[i], step.final_obs[i])  # obs is already a new episode
    assert step.obs[i][1] > step.final_obs[i][1]  # the new start is high above the deck


def test_gaussian_policy_log_prob_matches_the_formula():
    policy = GaussianPolicy(obs_dim=2, act_dim=1, hidden=[8], init_log_std=np.log(0.5))
    obs, action = torch.zeros(1, 2), torch.tensor([[0.3]])
    mean = policy.mean(obs)
    expected = -0.5 * ((0.3 - mean) / 0.5) ** 2 - np.log(0.5) - 0.5 * np.log(2 * np.pi)
    assert policy.dist(obs).log_prob(action).item() == pytest.approx(expected.item(), abs=1e-6)


def test_mlp_output_gain_keeps_initial_actions_small():
    net = mlp([11, 64, 64, 3], output_gain=0.01)
    assert net(torch.randn(100, 11)).abs().max().item() < 0.1


def test_vec_env_refuses_seeds_that_reach_the_eval_seeds():
    with pytest.raises(ValueError, match="evaluation seeds"):
        VecEnv(lambda: RocketLanderEnv(level="L0"), n=32, seed=9_980)


def test_fine_tuning_lets_the_normalizer_adapt_to_new_features():
    rms = RunningMeanStd((2,))
    rms.update(np.column_stack([np.random.default_rng(0).normal(size=10_000), np.zeros(10_000)]))
    rms.count = 5e6  # as after a long run on a level where feature 1 never varied
    rms.soften(max_count=1e4)
    new_level = np.random.default_rng(1).normal(0.0, 0.7, size=(20_000, 2))
    for batch in np.split(new_level, 20):
        rms.update(batch)
    assert rms.var[1] > 0.1
