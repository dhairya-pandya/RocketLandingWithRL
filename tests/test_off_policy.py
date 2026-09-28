import numpy as np
import pytest
import torch

from rocketlander.agents.sac import SACConfig, train_sac
from rocketlander.agents.td3 import TD3Config, train_td3
from rocketlander.common.checkpoint import load_checkpoint, save_checkpoint
from rocketlander.common.logger import Logger
from rocketlander.common.networks import QNetwork, TanhGaussianActor
from rocketlander.common.off_policy import soft_update
from rocketlander.common.replay_buffer import ReplayBuffer
from rocketlander.evaluation import EVAL_SEEDS, evaluate

TINY = dict(
    total_steps=1_000,
    num_envs=2,
    buffer_size=5_000,
    batch_size=32,
    learning_starts=400,
    updates_per_step=1,
    hidden=[16, 16],
    log_interval=500,
    eval_interval=10**9,
    eval_episodes=2,
)


def test_replay_buffer_overwrites_the_oldest_transitions():
    buffer = ReplayBuffer(capacity=3, obs_dim=2, act_dim=1, seed=0)
    for i in range(5):
        buffer.add(
            np.full((1, 2), i), np.zeros((1, 1)), np.array([i]), np.zeros((1, 2)), np.array([False])
        )
    assert len(buffer) == 3
    assert sorted(buffer.rewards.tolist()) == [2.0, 3.0, 4.0]
    batch = buffer.sample(8)
    assert batch["obs"].shape == (8, 2) and batch["actions"].shape == (8, 1)
    assert set(batch["rewards"].tolist()) <= {2.0, 3.0, 4.0}


def test_soft_update_moves_the_target_by_tau():
    source, target = QNetwork(2, 1, [4]), QNetwork(2, 1, [4])
    before = [p.clone() for p in target.parameters()]
    soft_update(target, source, tau=0.1)
    for b, t, s in zip(before, target.parameters(), source.parameters(), strict=True):
        assert torch.allclose(t, 0.9 * b + 0.1 * s)


def test_tanh_actor_log_prob_matches_torch_transformed_distribution():
    torch.manual_seed(0)
    actor = TanhGaussianActor(obs_dim=4, act_dim=3, hidden=[16])
    obs = torch.randn(64, 4)
    torch.manual_seed(1)
    action, log_prob = actor(obs)
    mean, log_std = actor.net(obs).chunk(2, dim=-1)
    squashed = torch.distributions.TransformedDistribution(
        torch.distributions.Normal(mean, log_std.clamp(-5, 2).exp()),
        torch.distributions.transforms.TanhTransform(),
    )
    expected = squashed.log_prob(action.clamp(-0.999999, 0.999999)).sum(-1)
    assert torch.allclose(log_prob, expected, atol=1e-2)
    assert action.abs().max() <= 1.0


@pytest.mark.parametrize(
    ("algo", "train", "config"),
    [("sac", train_sac, SACConfig(**TINY)), ("td3", train_td3, TD3Config(**TINY))],
)
def test_off_policy_training_runs_and_round_trips_a_checkpoint(tmp_path, algo, train, config):
    logger = Logger(tmp_path / "run", verbose=False)
    agent = train(config, "L0", seed=0, logger=logger)
    logger.close()
    obs = np.random.default_rng(0).normal(size=(5, 11)).astype(np.float32)
    assert np.all(np.abs(agent.act(obs)) <= 1.0) and np.isfinite(agent.value(obs[0]))
    save_checkpoint(tmp_path / "model.pt", agent, algo, config, level="L0", seed=0)
    loaded, meta = load_checkpoint(tmp_path / "model.pt")
    assert np.array_equal(agent.act(obs), loaded.act(obs)) and meta["algo"] == algo
    torch.load(tmp_path / "model.pt", weights_only=True)


@pytest.mark.slow
def test_sac_learns_to_land_on_l0(tmp_path):
    torch.set_num_threads(1)
    logger = Logger(tmp_path, verbose=False)
    agent = train_sac(SACConfig(total_steps=400_000), "L0", seed=1, logger=logger)
    assert evaluate(agent, "L0", EVAL_SEEDS[:50]).success_rate >= 0.5
