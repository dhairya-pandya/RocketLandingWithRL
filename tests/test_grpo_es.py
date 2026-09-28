import numpy as np
import pytest
import torch

from rocketlander.agents.es import ESAgent, ESConfig, centered_ranks, train_es
from rocketlander.agents.grpo import (
    GRPOConfig,
    group_advantages,
    process_advantages,
    rollout_groups,
    train_grpo,
)
from rocketlander.common.behavior_cloning import clone_pid
from rocketlander.common.checkpoint import load_checkpoint, save_checkpoint
from rocketlander.common.logger import Logger
from rocketlander.common.policy_agent import PolicyAgent
from rocketlander.evaluation import EVAL_SEEDS, evaluate


def test_group_advantages_are_zero_mean_unit_std():
    advantages = group_advantages(np.array([1.0, 2.0, 3.0, 6.0]))
    assert advantages.mean() == pytest.approx(0.0) and advantages.std() == pytest.approx(1.0)
    assert np.all(group_advantages(np.full(4, 5.0)) == 0.0)  # identical rollouts teach nothing


def test_process_advantages_sum_normalized_future_rewards():
    rewards = [np.array([1.0, 0.0]), np.array([0.0, -1.0])]
    advantages = process_advantages(rewards, gamma=0.5)
    flat = np.concatenate(rewards)
    z = (flat - flat.mean()) / flat.std()
    assert advantages[0] == pytest.approx([z[0] + 0.5 * z[1], z[1]])
    assert advantages[1] == pytest.approx([z[2] + 0.5 * z[3], z[3]])


def test_group_members_start_from_the_same_state():
    torch.manual_seed(0)
    groups = rollout_groups(PolicyAgent([16], -0.5), "L3", start_seeds=[5, 6], group_size=3)
    for group in groups:
        firsts = np.array([o[0] for o in group.obs])
        assert np.allclose(firsts, firsts[0])
    assert not np.allclose(groups[0].obs[0][0], groups[1].obs[0][0])


def test_stored_log_probs_match_the_policy_before_any_update():
    """Regression: observations were once stored as views that the next step overwrote."""
    torch.manual_seed(0)
    agent = PolicyAgent([16], -0.5)
    groups = rollout_groups(agent, "L0", start_seeds=[1], group_size=4)
    obs = np.concatenate(groups[0].obs)
    actions = torch.as_tensor(np.concatenate(groups[0].actions))
    with torch.no_grad():
        fresh = agent.policy.dist(agent.normalized(obs)).log_prob(actions).sum(-1).numpy()
    assert np.allclose(fresh, np.concatenate(groups[0].log_probs), atol=1e-5)


def test_centered_ranks_ignore_the_scale_of_fitness():
    ranks = centered_ranks(np.array([10.0, -1000.0, 3.0, 5.0]))
    assert ranks == pytest.approx([0.5, -0.5, -1 / 6, 1 / 6])


def test_es_batched_policy_matches_each_member_acting_alone():
    rng = np.random.default_rng(0)
    agent = ESAgent([8, 8])
    flats = rng.normal(0.0, 0.3, size=(3, len(agent.get_flat())))
    obs = rng.normal(size=(3, 11)).astype(np.float32)
    batched = agent.batched_policy(flats, obs)
    for i in range(3):
        agent.set_flat(flats[i])
        assert np.allclose(batched[i], agent.act(obs[i]), atol=1e-5)


def test_behavior_cloning_learns_to_land_like_the_pid():
    torch.manual_seed(0)
    agent = PolicyAgent([64, 64], -0.5)
    clone_pid(agent, "L0", seed=0, steps=200_000, noise=0.3, epochs=10)
    assert evaluate(agent, "L0", EVAL_SEEDS[:50]).success_rate >= 0.7  # 80-100% across seeds


TINY_GRPO = GRPOConfig(
    total_steps=3_000,
    starts_per_update=2,
    group_size=2,
    hidden=[16],
    eval_interval=10**9,
    eval_episodes=2,
)
TINY_ES = ESConfig(
    total_steps=3_000,
    population=4,
    episodes_per_member=1,
    hidden=[8],
    eval_interval=10**9,
    eval_episodes=2,
)


@pytest.mark.parametrize(
    ("algo", "train", "config"), [("grpo", train_grpo, TINY_GRPO), ("es", train_es, TINY_ES)]
)
def test_training_runs_and_round_trips_a_checkpoint(tmp_path, algo, train, config):
    logger = Logger(tmp_path / "run", verbose=False)
    agent = train(config, "L0", seed=0, logger=logger)
    logger.close()
    obs = np.random.default_rng(0).normal(size=(5, 11)).astype(np.float32)
    assert np.all(np.abs(agent.act(obs)) <= 1.0)
    save_checkpoint(tmp_path / "model.pt", agent, algo, config, level="L0", seed=0)
    loaded, meta = load_checkpoint(tmp_path / "model.pt")
    assert np.array_equal(agent.act(obs), loaded.act(obs)) and meta["algo"] == algo
    torch.load(tmp_path / "model.pt", weights_only=True)


@pytest.mark.slow
def test_grpo_learns_to_land_on_l0(tmp_path):
    torch.set_num_threads(1)
    agent = train_grpo(GRPOConfig(total_steps=3_000_000), "L0", 1, Logger(tmp_path, verbose=False))
    assert evaluate(agent, "L0", EVAL_SEEDS[:50]).success_rate >= 0.5


@pytest.mark.slow
def test_es_learns_to_land_on_l0(tmp_path):
    torch.set_num_threads(1)
    agent = train_es(ESConfig(total_steps=5_000_000), "L0", 1, Logger(tmp_path, verbose=False))
    assert evaluate(agent, "L0", EVAL_SEEDS[:50]).success_rate >= 0.5
