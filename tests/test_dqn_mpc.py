import numpy as np
import pytest
import torch

from rocketlander.agents.dqn import ACTIONS, BranchingQ, DQNConfig, nearest_action, train_dqn
from rocketlander.agents.mpc import MPCAgent, MPCConfig, nominal_model
from rocketlander.agents.pid import PIDAgent
from rocketlander.common.checkpoint import load_checkpoint, save_checkpoint
from rocketlander.common.logger import Logger
from rocketlander.envs.rocket_env import RocketLanderEnv
from rocketlander.evaluation import EVAL_SEEDS, evaluate


def test_action_grid_is_a_product_of_levels_inside_the_action_box():
    assert ACTIONS.shape == (75, 3) and np.abs(ACTIONS).max() <= 1.0
    assert np.array_equal(nearest_action(ACTIONS), np.arange(75))
    assert len({tuple(a) for a in ACTIONS}) == 75


def test_branching_q_adds_one_advantage_per_control_in_grid_order():
    torch.manual_seed(0)
    q = BranchingQ([16])
    obs = torch.randn(4, 11)
    value, throttle, gimbal, rcs = q.net(obs).split([1, 5, 5, 3], dim=-1)
    throttle, gimbal, rcs = (a - a.mean(-1, keepdim=True) for a in (throttle, gimbal, rcs))
    expected = torch.stack(
        [
            value[:, 0] + throttle[:, i] + gimbal[:, j] + rcs[:, k]
            for i in range(5)
            for j in range(5)
            for k in range(3)
        ],
        dim=1,
    )
    assert torch.allclose(q(obs), expected, atol=1e-6)
    assert q(obs[0]).shape == (75,)


class SnappedPID:
    """The PID restricted to the DQN's grid: shows the grid can express a landing."""

    def __init__(self) -> None:
        self.pid = PIDAgent()

    def act(self, obs, deterministic=True):
        return ACTIONS[nearest_action(self.pid.act(obs)[None])[0]]


def test_the_pid_snapped_to_the_grid_still_lands():
    assert evaluate(SnappedPID(), "L0", EVAL_SEEDS[:5]).success_rate == 1.0


def test_dqn_trains_and_round_trips_a_checkpoint(tmp_path):
    config = DQNConfig(
        total_steps=2_000,
        learning_starts=1_000,
        buffer_size=5_000,
        batch_size=32,
        num_envs=4,
        eval_interval=10**9,
        log_interval=10**9,
        eval_episodes=1,
        hidden=[32, 32],
    )
    logger = Logger(tmp_path / "run", verbose=False)
    agent = train_dqn(config, "L0", seed=0, logger=logger)
    logger.close()
    obs = np.random.default_rng(0).normal(size=(5, 11)).astype(np.float32)
    actions = agent.act(obs)
    assert actions.shape == (5, 3) and np.array_equal(ACTIONS[nearest_action(actions)], actions)
    assert np.isfinite(agent.value(obs[0]))
    save_checkpoint(tmp_path / "model.pt", agent, "dqn", config, level="L0", seed=0)
    loaded, meta = load_checkpoint(tmp_path / "model.pt")
    assert np.array_equal(agent.act(obs), loaded.act(obs)) and meta["algo"] == "dqn"


def test_nominal_model_recovers_the_rocket_relative_to_the_deck():
    env = RocketLanderEnv(level="L2")
    obs, _ = env.reset(seed=3)
    rocket, deck = nominal_model(obs)
    frame = env.frame()
    assert rocket.x == pytest.approx(frame.rocket.x - frame.deck.x, abs=1e-3)
    assert rocket.y == pytest.approx(frame.rocket.y - frame.deck.y, abs=1e-3)
    assert rocket.theta == pytest.approx(frame.rocket.theta, abs=1e-5)
    assert deck.angle == pytest.approx(frame.deck.angle, abs=1e-6)


def test_mpc_replans_every_hold_steps_and_acts_in_the_box():
    agent = MPCAgent(MPCConfig(candidates=8, elites=2, iterations=1))
    env = RocketLanderEnv(level="L0")
    obs, _ = env.reset(seed=0)
    actions = [agent.act(obs) for _ in range(agent.config.hold)]
    assert all(np.array_equal(a, actions[0]) for a in actions)  # one plan knot is held
    assert np.abs(actions[0]).max() <= 1.0


@pytest.mark.slow
def test_dqn_learns_to_land_on_l0(tmp_path):
    torch.set_num_threads(1)
    agent = train_dqn(DQNConfig(), "L0", seed=1, logger=Logger(tmp_path, verbose=False))
    assert evaluate(agent, "L0", EVAL_SEEDS[:50]).success_rate >= 0.3


@pytest.mark.slow
def test_mpc_lands_on_l0():
    assert evaluate(MPCAgent(), "L0", EVAL_SEEDS[:5]).success_rate >= 0.8
