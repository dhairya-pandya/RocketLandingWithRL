from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rocketlander.envs.physics import decode_action
from rocketlander.envs.rocket_env import (
    DECK_CRASH_BASE,
    DECK_CRASH_PER_MS,
    FUEL_COST,
    FUEL_SCALE,
    OBS_SIZE,
    SHAPING_GAMMA,
    TIME_COST,
    RocketLanderEnv,
)


def rollout(env, actions):
    """Step through a fixed action sequence; return observations and rewards until done."""
    observations, rewards = [], []
    for action in actions:
        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs)
        rewards.append(reward)
        if terminated or truncated:
            break
    return np.array(observations), np.array(rewards)


def fixed_actions(n=400, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, size=(n, 3)).astype(np.float32)


@pytest.mark.parametrize("level", ["L0", "L2", "L4"])
def test_passes_gymnasium_env_checker(level):
    check_env(RocketLanderEnv(level=level), skip_render_check=True)


def test_observation_shape_and_dtype():
    obs, _ = RocketLanderEnv().reset(seed=0)
    assert obs.shape == (OBS_SIZE,) and obs.dtype == np.float32


def test_same_seed_gives_identical_episodes():
    runs = []
    for _ in range(2):
        env = RocketLanderEnv(level="L3")
        env.reset(seed=123)
        runs.append(rollout(env, fixed_actions()))
    assert np.array_equal(runs[0][0], runs[1][0])
    assert np.array_equal(runs[0][1], runs[1][1])


def test_set_state_reproduces_the_future_exactly():
    env = RocketLanderEnv(level="L3")  # L3 has random wind, so the RNG must be restored too
    env.reset(seed=7)
    rollout(env, fixed_actions(20, seed=1))
    snapshot = env.get_state()
    first = rollout(env, fixed_actions(seed=2))

    env.set_state(snapshot)
    second = rollout(env, fixed_actions(seed=2))
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])


def test_reset_with_state_option_restores_snapshot():
    env = RocketLanderEnv(level="L2")
    start_obs, _ = env.reset(seed=3)
    snapshot = env.get_state()
    rollout(env, fixed_actions(50))
    restored_obs, _ = env.reset(options={"state": snapshot})
    assert np.array_equal(start_obs, restored_obs)


def test_sparse_reward_is_zero_until_the_episode_ends():
    env = RocketLanderEnv(level="L0", reward_mode="sparse")
    env.reset(seed=0)
    _, rewards = rollout(env, fixed_actions(2_000))
    assert np.all(rewards[:-1] == 0.0)
    assert rewards[-1] <= -DECK_CRASH_BASE or rewards[-1] >= 100.0


@pytest.mark.parametrize("gamma", [SHAPING_GAMMA, 0.99])
def test_shaping_telescopes_to_final_minus_initial_potential(gamma):
    """The shaping terms add up to gamma^T phi(s_T) - phi(s_0): progress toward a good touchdown."""
    env = RocketLanderEnv(level="L0", shaping_gamma=gamma)
    env.reset(seed=5)
    initial_potential = env.get_state().potential

    shaping_only = []
    for action in fixed_actions(2_000):
        _, reward, terminated, truncated, info = env.step(action)
        shaping_only.append(reward - info["task_reward"])
        if terminated or truncated:
            break
    steps = len(shaping_only)
    discounted = np.sum(gamma ** np.arange(steps) * np.array(shaping_only))
    final_potential = env.get_state().potential
    assert discounted == pytest.approx(gamma**steps * final_potential - initial_potential)


def test_task_reward_is_terminal_reward_minus_fuel_and_time_costs():
    shaped = RocketLanderEnv(level="L0")
    sparse = RocketLanderEnv(level="L0", reward_mode="sparse")
    shaped.reset(seed=5)
    sparse.reset(seed=5)
    params = shaped.get_state().setup.params
    for action in fixed_actions(2_000):
        _, terminal, terminated, truncated, _ = sparse.step(action)
        _, _, _, _, info = shaped.step(action)
        fuel = FUEL_COST * decode_action(action, params).throttle
        assert info["task_reward"] == pytest.approx(terminal - fuel - TIME_COST)
        if terminated or truncated:
            break


def test_deck_crash_penalty_grows_with_impact_speed():
    penalties, impacts = [], []
    for speed in (3.0, 8.0, 30.0):
        env = RocketLanderEnv(level="L0", reward_mode="sparse")
        env.reset(seed=0)
        state = env.get_state()
        deck_y = 0.0
        state.rocket = replace(state.rocket, x=0.0, y=deck_y + 11.1, vx=0.0, vy=-speed, theta=0.0)
        env.set_state(state)
        _, terminal, terminated, _, info = env.step(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
        assert terminated and info["outcome"] == "crashed"
        penalties.append(terminal)
        impacts.append(info["touchdown_speed"])
    assert penalties[0] > penalties[1] > penalties[2] == -100.0
    assert penalties[0] == pytest.approx(-(DECK_CRASH_BASE + DECK_CRASH_PER_MS * impacts[0]))


def test_fuel_is_observed_in_tonnes():
    env = RocketLanderEnv(level="L0")
    obs, _ = env.reset(seed=0)
    assert obs[9] == pytest.approx(env.get_state().rocket.fuel / FUEL_SCALE)
    assert 1.0 <= obs[9] <= 1.2  # L0 tanks hold 1000-1200 kg


def test_time_limit_truncates():
    env = RocketLanderEnv(level="L4")
    env.max_steps = 5
    env.reset(seed=0)
    hover = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(4):
        *_, terminated, truncated, _ = env.step(hover)
        assert not (terminated or truncated)
    *_, terminated, truncated, info = env.step(hover)
    assert truncated and not terminated and info["reason"] == "time limit"


def test_invalid_usage_raises():
    with pytest.raises(ValueError, match="reward_mode"):
        RocketLanderEnv(reward_mode="dense")
    with pytest.raises(RuntimeError, match="reset"):
        RocketLanderEnv().step(np.zeros(3, dtype=np.float32))


def test_frame_exposes_render_data():
    env = RocketLanderEnv(level="L1")
    env.reset(seed=0)
    env.step(np.zeros(3, dtype=np.float32))
    frame = env.frame()
    assert frame.level == "L1" and frame.forces is not None and frame.time > 0.0


def test_snapshot_is_independent_of_later_steps():
    env = RocketLanderEnv(level="L1")
    env.reset(seed=0)
    snapshot = env.get_state()
    height_before = snapshot.rocket.y
    rollout(env, fixed_actions(30))
    assert snapshot.rocket.y == height_before and snapshot.steps == 0


def test_env_can_be_reused_after_an_episode_ends():
    env = RocketLanderEnv(level="L0")
    env.reset(seed=0)
    rollout(env, fixed_actions(2_000))
    assert env.frame().judgement.done
    env.reset(seed=1)
    _, _, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float32))
    assert not (terminated or truncated) and not env.frame().judgement.done


def test_restoring_one_snapshot_twice_gives_identical_rollouts():
    env = RocketLanderEnv(level="L3")
    env.reset(seed=11)
    snapshot = env.get_state()
    runs = []
    for _ in range(2):
        env.set_state(snapshot)
        runs.append(rollout(env, fixed_actions(seed=3)))
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])
