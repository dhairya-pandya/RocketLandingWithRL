import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rocketlander.envs.physics import decode_action
from rocketlander.envs.rocket_env import (
    FUEL_COST,
    OBS_SIZE,
    SHAPING_GAMMA,
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
    assert rewards[-1] in (-100.0,) or rewards[-1] >= 100.0


def test_shaping_telescopes_to_minus_initial_potential():
    """Shaping with phi(terminal) = 0 adds exactly -phi(s0) to the discounted return."""
    env = RocketLanderEnv(level="L0")
    shaped = RocketLanderEnv(level="L0")
    sparse = RocketLanderEnv(level="L0", reward_mode="sparse")
    env.reset(seed=5)
    initial_potential = env.get_state().potential
    shaped.reset(seed=5)
    sparse.reset(seed=5)

    actions = fixed_actions(2_000)
    _, shaped_rewards = rollout(shaped, actions)
    _, sparse_rewards = rollout(sparse, actions)
    fuel = np.array(
        [
            FUEL_COST * decode_action(a, env.get_state().setup.params).throttle
            for a in actions[: len(shaped_rewards)]
        ]
    )
    shaping_only = shaped_rewards + fuel - sparse_rewards
    discounts = SHAPING_GAMMA ** np.arange(len(shaping_only))
    assert np.sum(discounts * shaping_only) == pytest.approx(-initial_potential)


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
