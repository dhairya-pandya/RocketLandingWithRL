import numpy as np
import pytest

from rocketlander.common.gae import compute_gae


def column(values):
    return np.array(values, dtype=float)[:, None]  # (T, 1): one environment


def test_matches_a_hand_computed_example():
    # gamma 0.9, lambda 0.8; episode terminates after the third step.
    rewards, values = column([1, 1, 1]), column([0.5, 0.5, 0.5])
    next_values, dones = column([0.5, 0.5, 0.0]), column([0, 0, 1])
    advantages, returns = compute_gae(rewards, values, next_values, dones, 0.9, 0.8)
    d2 = 1 + 0 - 0.5  # deltas: r + gamma * V(next) - V
    d1 = 1 + 0.9 * 0.5 - 0.5
    d0 = d1
    a2 = d2
    a1 = d1 + 0.9 * 0.8 * a2
    a0 = d0 + 0.9 * 0.8 * a1
    assert advantages[:, 0] == pytest.approx([a0, a1, a2])
    assert returns[:, 0] == pytest.approx([a0 + 0.5, a1 + 0.5, a2 + 0.5])


def test_episode_boundary_cuts_the_trace_but_truncation_still_bootstraps():
    # Step 0 ends an episode by truncation: its target bootstraps from V(final obs) = 2,
    # and step 1 (a new episode) must not leak into step 0's advantage.
    rewards, values = column([0, 10]), column([1, 1])
    next_values, dones = column([2, 0]), column([1, 1])
    advantages, _ = compute_gae(rewards, values, next_values, dones, 0.99, 0.95)
    assert advantages[0, 0] == pytest.approx(0 + 0.99 * 2 - 1)


def test_lambda_zero_is_one_step_td_and_one_is_monte_carlo():
    rng = np.random.default_rng(0)
    rewards, values = rng.normal(size=(5, 1)), rng.normal(size=(5, 1))
    next_values, dones = np.vstack([values[1:], [[0.0]]]), column([0, 0, 0, 0, 1])
    td, _ = compute_gae(rewards, values, next_values, dones, 0.9, 0.0)
    assert td == pytest.approx(rewards + 0.9 * next_values - values)
    mc, returns = compute_gae(rewards, values, next_values, dones, 0.9, 1.0)
    discounted = [sum(0.9**k * rewards[t + k, 0] for k in range(5 - t)) for t in range(5)]
    assert returns[:, 0] == pytest.approx(discounted)
