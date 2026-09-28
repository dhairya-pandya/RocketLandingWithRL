import pytest
from PIL import Image

from rocketlander.cli import play, record, watch
from rocketlander.cli.common import make_agent
from rocketlander.cli.play import KeyboardPilot
from rocketlander.envs.physics import RocketParams, RocketState, compute_forces, decode_action


def test_keyboard_throttle_is_sticky_clipped_and_gimbal_springs_back():
    pilot = KeyboardPilot()
    for _ in range(30):
        action = pilot.action({"w": True, "d": True}, dt=1 / 30)
    assert action[0] == pytest.approx(2 * 0.8 - 1) and action[1] == 1.0
    action = pilot.action({}, dt=1 / 30)
    assert action[0] == pytest.approx(2 * 0.8 - 1) and action[1] == 0.0
    for _ in range(100):
        action = pilot.action({"w": True}, dt=1 / 30)
    assert action[0] == 1.0
    for _ in range(100):
        action = pilot.action({"s": True, "e": True}, dt=1 / 30)
    assert action[0] == -1.0 and action[2] == -1.0


@pytest.mark.parametrize("key", ["a", "q"])
def test_left_hand_keys_both_lean_the_rocket_left(key):
    params = RocketParams()
    state = RocketState(
        x=0.0, y=100.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0, fuel=5_000.0, throttle=0.5
    )
    controls = decode_action(KeyboardPilot().action({key: True}, dt=1 / 30), params)
    assert compute_forces(state, controls, params, wind_speed=0.0).torque > 0  # CCW = lean left


def test_unknown_agent_raises():
    with pytest.raises(ValueError, match="Unknown agent"):
        make_agent("ppo-from-the-future")


def test_record_writes_an_animated_gif(tmp_path):
    out = tmp_path / "clip.gif"
    n = record.record(
        "pid", "L0", seed=0, out=out, every=4, scale=0.25, hold_seconds=0.5, max_steps=40
    )
    assert n == 10 + 3
    with Image.open(out) as gif:
        assert gif.n_frames == n and gif.size == (275, 180)


def test_watch_and_play_run_headless():
    """Smoke test: the interactive loops run end to end without a window."""
    watch.main(["--headless", "--max-frames", "40", "--level", "L1", "--agent", "random"])
    play.main(["--headless", "--max-frames", "20", "--level", "L0"])
