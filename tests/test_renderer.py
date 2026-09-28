import numpy as np
import pytest

from rocketlander.agents.pid import PIDAgent
from rocketlander.envs.physics import body_to_world
from rocketlander.envs.rocket_env import RocketLanderEnv
from rocketlander.render.renderer import Renderer
from rocketlander.render.scene import nozzle_geometry


def run(env, agent, renderer, steps):
    obs, _ = env.reset(seed=0)
    for _ in range(steps):
        obs, _, terminated, truncated, _ = env.step(agent.act(obs))
        renderer.draw(env.frame())
        if terminated or truncated:
            break


@pytest.mark.parametrize("level", ["L0", "L2", "L4"])
def test_draws_a_full_colour_image(level):
    renderer = Renderer(width=400, height=300)
    run(RocketLanderEnv(level=level), PIDAgent(), renderer, steps=20)
    image = renderer.to_array()
    assert image.shape == (300, 400, 3) and image.dtype == np.uint8
    assert len(np.unique(image.reshape(-1, 3), axis=0)) > 50
    renderer.close()


def test_env_rgb_array_render_mode():
    env = RocketLanderEnv(level="L1", render_mode="rgb_array")
    env.reset(seed=0)
    env.step(np.zeros(3, dtype=np.float32))
    assert env.render().shape[2] == 3
    env.close()
    assert RocketLanderEnv().render() is None
    with pytest.raises(ValueError, match="render_mode"):
        RocketLanderEnv(render_mode="ascii")


def test_frame_carries_controls_and_params():
    env = RocketLanderEnv(level="L0")
    env.reset(seed=0)
    assert env.frame().controls is None
    env.step(np.array([1.0, 0.5, 0.0], dtype=np.float32))
    frame = env.frame()
    assert frame.controls.throttle == 1.0 and frame.params.length > 0
    assert not frame.time_limit_reached


def test_crash_explodes_and_hides_the_rocket():
    env = RocketLanderEnv(level="L0")
    renderer = Renderer(width=400, height=300)
    falling = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    env.reset(seed=0)
    done = False
    while not done:
        _, _, done, _, info = env.step(falling)
        renderer.draw(env.frame())
    assert info["outcome"] == "crashed"
    assert renderer._exploded and len(renderer.particles.p) > 100
    renderer.close()


def test_overlay_keys_toggle_and_reset_clears_history():
    renderer = Renderer(width=400, height=300)
    assert renderer.toggle("f") and renderer.show["forces"]
    assert not renderer.toggle("x")
    renderer.telemetry.record(np.zeros(3), 1.0, 2.0)
    renderer.reset()
    assert len(renderer.telemetry.actions) == 0
    renderer.close()


def test_rendering_does_not_change_the_simulation():
    runs = []
    for mode in (None, "rgb_array"):
        env = RocketLanderEnv(level="L3", render_mode=mode)
        obs, _ = env.reset(seed=4)
        observations = []
        for _ in range(60):
            obs, *_ = env.step(PIDAgent().act(obs))
            env.render()
            observations.append(obs)
        runs.append(np.array(observations))
        env.close()
    assert np.array_equal(runs[0], runs[1])


def test_new_episode_clears_trail_and_particles():
    env = RocketLanderEnv(level="L0")
    renderer = Renderer(width=400, height=300)
    run(env, PIDAgent(), renderer, steps=30)
    assert len(renderer.telemetry.trail) > 1
    env.reset(seed=1)
    renderer.draw(env.frame())  # time jumps back to 0: a new episode
    assert len(renderer.telemetry.trail) == 1 and len(renderer.particles.p) == 0
    renderer.close()


def test_rcs_puffs_fire_on_the_side_that_produces_the_torque():
    env = RocketLanderEnv(level="L0")
    renderer = Renderer(width=400, height=300)
    env.reset(seed=0)
    env.step(np.array([-1.0, 0.0, 1.0], dtype=np.float32))  # engine off, CCW side jets
    renderer.draw(env.frame())
    rocket, puffs = env.frame().rocket, renderer.particles.p
    # A CCW torque from a nose jet pushes the nose left, so the exhaust blows out to the right.
    assert len(puffs) > 0 and np.all(puffs.pos[:, 0] > rocket.x) and np.all(puffs.vel[:, 0] > 0)
    renderer.close()


def test_effects_keep_moving_after_the_episode_ends():
    env = RocketLanderEnv(level="L0")
    renderer = Renderer(width=400, height=300)
    env.reset(seed=0)
    done = False
    while not done:
        _, _, done, _, _ = env.step(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
        renderer.draw(env.frame())
    at_crash = len(renderer.particles.p)
    for _ in range(30):  # one second of the outcome hold: sim time no longer advances
        renderer.draw(env.frame())
    assert len(renderer.particles.p) < 0.7 * at_crash
    renderer.close()


def test_human_render_mode_draws_during_step():
    env = RocketLanderEnv(level="L0", render_mode="human")
    env.reset(seed=0)
    env.step(np.zeros(3, dtype=np.float32))
    assert env._renderer is not None and env._renderer.window is not None
    env.close()


def test_nozzle_stays_above_the_leg_tips():
    env = RocketLanderEnv(level="L0")
    env.reset(seed=0)
    frame = env.frame()
    exit_point, _ = nozzle_geometry(frame.rocket, frame.params, gimbal=0.0)
    lowest_leg = body_to_world(frame.rocket, frame.params.leg_tips_body())[:, 1].min()
    assert exit_point[1] > lowest_leg


def test_particles_never_show_inside_the_hull():
    env = RocketLanderEnv(level="L0")
    renderer = Renderer(width=400, height=300)
    run(env, PIDAgent(), renderer, steps=2_000)
    deck, p = env.frame().deck, renderer.particles.p
    over_deck = np.abs(p.pos[:, 0] - deck.x) <= deck.half_width
    surface = deck.y + (p.pos[:, 0] - deck.x) * np.tan(deck.angle)
    assert not np.any(over_deck & (p.pos[:, 1] < surface))
    renderer.close()
