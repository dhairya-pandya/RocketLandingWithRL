import numpy as np
import pytest

from rocketlander.render.camera import MAX_SCALE, MIN_SCALE, Camera


def test_screen_and_world_round_trip_with_y_up():
    camera = Camera(800, 600)
    camera.center, camera.scale = np.array([10.0, 50.0]), 4.0
    assert np.allclose(camera.to_screen(np.array([10.0, 50.0])), [400, 300])
    assert camera.to_screen(np.array([10.0, 60.0]))[1] < 300  # higher in the world = up on screen
    points = np.array([[0.0, 0.0], [25.0, -3.0]])
    assert np.allclose(camera.to_world(camera.to_screen(points)), points)


def test_target_keeps_rocket_and_deck_on_screen():
    camera = Camera(800, 600)
    rocket, deck = np.array([120.0, 400.0]), np.array([0.0, 0.0])
    camera.center, camera.scale = camera.target(rocket, deck)
    for point in (rocket, deck):
        x, y = camera.to_screen(point)
        assert 0 <= x <= 800 and 0 <= y <= 600


def test_zoom_is_clamped():
    camera = Camera(800, 600)
    assert camera.target(np.array([0.0, 11.0]), np.zeros(2))[1] == MAX_SCALE
    assert camera.target(np.array([0.0, 50_000.0]), np.zeros(2))[1] == MIN_SCALE


def test_follow_snaps_first_then_moves_smoothly():
    camera = Camera(800, 600)
    camera.follow(np.array([0.0, 100.0]), np.zeros(2), dt=1 / 30)
    first = camera.center.copy()
    camera.follow(np.array([0.0, 300.0]), np.zeros(2), dt=1 / 30)
    target, _ = camera.target(np.array([0.0, 300.0]), np.zeros(2))
    assert first[1] < camera.center[1] < target[1]
    camera.reset()
    camera.follow(np.array([0.0, 300.0]), np.zeros(2), dt=1 / 30)
    assert camera.center[1] == pytest.approx(target[1])
