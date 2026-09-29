import numpy as np
import pytest

from rocketlander.agents.pid import PIDAgent
from rocketlander.agents.random_agent import RandomAgent
from rocketlander.cli import compare
from rocketlander.envs.landing import Outcome
from rocketlander.race import Race, _from_deck, _to_deck
from rocketlander.render.camera import Camera
from rocketlander.render.race_view import RaceRenderer, display_color


def test_racers_share_the_start_state_and_identical_agents_fly_identically():
    race = Race([("a", PIDAgent()), ("b", PIDAgent())], "L3", seed=5)
    for _ in range(200):
        race.step()
    a, b = race.racers
    assert a.frame.rocket == b.frame.rocket
    assert a.frame.wind_speed == b.frame.wind_speed


def test_race_finishes_and_stops_stepping_finished_racers():
    race = Race([("pid", PIDAgent()), ("random", RandomAgent(seed=0))], "L0", seed=1)
    while not race.finished:
        race.step()
    pid, random = race.racers
    assert pid.frame.judgement.outcome is Outcome.LANDED
    assert random.frame.judgement.outcome is not Outcome.LANDED
    frozen = random.frame
    race.step()
    assert random.frame is frozen


def test_a_landed_rocket_rides_along_with_the_deck():
    race = Race([("pid", PIDAgent())], "L2", seed=3)
    while not race.finished:
        race.step()
    racer = race.racers[0]
    before = race.rocket(racer)
    race.time += 5.0  # the ship keeps moving after touchdown
    after, deck = race.rocket(racer), race.deck()
    assert (after.x, after.y) != (before.x, before.y)
    assert np.allclose(_to_deck(after, deck), racer.pose_on_deck)


def test_deck_frame_round_trip():
    race = Race([("pid", PIDAgent())], "L2", seed=0)
    deck, rocket = race.deck(), race.racers[0].frame.rocket
    x, y, theta = _from_deck(_to_deck(rocket, deck), deck)
    assert np.allclose((x, y, theta), (rocket.x, rocket.y, rocket.theta))


def test_camera_frames_several_rockets_like_one_when_they_coincide():
    one, many = Camera(800, 600), Camera(800, 600)
    rocket, deck = np.array([30.0, 200.0]), np.array([0.0, 0.0])
    single = one.target(rocket, deck)
    double = many.target(np.stack([rocket, rocket]), deck)
    assert np.allclose(single[0], double[0]) and single[1] == double[1]
    wide = many.target(np.array([[-300.0, 200.0], [300.0, 200.0]]), deck)
    assert wide[1] < single[1]  # zooms out to fit both


def test_race_renderer_draws_every_racer_headless():
    renderer = RaceRenderer(window=False)
    race = Race([("pid", PIDAgent()), ("random", RandomAgent(seed=0))], "L1", seed=0)
    for _ in range(5):
        race.step()
        renderer.draw_race(race, "L1")
    image = renderer.to_array()
    assert image.shape == (720, 1100, 3) and image.std() > 0
    renderer.close()


def test_compare_cli_records_a_gif_and_runs_the_viewer_headless(tmp_path, capsys):
    out = tmp_path / "race.gif"
    compare.main(["--agent", "random", "random", "--level", "L0", "--record", str(out)])
    compare.main(["--agent", "pid", "random", "--viewer", "--headless", "--max-frames", "10"])
    assert out.exists() and "Wrote" in capsys.readouterr().out


def test_duplicate_labels_are_made_unique_and_each_racer_is_drawn():
    race = Race([("pid", PIDAgent()), ("pid", RandomAgent(seed=0))], "L0", seed=1)
    assert [r.label for r in race.racers] == ["pid", "pid 2"]
    renderer = RaceRenderer(window=False)
    while not race.finished:
        race.step()
        renderer.draw_race(race, "L0")
    assert renderer._wrecked == {1}  # only the random racer exploded
    renderer.close()


def test_finished_racers_that_did_not_land_are_dimmed():
    race = Race([("pid", PIDAgent()), ("random", RandomAgent(seed=0))], "L0", seed=1)
    while not race.finished:
        race.step()
    landed, failed = race.racers
    assert display_color(landed) == landed.color
    assert display_color(failed) != failed.color


def test_compare_cli_without_a_mode_is_a_usage_error(capsys):
    with pytest.raises(SystemExit):
        compare.main(["--agent", "pid"])
    assert "--viewer" in capsys.readouterr().err
