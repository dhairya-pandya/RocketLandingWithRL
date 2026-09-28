import numpy as np

from rocketlander.render.effects import ParticleSystem

UP = np.array([0.0, 1.0])


def test_flame_amount_scales_with_throttle():
    counts = []
    for throttle in (0.0, 0.3, 1.0):
        system = ParticleSystem(seed=0)
        system.emit_flame(np.zeros(2), UP, throttle, dt=1 / 30)
        counts.append(len(system.p))
    assert counts[0] == 0 and 0 < counts[1] < counts[2]


def test_exhaust_moves_opposite_to_thrust():
    system = ParticleSystem(seed=0)
    system.emit_flame(np.zeros(2), UP, 1.0, dt=1 / 30)
    assert np.all(system.p.vel[:, 1] < 0)


def test_particles_expire_and_are_capped():
    system = ParticleSystem(seed=0, max_particles=100)
    system.explode(np.zeros(2), n=250)
    assert len(system.p) == 100
    for _ in range(60):
        system.update(dt=1 / 30)
    assert len(system.p) == 0


def test_fade_goes_from_one_to_zero():
    system = ParticleSystem(seed=0)
    system.emit_rcs(np.zeros(2), UP, dt=1 / 30)
    assert np.allclose(system.fade(), 1.0)
    system.update(dt=0.1)
    assert np.all(system.fade() < 1.0)


def test_wind_streaks_follow_the_wind():
    system = ParticleSystem(seed=0)
    for _ in range(30):
        system.emit_wind(np.zeros(2), np.full(2, 100.0), wind=8.0, dt=1 / 30)
    assert len(system.p) > 0 and np.all(system.p.streak) and np.all(system.p.vel[:, 0] > 0)
