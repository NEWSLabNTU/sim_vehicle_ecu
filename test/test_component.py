"""The control component interface (scensim roadmap 091 step 2): the shipped onboard computers
behind it, a user's own found by entry point from a workspace of its own, and the host's
checks."""
import copy
import math
import sys
import textwrap
import unittest

import pytest

from sim_vehicle_ecu import component as comp
from sim_vehicle_ecu import Actuation, Calibration, Ecu, Request, Sensors

from test_calibration import MODEL3

DT = 1 / 60


def drive(c, steps=240):
    """A run's commands and the sensors a rolling car would read: the actuations, one a step."""
    out = []
    v = 0.0
    for k in range(steps):
        req = Request(acceleration=1.5 if k < 120 else -2.0, velocity=10.0 if k < 120 else 0.0,
                      tire_angle=0.1 * math.sin(k / 30))
        sensors = None if k == 0 else Sensors((v / 0.37,) * 4, 0.01, 1.0, 0.0)
        out.append(c.step(req, sensors, DT))
        v = max(0.0, v + (1.5 if k < 120 else -2.0) * DT)
    return out


def ideal_doc():
    d = copy.deepcopy(MODEL3)
    d["powertrain"]["model"] = "ideal"
    return d


# A user's own component, in a package of its own outside this repository: here a sample that
# realizes Autoware's request as `ideal` does, through the public Ecu.
SAMPLE = textwrap.dedent('''
    from sim_vehicle_ecu import Calibration, Ecu


    class SampleIdeal:
        commands = "autoware"

        def __init__(self, vehicle, params, seed):
            self.seen = (dict(params), seed)
            self.ecu = Ecu(Calibration.from_spec(vehicle, {"powertrain": "ideal"}))

        def reset(self):
            self.ecu.reset()

        def step(self, command, sensors, dt):
            return self.ecu.step(command, sensors, dt)


    class NoStep:
        commands = "autoware"

        def __init__(self, vehicle, params, seed):
            pass

        def reset(self):
            pass


    class Telepathic:
        commands = "telepathy"

        def __init__(self, vehicle, params, seed):
            pass

        def reset(self):
            pass

        def step(self, command, sensors, dt):
            return None
''')


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A workspace's site-packages with the sample package and its dist-info, on the path."""
    site = tmp_path / "ws" / "install" / "lib" / "python3" / "site-packages"
    (site / "my_racer_model").mkdir(parents=True)
    (site / "my_racer_model" / "__init__.py").write_text(SAMPLE)
    (site / "broken_model").mkdir()
    (site / "broken_model" / "__init__.py").write_text("raise ImportError('needs rclpy')\n")
    info = site / "my_racer_model-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: my-racer-model\nVersion: 0.1.0\n")
    (info / "entry_points.txt").write_text(textwrap.dedent('''
        [scensim.vehicle_components]
        my_racer.sample_ideal = my_racer_model:SampleIdeal
        my_racer.no_step = my_racer_model:NoStep
        my_racer.telepathic = my_racer_model:Telepathic
        my_racer.broken = broken_model:Whatever
    '''))
    monkeypatch.syspath_prepend(str(site))
    yield site
    for m in ("my_racer_model", "broken_model"):
        sys.modules.pop(m, None)


class TestShipped(unittest.TestCase):
    def test_the_spec_s_model_is_the_shipped_onboard_computers(self):
        for model, cls in (("ideal", comp.Ideal), ("ev", comp.Ev)):
            d = copy.deepcopy(MODEL3)
            d["powertrain"]["model"] = model
            c = comp.for_spec(d)
            self.assertIsInstance(c, cls)
            self.assertEqual(c.commands, "autoware")
            want = drive(Ecu(Calibration.from_spec(d)))
            self.assertEqual(drive(c), want)

    def test_overrides_reach_the_calibration_and_the_driven_wheels_follow_it(self):
        c = comp.for_spec(MODEL3, {"esc_enabled": True})
        self.assertTrue(c.calibration.esc_enabled)
        self.assertEqual(c.driven, (False, False, True, True))  # the Model 3 RWD
        self.assertEqual(comp.for_spec(ideal_doc()).driven, (True,) * 4)
        self.assertEqual(c.brake_floor, c.calibration.brake_floor)

    def test_a_component_powertrain_is_not_an_ecu_powertrain(self):
        d = copy.deepcopy(MODEL3)
        d["powertrain"] = {"model": "component", "component": "x"}
        with self.assertRaisesRegex(ValueError, "the user's whole control component"):
            Ecu(Calibration.from_spec(d))


def test_a_users_component_by_entry_point_drives_as_ideal(workspace):
    """091 step 2's exit, at the library: a component from a workspace outside this repository,
    named by its entry point, steps the shipped car exactly as `ideal`."""
    d = ideal_doc()
    d["powertrain"] = {"model": "component", "component": "my_racer.sample_ideal",
                       "params": {"erpm_gain": 4614.0}}
    c = comp.for_spec(d, seed=7)
    assert type(c).__name__ == "SampleIdeal" and c.seen == ({"erpm_gain": 4614.0}, 7)
    assert drive(c) == drive(comp.for_spec(ideal_doc()))
    # a run's override names another, with its own params
    c = comp.for_spec(MODEL3, component={"entry_point": "my_racer.sample_ideal",
                                         "params": {"x": 1}})
    assert c.seen == ({"x": 1}, 0)


def test_what_is_refused_names_the_fix(workspace):
    with pytest.raises(comp.ComponentError, match="no control component 'nope'.*known: .*ev"):
        comp.load("nope", MODEL3)
    with pytest.raises(comp.ComponentError, match="'my_racer.broken'.*does not import.*rclpy"):
        comp.load("my_racer.broken", MODEL3)
    with pytest.raises(comp.ComponentError, match="no step"):
        comp.load("my_racer.no_step", MODEL3)
    with pytest.raises(comp.ComponentError, match="commands = 'telepathy'"):
        comp.load("my_racer.telepathic", MODEL3)


def test_an_actuation_is_checked():
    ok = Actuation((1.0, 1.0, 0.0, 0.0), (0.0,) * 4, 0.1)
    assert comp.checked(ok) is ok
    assert Actuation((0.0,) * 4, (0.0,) * 4, 0.0, {"erpm": 0.0}).telemetry == {"erpm": 0.0}
    for bad, why in ((Actuation((math.nan, 0, 0, 0), (0.0,) * 4, 0.0), "not finite"),
                     (Actuation((0.0,) * 3, (0.0,) * 4, 0.0), "3 drive"),
                     (Actuation((0.0,) * 4, (0.0, -1.0, 0.0, 0.0), 0.0), "negative brake"),
                     (None, "not an Actuation")):
        with pytest.raises(comp.ComponentError, match=why):
            comp.checked(bad)


def test_the_cost_budget_and_its_timing():
    assert comp.budget(600) == pytest.approx(1e-4)
    assert comp.budget(60) == pytest.approx(1e-3)
    mean, worst = comp.timing(comp.for_spec(MODEL3), Request(1.0, 5.0), None, DT, calls=50)
    assert 0 < mean <= worst


def test_ground_truth_is_only_what_a_component_declares():
    """The user, 2026-11-20 (scensim 091): a component gets the car's own sensors by default, and
    ground truth only as declared from TRUTH_INPUTS (what the user also measured on the car)."""
    class Declares:
        commands = "autoware"
        truth = ("speed", "pose")

        def reset(self):
            pass

        def step(self, command, sensors, dt):
            return None

    assert comp.check(Declares()).truth == ("speed", "pose")
    Declares.truth = ("speed", "telepathy")
    with pytest.raises(comp.ComponentError, match="truth = .*telepathy"):
        comp.check(Declares())
    Declares.truth = "speed"
    with pytest.raises(comp.ComponentError, match="a tuple of"):
        comp.check(Declares())
    assert Sensors().truth is None


def test_the_truth_feed_delays_adds_noise_and_gives_only_the_declared():
    values = lambda k: {"speed": float(k), "pose": (float(k), 0.0, 0.0), "yaw": 9.0}
    feed = comp.TruthFeed(("speed", "pose"), {"speed": {"latency_s": 0.05}})
    got = [feed(k * 0.01, values(k)) for k in range(10)]
    assert set(got[0]) == {"speed", "pose"}                    # only the declared
    assert [g["speed"] for g in got] == [0, 0, 0, 0, 0, 0, 1, 2, 3, 4]  # 0.05 s late
    assert [g["pose"][0] for g in got] == list(range(10))     # no latency set: as is
    noisy = comp.TruthFeed(("speed",), {"speed": {"noise_std": 0.1}}, seed=3)
    a = [noisy(k * 0.01, values(k))["speed"] for k in range(5)]
    noisy.reset()
    assert a == [noisy(k * 0.01, values(k))["speed"] for k in range(5)]  # seeded
    assert a != [float(k) for k in range(5)]
    assert comp.TruthFeed((), {})(0.0, values(0)) is None
    with pytest.raises(comp.ComponentError, match="does not declare"):
        comp.TruthFeed(("speed",), {"pose": {"latency_s": 0.1}})
    with pytest.raises(comp.ComponentError, match="unknown"):
        comp.TruthFeed(("speed",), {"speed": {"delay": 0.1}})
