"""Session inputs (scensim roadmap 091 W12): a component may declare the session's conditions it
needs, from SESSION_INPUTS (today the battery's voltage). They are constants for the run, so the
component gets them when it is constructed, `session={name: value}` (a batch class: arrays [N],
slot i's from slot i's record), not in each step's sensors, which are None on the first step
after a reset. A class that declares nothing is constructed as before. A declared input the
run's session record (`scensim.session/1`) lacks is refused when the component is loaded,
naming the record's key (SESSION_SOURCES)."""
import numpy as np
import pytest

from sim_vehicle_ecu import batch as B
from sim_vehicle_ecu import component as comp
from sim_vehicle_ecu.signals import Actuation

VEHICLE = {"geometry": {"wheel_radius": 0.05}}


def record(v):
    return {"schema": "scensim.session/1", "battery": {"voltage_v": v, "source": "measured"}}


class Declares:
    commands = "vesc"
    session = ("battery_v",)

    def __init__(self, vehicle, params, seed, session=None):
        self.got = session

    def reset(self):
        pass

    def step(self, command, sensors, dt):
        return Actuation((0.0,) * 4, (0.0,) * 4, 0.0)


class Plain:
    commands = "vesc"

    def __init__(self, vehicle, params, seed):
        self.made = True

    def reset(self):
        pass

    def step(self, command, sensors, dt):
        return Actuation((0.0,) * 4, (0.0,) * 4, 0.0)


@pytest.fixture
def named(monkeypatch):
    table = {"declares": Declares, "plain": Plain}
    monkeypatch.setattr(comp, "resolve", lambda name: table[name])


def test_battery_voltage_is_a_session_input_with_its_source():
    assert "battery_v" in comp.SESSION_INPUTS
    assert comp.SESSION_SOURCES["battery_v"] == ("battery", "voltage_v")


def test_a_declared_input_is_given_at_construction(named):
    c = comp.load("declares", VEHICLE, {}, 0, session=record(7.92))
    assert c.got == {"battery_v": 7.92}


def test_a_class_declaring_nothing_is_constructed_as_before(named):
    assert comp.load("plain", VEHICLE, {}, 0, session=record(7.92)).made
    assert comp.load("plain", VEHICLE, {}, 0).made


def test_a_declared_input_the_session_lacks_is_refused_naming_the_key(named):
    with pytest.raises(comp.ComponentError, match=r"session\.battery\.voltage_v"):
        comp.load("declares", VEHICLE, {}, 0)
    with pytest.raises(comp.ComponentError, match=r"session\.battery\.voltage_v"):
        comp.load("declares", VEHICLE, {}, 0, session={"schema": "scensim.session/1"})
    with pytest.raises(comp.ComponentError, match="voltage_v"):
        comp.load("declares", VEHICLE, {}, 0, session=record("seven"))


def test_only_known_session_inputs_may_be_declared(named, monkeypatch):
    monkeypatch.setattr(Declares, "session", ("battery_v", "mood"))
    with pytest.raises(comp.ComponentError, match="session = .*mood"):
        comp.load("declares", VEHICLE, {}, 0, session=record(7.9))
    monkeypatch.setattr(Declares, "session", "battery_v")
    with pytest.raises(comp.ComponentError, match="a tuple of"):
        comp.load("declares", VEHICLE, {}, 0, session=record(7.9))


def test_for_spec_hands_the_session_through(named):
    c = comp.for_spec(VEHICLE, component={"entry_point": "declares"}, session=record(8.1))
    assert c.got == {"battery_v": 8.1}


class DeclaresBatch:
    commands = "vesc"
    truth = ()
    session = ("battery_v",)

    def __init__(self, vehicle, params, seeds, session=None):
        self.got = session

    def reset_batch(self, mask):
        pass

    def step_batch(self, commands, sensors, dt):
        return None


class Batched:
    batch = DeclaresBatch
    commands = "vesc"


def test_a_batch_gets_each_slots_value_as_an_array():
    b = B.load_batch(Batched, VEHICLE, {}, seeds=[1, 2], session=[record(7.4), record(8.4)])
    assert np.allclose(b.got["battery_v"], [7.4, 8.4])
    one = B.load_batch(Batched, VEHICLE, {}, seeds=[1, 2, 3], session=record(7.9))
    assert np.allclose(one.got["battery_v"], [7.9, 7.9, 7.9])
    with pytest.raises(comp.ComponentError, match=r"session\.battery\.voltage_v"):
        B.load_batch(Batched, VEHICLE, {}, seeds=[1])
    with pytest.raises(B.BatchError, match="2 session records for 3 slots"):
        B.load_batch(Batched, VEHICLE, {}, seeds=[1, 2, 3], session=[record(7.4), record(8.4)])


def test_a_loop_batch_builds_each_slot_with_its_own_value(named):
    b = B.load_batch("declares", VEHICLE, {}, seeds=[1, 2], session=[record(7.4), record(8.4)])
    assert [c.got for c in b.components] == [{"battery_v": 7.4}, {"battery_v": 8.4}]


def test_a_scalar_of_a_batch_gets_its_value():
    class Scalar(B.ScalarOf):
        batch = DeclaresBatch

    s = comp.check(Scalar(VEHICLE, {}, 0, session={"battery_v": 7.7}))
    assert np.allclose(s._b.got["battery_v"], [7.7])
    assert comp.declared_session(Scalar) == ("battery_v",)
