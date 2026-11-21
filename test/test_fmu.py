"""The `fmu` component (scensim roadmap 091 W4): an FMI 2.0 co-simulation unit stepped as a car's
control component. The unit is test/fmu_racer/racer.c, built here with cc against the FMI 2.0
headers in test/fmi2/ (BSD-2, test/fmi2/LICENSE.txt); skipped without a C compiler."""
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from sim_vehicle_ecu import component, signals
from sim_vehicle_ecu.component import ComponentError
from sim_vehicle_ecu import fmu as fmumod

HERE = Path(__file__).resolve().parent
CC = shutil.which("cc") or shutil.which("gcc")
pytestmark = pytest.mark.skipif(CC is None, reason="no C compiler to build the test FMU")


def build(out, variable="true", once="false", name="racer.fmu"):
    """The test unit as a .fmu at out/name."""
    work = out / f"build-{variable}-{once}"
    (work / "binaries" / "linux64").mkdir(parents=True, exist_ok=True)
    subprocess.run([CC, "-shared", "-fPIC", "-O2", "-I", str(HERE / "fmi2"),
                    str(HERE / "fmu_racer" / "racer.c"), "-o",
                    str(work / "binaries" / "linux64" / "racer.so")], check=True)
    xml = (HERE / "fmu_racer" / "modelDescription.xml.in").read_text()
    (work / "modelDescription.xml").write_text(xml.replace("@VARIABLE@", variable)
                                               .replace("@ONCE@", once))
    fmu = out / name
    with zipfile.ZipFile(fmu, "w") as z:
        for f in ("modelDescription.xml", "binaries/linux64/racer.so"):
            z.write(work / f, f)
    return fmu


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SCENSIM_FMU_CACHE", str(tmp_path / "cache"))


@pytest.fixture
def unit(tmp_path):
    return build(tmp_path)


def params(**over):
    p = {"fmu": "racer.fmu", "commands": "vesc",
         "inputs": {"speed_ref": "command.value", "servo": "command.servo",
                    "wheel_rl": "sensors.wheel_speeds[2]", "wheel_rr": "sensors.wheel_speeds[3]"},
         "outputs": {"drive[2]": "torque_rl", "drive[3]": "torque_rr", "tire_angle": "steer",
                     "telemetry.erpm": "erpm", "telemetry.steps": "steps"},
         "parameters": {"kp": 0.8}, "seed": "seed", "driven": [False, False, True, True]}
    p.update(over)
    return p


def car(unit, run_seed=0, **over):
    return component.load("fmu", {}, params(**over), run_seed, base=unit.parent)


VESC = signals.VescCommand(motor="speed", value=2.0, servo=0.6, age=0.0)
STILL = signals.Sensors(wheel_speeds=(0.0, 0.0, 10.0, 10.0))


def test_the_mapping_drives_the_wheels_and_the_tire(unit):
    c = car(unit)
    assert c.commands == "vesc" and c.driven == (False, False, True, True)
    assert len(c.fmu_sha256) == 64
    a = component.checked(c.step(VESC, STILL, 0.004))
    # err = 2.0 - 0.05 * 10 = 1.5; integ = 0.006; torque = 0.8 * 1.5 + 0.1 * 0.006, halved per wheel
    assert a.drive == (0.0, 0.0, pytest.approx(0.6003), pytest.approx(0.6003))
    assert a.brake == (0.0,) * 4
    assert a.tire_angle == pytest.approx((0.6 - 0.53) * -1.2)
    assert a.telemetry == {"erpm": pytest.approx(200.0), "steps": 1}
    assert isinstance(a.telemetry["steps"], int) and isinstance(a.drive[2], float)


def test_reset_starts_again_and_the_seed_draws_the_noise(unit):
    p = {"parameters": {"kp": 0.8, "noise_std": 0.5}}
    a, b, other = car(unit, 7, **p), car(unit, 7, **p), car(unit, 8, **p)
    run = [a.step(VESC, STILL, 0.004).drive for _ in range(20)]
    assert run == [b.step(VESC, STILL, 0.004).drive for _ in range(20)]   # same seed, same car
    assert run != [other.step(VESC, STILL, 0.004).drive for _ in range(20)]
    a.reset()
    assert [a.step(VESC, STILL, 0.004).drive for _ in range(20)] == run  # the integrator too
    assert a.step(VESC, STILL, 0.004).telemetry["steps"] == 21


def test_no_sensors_yet_and_no_servo_yet_hold_the_last_value(unit):
    c = car(unit)
    first = c.step(signals.VescCommand(), None, 0.004)   # before any command and any reading
    # servo held at its start value 0.53: no steer; speed_ref from value 0.0, wheels at start 0
    assert first.tire_angle == pytest.approx(0.0) and first.drive[2] == pytest.approx(0.0)
    c.step(VESC, STILL, 0.004)
    held = c.step(signals.VescCommand(motor="speed", value=2.0, servo=None), None, 0.004)
    assert held.tire_angle == pytest.approx((0.6 - 0.53) * -1.2)   # the servo stays where it was
    assert held.telemetry["erpm"] == pytest.approx(200.0)          # the wheels too


def test_the_vesc_motor_mode_is_its_index(unit, tmp_path):
    c = car(unit, inputs={"speed_ref": "command.motor"}, outputs={"telemetry.e": "erpm"})
    a = c.step(signals.VescCommand(motor="duty", value=0.1), STILL, 0.004)
    assert a.drive == (0.0,) * 4   # unmapped targets are 0
    v = car(unit, inputs={"speed_ref": "command.motor"}, outputs={"drive[0]": "torque_rl"})
    assert v.step(signals.VescCommand(motor="current"), signals.Sensors(), 0.004).drive[0] == \
        pytest.approx(0.8 * 2 / 2 + 0.1 * 2 * 0.004 / 2)   # current is mode 2


def test_ackermann_autoware_truth_dt_and_booleans(unit):
    c = car(unit, commands="ackermann", truth=["speed"],
            inputs={"speed_ref": "command.speed", "wheel_rl": "sensors.truth.speed",
                    "brake_on": "time"},
            outputs={"drive[0]": "torque_rl", "brake[1]": "brake"})
    s = signals.Sensors(truth={"speed": 20.0})
    a = c.step(signals.AckermannCommand(speed=1.0), s, 0.01)
    assert a.drive[0] == pytest.approx((0.8 * 0.5 + 0.1 * 0.5 * 0.01) / 2)   # 1 - 0.05 * 20 / 2
    assert a.brake == (0.0, 0.0, 0.0, 0.0)            # time 0 -> brake_on false
    assert c.step(signals.AckermannCommand(speed=1.0), s, 0.01).brake[1] == 2.0   # time 0.01
    w = car(unit, commands="autoware", inputs={"speed_ref": "command.velocity"},
            outputs={"drive[0]": "torque_rl"})
    assert w.step(None, None, 0.004).drive[0] == 0.0   # no command yet: speed_ref holds its start
    assert w.step(signals.Request(velocity=3.0), None, 0.004).drive[0] > 0.0
    assert component.load("fmu", {}, params(commands="autoware", inputs={"servo": "dt"},
                                             outputs={"tire_angle": "steer"}), 0,
                          base=unit.parent).step(None, None, 0.25).tire_angle == \
        pytest.approx((0.25 - 0.53) * -1.2)


@pytest.mark.parametrize("over, says", [
    ({"inputs": {"nope": "dt"}}, "input 'nope' is not a variable"),
    ({"inputs": {"torque_rl": "dt"}}, "has causality 'output', not input"),
    ({"outputs": {"drive[0]": "kp"}}, "has causality 'parameter', not output"),
    ({"parameters": {"speed_ref": 1.0}}, "has causality 'input', not parameter"),
    ({"seed": "brake_on"}, "has causality 'input', not parameter"),
    ({"inputs": {"servo": "command.steering_angle"}}, "a vesc command has"),
    ({"inputs": {"servo": "sensors.wheel_speeds[4]"}}, "sensors.wheel_speeds[0..3]"),
    ({"inputs": {"servo": "sensors.truth.speed"}}, "needs the name in params.truth"),
    ({"truth": ["pose"], "inputs": {"servo": "sensors.truth.pose"}}, "index it [0..2]"),
    ({"outputs": {"drive[4]": "torque_rl"}}, "one of drive[0..3]"),
    ({"commands": "joystick"}, "params.commands 'joystick'"),
    ({"truth": ["altitude"]}, "params.truth"),
    ({"gains": 1}, "unknown params ['gains']"),
])
def test_a_map_the_unit_cannot_take_is_refused_by_name(unit, over, says):
    with pytest.raises(ComponentError, match=__import__("re").escape(says)):
        car(unit, **over)


def test_the_unit_is_found_relative_to_the_spec_or_refused(unit, tmp_path):
    with pytest.raises(ComponentError, match="relative and its host gave no spec directory"):
        component.load("fmu", {}, params(), 0)
    assert component.load("fmu", {}, params(fmu=str(unit)), 0) is not None   # absolute
    with pytest.raises(ComponentError, match="cannot read"):
        component.load("fmu", {}, params(fmu="missing.fmu"), 0, base=tmp_path)
    (tmp_path / "bad.fmu").write_text("not a zip")
    with pytest.raises(ComponentError, match="not an FMU"):
        component.load("fmu", {}, params(fmu="bad.fmu"), 0, base=tmp_path)
    spec = {"powertrain": {"model": "component", "component": "fmu", "params": params()}}
    assert isinstance(component.for_spec(spec, base=unit.parent), fmumod.Fmu)


def test_a_fixed_step_unit_refuses_another_dt(tmp_path):
    fixed = build(tmp_path, variable="false", name="fixed.fmu")
    c = car(fixed, fmu="fixed.fmu")
    c.step(VESC, STILL, 0.004)
    c.step(VESC, STILL, 0.004)
    with pytest.raises(ComponentError, match="cannot take a variable step size"):
        c.step(VESC, STILL, 0.005)
    c.reset()
    c.step(VESC, STILL, 0.005)   # a new episode may take another rate
    with pytest.raises(ComponentError, match="not a positive number"):
        c.step(VESC, STILL, 0.0)


def test_one_process_steps_many_instances_unless_the_unit_says_once(unit, tmp_path):
    cars = [car(unit, run_seed=s, parameters={"noise_std": 0.5}) for s in range(4)]
    out = [c.step(VESC, STILL, 0.004).drive[2] for c in cars]
    assert len(set(out)) == 4   # each its own state and seed (a batched host loops them, 092 S3)
    once = build(tmp_path, once="true", name="once.fmu")
    first = car(once, fmu="once.fmu")
    with pytest.raises(ComponentError, match="canBeInstantiatedOnlyOncePerProcess"):
        car(once, fmu="once.fmu")
    first.close()
    car(once, fmu="once.fmu").close()   # the first ended: another may start


def test_a_unit_error_names_its_message(unit):
    """The unit's own error (here a non-positive step it is handed through its C API: step()
    refuses one before) and its logged message, which is the format string: a ctypes callback
    cannot read the C call's varargs."""
    c = car(unit)
    status = c._lib.fmi2DoStep(c._c, 0.0, -1.0, 1)
    assert status == 3 and c._log[-1] == "error: step size %g is not positive"
    with pytest.raises(ComponentError, match=r"fmi2DoStep\(t=0.0\) returned error \(error: step "
                                             r"size %g is not positive\)"):
        c._check(status, "fmi2DoStep(t=0.0)")


def test_it_is_cheap_enough(unit):
    c = car(unit)
    mean, worst = component.timing(c, VESC, STILL, 0.004, calls=2000)
    assert mean < component.budget(600)   # 0.1 ms per call at 600 Hz


def test_a_model_exchange_unit_or_another_fmi_version_is_refused():
    with pytest.raises(ComponentError, match="not a co-simulation unit"):
        fmumod.ModelDescription('<fmiModelDescription fmiVersion="2.0"><ModelExchange '
                                'modelIdentifier="m"/></fmiModelDescription>')
    with pytest.raises(ComponentError, match="FMI 2.0 units only"):
        fmumod.ModelDescription('<fmiModelDescription fmiVersion="3.0"/>')


def test_two_unpacks_of_one_unit_share_the_first_directory(unit, monkeypatch):
    """243: two processes making one unit at once must never write a binary the other may have
    loaded: each extracts aside and renames; the one whose rename loses uses the winner's."""
    first, sha = fmumod.unpack(unit)
    so = first / "binaries" / "linux64" / "racer.so"
    stamp = so.stat().st_mtime_ns
    again, _ = fmumod.unpack(unit)
    assert again == first and so.stat().st_mtime_ns == stamp   # found, not rewritten
    # the race: another process's rename won after this one found no marker
    import shutil
    shutil.rmtree(first)
    real = fmumod.os.rename

    def lose(src, dst):
        real(src, str(src) + ".winner")                # stands in for the other's directory
        real(str(src) + ".winner", dst)
        raise OSError(39, "Directory not empty")
    monkeypatch.setattr(fmumod.os, "rename", lose)
    won, _ = fmumod.unpack(unit)
    assert won == first and (won / ".unpacked").exists()
    assert not [d for d in fmumod.cache_dir().iterdir() if d.name.startswith(".")]  # temp gone
