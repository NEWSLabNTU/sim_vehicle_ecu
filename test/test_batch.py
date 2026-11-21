"""The batched form (scensim roadmap 092 S3): step_batch over N slots equals N independent
scalar steps, bit for bit, for a vectorised component and for the LoopBatch of any other (ev,
the fmu over N instances); per-slot seeds; resets by mask; the checks name what is wrong."""

import numpy as np
import pytest

from sim_vehicle_ecu import batch as B
from sim_vehicle_ecu import component as comp
from sim_vehicle_ecu import signals
from sim_vehicle_ecu.signals import Request, Sensors, VescCommand

from batch_cars import DT, Lag, LagScalar, run_both, same, vesc_run
from test_calibration import MODEL3
from test_fmu import CC, build, params as fmu_params


@pytest.mark.parametrize("n", [1, 7, 64])
def test_a_vectorised_batch_equals_n_independent_scalar_cars_bit_for_bit(n):
    seeds = [100 + i for i in range(n)]
    b = B.load_batch(Lag, {}, {"noise": 0.01}, seeds)
    assert b.vectorised and b.n == n
    run_both(b, [LagScalar({}, {"noise": 0.01}, s) for s in seeds], steps=300)


def test_the_scalar_of_a_batch_is_the_batch_at_one_slot():
    a, s = Lag({}, {}, 3), LagScalar({}, {}, 3)
    for cmds, sens, _ in vesc_run(1, 200, seed=2):
        assert same(a.step(cmds[0], sens[0], DT), s.step(cmds[0], sens[0], DT))


def test_per_slot_seeds_are_what_single_cars_get():
    b = B.load_batch(Lag, {}, {"noise": 0.05}, [5, 5, 6])
    b.reset_batch(np.ones(3, bool))
    cmds = B.commands_to_batch("vesc", [VescCommand()] * 3)
    sens = B.sensors_to_batch([None] * 3, ("speed",))
    a = b.step_batch(cmds, sens, DT)["tire_angle"]
    assert a[0] == a[1] != a[2]                       # same seed, same draw; another, another
    assert a[0] == LagScalar({}, {"noise": 0.05}, 5).step(VescCommand(), None, DT).tire_angle


def test_loopbatch_of_ev_equals_n_ev_cars_and_is_flagged():
    seeds = [1, 2, 3, 4]
    b = B.load_batch("ev", MODEL3, None, seeds)
    assert isinstance(b, B.LoopBatch) and not b.vectorised and b.loop_of == "ev"
    rng = np.random.default_rng(4)
    runs = []
    for k in range(240):
        cmds = [Request() if k == 0 else Request(acceleration=float(rng.normal(0, 1.5)),
                                             velocity=float(abs(rng.normal(8, 3))),
                                             tire_angle=float(rng.normal(0, 0.1)))
                for _ in seeds]
        v = 0.05 * k
        sens = [None if k == 0 else Sensors((v / 0.37,) * 4, 0.01, 1.0, 0.0) for _ in seeds]
        runs.append((cmds, sens, rng.random(4) < 0.02))
    run_both(b, [comp.load("ev", MODEL3, None, s) for s in seeds], 0, kind="autoware",
             truth=(), runs=runs)
    with pytest.raises(B.BatchError, match=r"ev: has no batched form \(it runs as LoopBatch"):
        B.require_vectorised(b)
    assert B.require_vectorised(B.load_batch(Lag, {}, {}, [1])) is not None


@pytest.mark.skipif(CC is None, reason="no C compiler to build the test FMU")
def test_the_fmu_runs_as_a_loop_over_its_instances(tmp_path, monkeypatch):
    monkeypatch.setenv("SCENSIM_FMU_CACHE", str(tmp_path / "cache"))
    unit = build(tmp_path)
    p = fmu_params()
    seeds = [11, 12, 13]
    b = B.load_batch("fmu", {}, p, seeds, base=tmp_path)
    assert isinstance(b, B.LoopBatch) and not b.vectorised
    singles = [comp.load("fmu", {}, p, s, base=unit.parent) for s in seeds]
    runs = [([VescCommand(motor="speed", value=float(k % 5), servo=0.4 + 0.01 * i, age=0.0)
              for i in range(3)],
             [None if k == 0 else Sensors((1.0, 1.0, 2.0 + i, 2.0), 0.0, 0.0, 0.0)
              for i in range(3)], np.zeros(3, bool)) for k in range(50)]
    run_both(b, singles, 0, truth=(), runs=runs)
    once = build(tmp_path, once="true", name="once.fmu")
    with pytest.raises(comp.ComponentError, match="canBeInstantiatedOnlyOncePerProcess"):
        B.load_batch("fmu", {}, fmu_params(fmu="once.fmu"), [1, 2], base=once.parent)


def test_conversions_keep_none_and_types():
    cmds = [VescCommand(), VescCommand("duty", 0.3, 0.7, 0.01), VescCommand("brake", 5.0, None)]
    arr = B.commands_to_batch("vesc", cmds)
    assert list(arr["motor"]) == [-1, signals.MOTOR_MODES.index("duty"),
                                  signals.MOTOR_MODES.index("brake")]
    assert B.batch_to_commands("vesc", arr) == cmds
    reqs = [None, Request(1.0, 2.0, 0.1, signals.GEAR_DRIVE)]
    assert B.batch_to_commands("autoware", B.commands_to_batch("autoware", reqs)) == reqs
    sens = [None, Sensors((1.0, 2.0, 3.0, 4.0), 0.1, 0.2, 0.3, {"pose": (1.0, 2.0, 0.5)})]
    back = B.batch_to_sensors(B.sensors_to_batch(sens, ("pose",)), ("pose",))
    assert back == sens


def test_a_reading_without_a_declared_truth_is_marked_not_zeroed_silently():
    sens = [Sensors((1.0,) * 4, 0.0, 0.0, 0.0, {"speed": 2.0}), Sensors((1.0,) * 4, 0.0, 0.0, 0.0, {})]
    arr = B.sensors_to_batch(sens, ("speed",))
    assert list(arr["truth_valid"]["speed"]) == [True, False]
    assert B.batch_to_sensors(arr, ("speed",)) == sens


def test_reductions_are_written_out_in_both_forms():
    """Three orders of one sum of four that differ in the last bit (2.6): numpy's pairwise
    ((a + b) + (c + d)), the left-to-right loop, and Python 3.12's sum(), which is compensated
    (Neumaier) and so differs from both. Only the written-out order is the same everywhere."""
    import sys
    rng = np.random.default_rng(0)
    w = rng.normal(0, 1e3, (20000, 4)) + rng.normal(0, 1e-3, (20000, 4))
    pairwise = (w[:, 0] + w[:, 1]) + (w[:, 2] + w[:, 3])
    ordered = ((w[:, 0] + w[:, 1]) + w[:, 2]) + w[:, 3]
    loop = np.array([((r[0] + r[1]) + r[2]) + r[3] for r in w.tolist()])
    builtin = np.array([sum(r) for r in w.tolist()])
    assert np.array_equal(ordered, loop)
    assert not np.array_equal(pairwise, loop)
    if sys.version_info >= (3, 12):
        assert not np.array_equal(builtin, loop)


def test_telemetry_on_some_slots_only_is_named():
    a = signals.Actuation((0.0,) * 4, (0.0,) * 4, 0.0, {"erpm": 1.0})
    b = signals.Actuation((0.0,) * 4, (0.0,) * 4, 0.0, None)
    with pytest.raises(B.BatchError, match="slots 1, 2 report none, the others do"):
        B.actuations_to_batch([a, b, b])


def test_the_checks_name_the_field_and_slot():
    good = {"drive": np.zeros((2, 4)), "brake": np.zeros((2, 4)), "tire_angle": np.zeros(2)}
    assert B.checked_batch(good, 2) is good
    for bad, words in (({**good, "drive": np.zeros((2, 3))}, r"drive: shape \(2, 3\)"),
                       ({**good, "tire_angle": np.array([0.0, np.nan])},
                        "tire_angle: not finite in slot 1"),
                       ({**good, "brake": np.array([[0, 0, 0, 0], [0, -1, 0, 0]], float)},
                        "brake: negative in slot 1"),
                       ([], "not a dict of arrays")):
        with pytest.raises(B.BatchError, match=words):
            B.checked_batch(bad, 2)

    class NoReset:
        commands = "vesc"

        def __init__(self, *a):
            pass

        def step_batch(self, *a):
            pass

    class Wrapper:
        batch = NoReset

    with pytest.raises(B.BatchError, match="no reset_batch"):
        B.load_batch(Wrapper, {}, {}, [1])
    with pytest.raises(B.BatchError, match=r"a mask of shape \(3,\), not \(2,\)"):
        B.load_batch(LagScalar, {}, {}, [1, 2]).reset_batch(np.ones(3, bool))
