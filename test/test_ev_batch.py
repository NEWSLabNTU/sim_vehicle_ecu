"""`ev`'s batched form (sim_vehicle_ecu.ev_batch) against its scalar form, `Ecu`: N slots, each
its own scripted drive and calibration (ev_runs.py), the scalar car per slot alongside, every
actuation (drive and brake per wheel, tire angle) and the state behind it compared every step,
across resets of some slots mid-run; every branch asserted taken. Then torch against numpy, bit
for bit, on the CPU and on a GPU, and no host sync in a GPU step."""
import sys

import numpy as np
import pytest

from sim_vehicle_ecu import batch as B
from sim_vehicle_ecu.ecu import Ecu
from sim_vehicle_ecu.calibration import Calibration
from sim_vehicle_ecu.ev_batch import EvBatch
from sim_vehicle_ecu.signals import Request, Sensors

import ev_runs as E
from test_calibration import MODEL3

N, STEPS = 12, 1200
# Python 3.12's sum() is compensated: the scalar form's wheel sums have other last bits there
# (batch.py), so on 3.12 the comparison with the scalar form allows for them.
EXACT = sys.version_info < (3, 12)
# The tolerance where the forms cannot be equal: the tire angle's sin/cos/tan (xp.sin_cos, not
# math's; within 4 ulp) in a car that steers. Seen: 7e-16 relative at most.
RTOL = 1e-9


def batch_and_cars(config, params, **kw):
    per = [{**E.CONFIGS[config], **p} for p in params]
    return (B.load_batch("ev", MODEL3, per, list(range(len(per))), **kw),
            E.scalar_cars(MODEL3, params, E.CONFIGS[config]))


def test_ev_has_a_vectorised_batch():
    b = B.load_batch("ev", MODEL3, None, [1, 2, 3])
    assert isinstance(b, EvBatch) and b.vectorised and b.n == 3
    assert B.require_vectorised(b) is b
    assert b.driven == (False, False, True, True) and b.commands == "autoware"


@pytest.mark.parametrize("config", list(E.CONFIGS))
def test_each_slot_is_its_scalar_car_bit_for_bit(config):
    params, kinds, runs = E.make_runs(N, STEPS, seed=1)
    b, cars = batch_and_cars(config, params)
    hits = {}
    worst = E.compare(b, cars, runs, rtol=None if EXACT else 1e-12, hits=hits)
    assert worst == 0.0 or not EXACT
    got = E.folded(hits)
    need = E.BRANCHES if config == "rear_open" else (
        "abs_release", "abs_reapply", "tc_cut", "esc_oversteer", "hold", "reverse",
        "regen_blend", "slip_event_start", "axle_mu_found", "reset")
    missing = [k for k in need if not got.get(k)]
    assert not missing, (missing, got)
    if config in ("rear_locked", "rear_lsd"):
        assert got.get("diff_transfer"), got
    if config == "rear_lsd":
        assert got.get("diff_slipping"), got


def test_a_car_that_steers_is_its_scalar_car_within_the_trig_tolerance():
    params, kinds, runs = E.make_runs(21, STEPS, seed=3, steer=True)   # 3 slots of each kind
    b, cars = batch_and_cars("rear_open", params)
    hits = {}
    worst = E.compare(b, cars, runs, rtol=RTOL, atol=1e-9, hits=hits)
    assert worst < 1e-12
    got = E.folded(hits)
    missing = [k for k in E.BRANCHES + E.STEERING if not got.get(k)]
    assert not missing, (missing, got)


def test_the_step_length_may_change_every_step():
    """The lags' factors per dt, the dead time's ring as dt shrinks (1/60 s to 1/600 s)."""
    dts = lambda k: 1 / 60 if k < 150 else ((1 / 240, 1 / 120)[k % 2] if k < 500 else 1 / 600)  # noqa
    params, kinds, runs = E.make_runs(N, 900, seed=5, dts=dts)
    b, cars = batch_and_cars("rear_open", params)
    E.compare(b, cars, runs, rtol=None if EXACT else 1e-12)
    assert b.dead.k >= int(0.1 / (1 / 600)) + 3


def test_no_command_yet_is_a_default_request():
    b = B.load_batch("ev", MODEL3, None, [1, 2])
    car = Ecu(Calibration.from_spec(MODEL3, {"powertrain": "ev"}))
    b.reset_batch(np.ones(2, bool))
    s = Sensors((5.0,) * 4, 0.0, 0.0, 0.0)
    for k in range(50):
        cmds = B.commands_to_batch("autoware", [None, Request()])
        act = B.batch_to_actuations(b.step_batch(cmds, B.sensors_to_batch([s, s]), E.DT))
        want = car.step(Request(), s, E.DT)
        assert E._bits(act[0].brake) == E._bits(want.brake) == E._bits(act[1].brake)
        assert E._bits(act[0].drive) == E._bits(want.drive)


def test_the_scalar_of_the_batch_is_the_scalar_car():
    class EvOne(B.ScalarOf):
        batch = EvBatch
    params, kinds, runs = E.make_runs(6, 400, seed=7)
    for i in range(6):
        one = EvOne(MODEL3, {**params[i]}, 0)
        car = E.scalar_cars(MODEL3, [params[i]], {})[0]
        for reqs, sens, resets, dt in runs:
            if resets[i]:
                one.reset()
                car.reset()
            a, w = one.step(reqs[i], sens[i], dt), car.step(reqs[i], sens[i], dt)
            if EXACT:
                assert E._bits(a.drive + a.brake) == E._bits(w.drive + w.brake)


def test_the_powertrain_shape_is_one_for_every_slot():
    with pytest.raises(B.BatchError, match="'diff_type' differs across slots"):   # per_key's
        B.load_batch("ev", MODEL3, [{"diff_type": "open"}, {"diff_type": "locked"}], [1, 2])
    with pytest.raises(B.BatchError, match="'front_drive_share' differs between slots"):
        B.load_batch("ev", MODEL3, [{"drive_axle": "all", "front_drive_share": x}
                                    for x in (0.3, 0.5)], [1, 2])
    with pytest.raises(B.BatchError, match="drive_axle 'side'"):
        B.load_batch("ev", MODEL3, {"drive_axle": "side"}, [1])
    with pytest.raises(B.BatchError, match=r"a mask of shape \(3,\), not \(2,\)"):
        B.load_batch("ev", MODEL3, None, [1, 2]).reset_batch(np.ones(3, bool))


def test_per_slot_numbers_are_arrays_shared_ones_too():
    b = B.load_batch("ev", MODEL3, [{"esc_enabled": True}, {"esc_enabled": False}], [1, 2])
    assert b.p["esc_enabled"].tolist() == [True, False]
    assert b.p["mass"].tolist() == [1760.0, 1760.0]


# --- torch ----------------------------------------------------------------------------------

def _torch():
    return pytest.importorskip("torch")


def _devices():
    try:
        import torch
    except ImportError:
        return ["cpu"]
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _to(d, device):
    import torch
    return {k: (_to(v, device) if isinstance(v, dict) else
                torch.as_tensor(np.asarray(v), device=device)) for k, v in d.items()}


def _bits_t(a):
    a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
    return a.tobytes()


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("config", ["rear_open", "rear_lsd", "all_open_static"])
def test_torch_equals_numpy_bit_for_bit(device, config):
    """Steering included: xp.sin_cos is the same in every namespace (32 slots at most on a GPU
    shared with others)."""
    torch = _torch()
    n = 16
    params, kinds, runs = E.make_runs(n, 600, seed=11, steer=True)
    per = [{**E.CONFIGS[config], **p} for p in params]
    a = B.load_batch("ev", MODEL3, per, list(range(n)))
    t = B.load_batch("ev", MODEL3, per, list(range(n)), array_api="torch", device=device)
    assert t.xp.device.type == device
    a.reset_batch(np.ones(n, bool))
    t.reset_batch(torch.ones(n, dtype=torch.bool, device=device))
    for reqs, sens, resets, dt in runs:
        if resets.any():
            a.reset_batch(resets)
            t.reset_batch(torch.as_tensor(resets, device=device))
        c, s = B.commands_to_batch("autoware", reqs), B.sensors_to_batch(sens)
        x = B.checked_batch(a.step_batch(c, s, dt), n)
        y = B.checked_batch(t.step_batch(_to(c, device), _to(s, device), dt), n)
        assert y["drive"].device.type == device
        for f in ("drive", "brake", "tire_angle"):
            assert _bits_t(x[f]) == _bits_t(y[f]), f


def test_a_gpu_step_never_waits_for_the_device(monkeypatch):
    """After a warm step (the lag factors for this dt and the dead time's ring are made once,
    from the host), a step brings nothing back: any bool(), item(), tolist(), nonzero() or copy
    to the host raises, and so does any synchronising call under torch's sync debug mode."""
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    n = 32
    params, kinds, runs = E.make_runs(n, 80, seed=13, steer=True)
    t = B.load_batch("ev", MODEL3, params, list(range(n)), array_api="torch", device="cuda")
    t.reset_batch(torch.ones(n, dtype=torch.bool, device="cuda"))
    feed = [(_to(B.commands_to_batch("autoware", q), "cuda"), _to(B.sensors_to_batch(s), "cuda"),
             torch.as_tensor(r, device="cuda"), dt) for q, s, r, dt in runs]
    t.step_batch(feed[0][0], feed[0][1], feed[0][3])
    torch.cuda.synchronize()

    def refuse(name):
        def f(*a, **k):
            raise AssertionError(f"host sync in step_batch: Tensor.{name}")
        return f
    for name in ("__bool__", "item", "tolist", "nonzero", "__float__", "__int__", "cpu",
                 "numpy"):
        monkeypatch.setattr(torch.Tensor, name, refuse(name))
    monkeypatch.setattr(torch, "nonzero", refuse("torch.nonzero"))
    debug = hasattr(torch.cuda, "set_sync_debug_mode")
    if debug:
        torch.cuda.set_sync_debug_mode("error")
    try:
        for c, s, r, dt in feed[1:]:
            t.reset_batch(r)
            t.step_batch(c, s, dt)
    finally:
        if debug:
            torch.cuda.set_sync_debug_mode("default")


def test_a_steer_limit_past_sin_cos_is_refused():
    """xp.sin_cos is accurate for |x| <= 1.6 only (241): a car steering further is refused."""
    import pytest
    from sim_vehicle_ecu.batch import BatchError
    with pytest.raises(BatchError, match="max_steer_angle"):
        B.load_batch("ev", MODEL3, {"max_steer_angle": 1.7}, [1, 2])


def test_a_wheel_settling_from_a_wheelspin_is_the_scalar_cars():
    """Issue 106's detector change, slot for slot: the scalar's run of a wheel coming down from
    a wheelspin into regen at a release of 0.15 (test_regen_slip's case), its every request and
    sensor reading replayed into the batch, gives the same actuation bit for bit."""
    import test_regen_slip as R
    over = {"regen_slip_release": 0.15}
    c = Calibration.from_spec(MODEL3, dict(over, powertrain="ev"))
    feed = []

    class Rec(Ecu):
        def step(self, request, sensors, dt):
            out = super().step(request, sensors, dt)
            feed.append((request, sensors, dt, out))
            return out
    for spin in (1.3, 2.0):
        feed.clear()
        R.drive(c, -1.2, 1.0, 0.3, 0.2, sliding=1.0, spin=spin, ecu=Rec(c))
        b = B.load_batch("ev", MODEL3, over, [0])
        b.reset_batch(np.ones(1, bool))
        for request, sensors, dt, want in feed:
            act = B.batch_to_actuations(b.step_batch(B.commands_to_batch("autoware", [request]),
                                                     B.sensors_to_batch([sensors]), dt))[0]
            assert E._bits(act.drive) == E._bits(want.drive), spin
            assert E._bits(act.brake) == E._bits(want.brake), spin
