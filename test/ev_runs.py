"""Recorded inputs for `ev`'s batched form against its scalar form (test_ev_batch.py, the bench):
per-slot scripted drives that force every branch of Ecu (ABS cycling, traction control, ESC,
regen blending and its slip control, holds, gears, sensor dropouts, resets), each slot its own
script, amplitudes and calibration. Open loop: the sensors are scripted, not a plant's, so the
batch and N scalar cars see the very same inputs. pytest-free (the bench uses it too)."""
import math

import numpy as np

from sim_vehicle_ecu import batch as B
from sim_vehicle_ecu.calibration import Calibration
from sim_vehicle_ecu.ecu import Ecu
from sim_vehicle_ecu.signals import (GEAR_DRIVE, GEAR_NEUTRAL, GEAR_PARK, GEAR_REVERSE,
                                     Request, Sensors)

R = 0.37
DT = 1 / 240
KINDS = ("abs", "tc", "regen_ice", "esc", "gears", "random", "turn")

# the structures a calibration may give the powertrain (one per batch)
CONFIGS = {
    "rear_open": {},
    "rear_locked": {"diff_type": "locked"},
    "rear_lsd": {"diff_type": "lsd", "diff_preload": 40.0, "diff_power_lock": 0.3,
                 "diff_coast_lock": 0.15},
    "front_open": {"drive_axle": "front"},
    "all_open_static": {"drive_axle": "all", "front_drive_share": 0.4,
                        "brake_distribution": "static"},
}


def slot_params(i, kind, rng):
    """Slot i's calibration overrides: varied per slot, and the switches its script needs."""
    p = {"esc_enabled": kind in ("esc", "turn") or i % 5 == 0,
         "regen_mu_recover_rate": 2.0 if i % 2 else 0.001,
         "abs_slip_release": float(rng.uniform(0.18, 0.22)),
         "brake_dead_time": [0.08, 0.0, 0.05, 0.1][i % 4],
         "brake_time_constant": [0.06, 0.06, 0.0, 0.04][i % 4],
         "motor_time_constant": [0.04, 0.0, 0.03][i % 3],
         "est_sideslip_washout": [2.0, 0.0][i % 2 if i % 7 == 3 else 0],
         "steer_time_constant": [0.0, 0.05][i % 2],
         "steer_rate_limit": [0.0, 0.8, 0.0][i % 3],
         "tc_enabled": i % 11 != 10,
         "abs_enabled": i % 13 != 12}
    if kind == "regen_ice":
        # ABS past the regen slip control's release (the Model 3's ABS releases first, at 0.2,
        # and its abs_event then takes regen away for the stop)
        p["abs_slip_release"] = 0.4
    return p


def _sens(rng, v, wheels, yaw=0.0, ax=0.0, ay=0.0, noise=0.01):
    w = tuple(float(x + rng.normal(0, noise)) for x in wheels)
    return Sensors(w, float(yaw + rng.normal(0, 0.001)), float(ax + rng.normal(0, 0.02)),
                   float(ay + rng.normal(0, 0.02)))


class Script:
    """One slot's drive: `step(k, dt) -> (Request, Sensors or None)`."""

    def __init__(self, kind, rng, steer):
        self.kind, self.rng, self.steer = kind, rng, steer
        self.v = float(rng.uniform(15, 24)) if kind in ("abs", "regen_ice", "esc", "turn") else 0.0
        self.t = 0.0
        self.angle = 0.0
        self.lockf = float(rng.choice([0.55, 0.66]))   # an abrupt lock finds a low grip

    def step(self, k, dt):
        rng, kind = self.rng, self.kind
        self.t += dt
        t = self.t
        ang = (0.25 * math.sin(0.7 * t) if self.steer else 0.0)
        if kind == "abs":
            if self.v > 0.05:
                req = Request(-8.0, 0.0, ang)
                ax = -7.5
                self.v = max(0.0, self.v + ax * dt)
                ph = k % 24
                f = 0.55 if ph < 7 else (0.9 if ph < 10 else 0.995)
                rear = 0.6 if (k // 24) % 3 == 1 and ph < 5 else 1.0
                wheels = [f * self.v / R, f * self.v / R, rear * self.v / R, rear * self.v / R]
            else:
                self.v, ax = 0.0, 0.0
                req = Request(0.0, 0.0, ang, GEAR_PARK if (k // 50) % 2 else GEAR_DRIVE)
                wheels = [0.0] * 4
            return req, _sens(rng, self.v, wheels, ax=ax, noise=0.002)
        if kind == "tc":
            ph = k % 400
            if ph < 250:
                req = Request(3.0, 15.0, ang)
                ax = 2.5
                self.v = self.v + ax * dt
                spin = (k % 30) < 10
                rear = (self.v * 1.5 + 1.5) / R if spin else self.v / R
                wheels = [self.v / R, self.v / R, rear, rear]
                if (k % 60) == 45:
                    ax = -0.5   # a drive with no push: the friction estimate holds
            elif ph < 320:
                req, ax = Request(0.0, self.v, ang), -0.1
                self.v = max(0.0, self.v + ax * dt)
                wheels = [self.v / R] * 4
            else:
                req, ax = Request(-1.5, 0.0, ang), -1.5
                self.v = max(0.0, self.v + ax * dt)
                wheels = [self.v / R] * 4
            return req, _sens(rng, self.v, wheels, ax=ax)
        if kind == "regen_ice":
            ph = k % 300
            if self.v < 4.0:
                self.v = float(rng.uniform(15, 22))
            req = Request(-3.0, 0.0, ang)
            if 60 <= ph < 100:      # the rear (driven) wheels slip; the car barely slows
                ax = -0.6
                lockf = self.lockf if ph < 80 else 0.8
                wheels = [self.v / R, self.v / R, lockf * self.v / R, lockf * self.v / R]
            elif 100 <= ph < 104:   # a driven wheel spins up as it lets go
                ax = -0.6
                wheels = [self.v / R, self.v / R, 1.1 * self.v / R, 1.1 * self.v / R]
            else:
                ax = -2.8
                wheels = [self.v / R] * 4
            self.v = max(0.0, self.v + ax * dt)
            return req, _sens(rng, self.v, wheels, ax=ax, noise=0.002)
        if kind == "esc":
            ph = k % 480
            req = Request(0.3 if ph < 300 else -0.5, self.v, ang)
            yaw = 0.6 if ph < 120 else (-0.6 if ph < 240 else 0.05)
            ay = 0.0 if ph < 240 else 4.0
            wheels = [self.v / R] * 4
            return req, _sens(rng, self.v, wheels, yaw=yaw, ay=ay)
        if kind == "turn":
            want = 0.3 if (k // 200) % 2 == 0 else -0.25
            req = Request(0.2, self.v, want)
            yaw = 0.3 * (self.v * math.tan(self.angle) / 3.005)   # turning far less
            self.angle += (max(-0.7, min(0.7, want)) - self.angle) * 0.2
            wheels = [self.v / R] * 4
            return req, _sens(rng, self.v, wheels, yaw=yaw, ay=self.v * yaw)
        if kind == "gears":
            ph = k % 600
            if ph < 100:
                self.v = 0.0
                return Request(0.0, 0.0, ang, GEAR_PARK), _sens(rng, 0.0, [0.0] * 4, noise=0.0)
            if ph < 250:            # reverse: away, then braking
                acc = 1.0 if ph < 200 else -1.0
                self.v = min(0.0, self.v - acc * dt) if ph < 200 else min(0.0, self.v + 1.0 * dt)
                return (Request(acc, 3.0 if ph < 200 else 0.0, ang, GEAR_REVERSE),
                        _sens(rng, self.v, [self.v / R] * 4, ax=-acc))
            if ph < 400:            # forward, then neutral coasting
                if ph < 330:
                    self.v += 2.0 * dt
                    return Request(2.0, 8.0, ang), _sens(rng, self.v, [self.v / R] * 4, ax=2.0)
                self.v = max(0.0, self.v - 0.2 * dt)
                return (Request(0.5, 8.0, ang, GEAR_NEUTRAL),
                        _sens(rng, self.v, [self.v / R] * 4, ax=-0.2))
            self.v = max(0.0, self.v - 2.0 * dt)
            return Request(-2.0, 0.0, ang), _sens(rng, self.v, [self.v / R] * 4, ax=-2.0)
        # random: anything, dropouts included
        req = Request(float(rng.normal(0, 3)), float(abs(rng.normal(6, 5))),
                      ang + (float(rng.normal(0, 0.2)) if self.steer else 0.0),
                      int(rng.choice([GEAR_DRIVE] * 6 + [GEAR_REVERSE, GEAR_NEUTRAL, GEAR_PARK])))
        if rng.random() < 0.06:
            return req, None
        self.v = float(np.clip(self.v + rng.normal(0, 0.3), -5, 30))
        wheels = [self.v / R + float(rng.normal(0, 3)) for _ in range(4)]
        return req, _sens(rng, self.v, wheels, yaw=float(rng.normal(0, 0.3)),
                          ax=float(rng.normal(0, 4)), ay=float(rng.normal(0, 3)), noise=0.0)


def make_runs(n, steps, seed=0, steer=False, dts=None, kinds=None):
    """(params per slot, runs): runs a list of (requests, sensors, resets, dt) per step."""
    rng = np.random.default_rng(seed)
    kinds = kinds or [k for k in KINDS if steer or k != "turn"]
    slot_kinds = [kinds[i % len(kinds)] for i in range(n)]
    params = [slot_params(i, k, rng) for i, k in enumerate(slot_kinds)]
    scripts = [Script(k, np.random.default_rng(seed * 1000 + i), steer)
               for i, k in enumerate(slot_kinds)]
    runs = []
    for k in range(steps):
        dt = dts(k) if dts else DT
        reqs, sens = [], []
        for s in scripts:
            q, x = s.step(k, dt)
            reqs.append(q)
            sens.append(None if k == 0 else x)
        resets = rng.random(n) < 0.002
        if k == steps // 2:
            resets[::4] = True
        runs.append((reqs, sens, resets, dt))
    return params, slot_kinds, runs


def ecu_state(e):
    """The scalar car's state, in the batch's terms (for the comparison)."""
    pt, est, esc = e.powertrain, e.estimator, e.esc
    return {"speed": est.speed, "sideslip": est.sideslip, "friction": est.friction,
            "motor_torque": pt.motor_torque, "pressure0": pt.pressure[0],
            "pressure1": pt.pressure[1], "tc_f": pt.tc.factor, "rsc_f": pt.rsc.factor,
            **{f"abs_f{i}": m.factor for i, m in enumerate(pt.abs)},
            **{f"esc_brake{i}": b for i, b in enumerate(esc.brake)},
            "motor_factor": esc.motor_factor, "tire_angle": e.tire_angle,
            "axle_known": pt.axle_mu is not None,
            "axle_mu": pt.axle_mu if pt.axle_mu is not None else None,
            "abs_event": pt.abs_event, "slip_event": pt.slip_event,
            "wind0": pt.wind[0], "wind1": pt.wind[1]}


def batch_state(b, i, host):
    g = lambda a: host(a)[i].item()   # noqa: E731
    return {"speed": g(b.speed), "sideslip": g(b.sideslip), "friction": g(b.friction),
            "motor_torque": g(b.motor_torque), "pressure0": g(b.pressure[0]),
            "pressure1": g(b.pressure[1]), "tc_f": g(b.tc_f), "rsc_f": g(b.rsc_f),
            **{f"abs_f{k}": g(b.abs_f[k]) for k in range(4)},
            **{f"esc_brake{k}": g(b.esc_brake[k]) for k in range(4)},
            "motor_factor": g(b.motor_factor), "tire_angle": g(b.tire_angle),
            "axle_known": bool(g(b.axle_known)),
            "axle_mu": g(b.axle_mu) if g(b.axle_known) else None,
            "abs_event": bool(g(b.abs_event)), "slip_event": bool(g(b.slip_event)),
            "wind0": g(b.wind[0]), "wind1": g(b.wind[1])}


def scalar_cars(vehicle, params, config):
    return [Ecu(Calibration.from_spec(vehicle, {**config, **p, "powertrain": "ev"}))
            for p in params]


def _bits(x):
    return np.asarray(x, dtype=np.float64).tobytes()


def _close(a, b, rtol, atol):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return bool(np.all(np.abs(a - b) <= atol + rtol * np.abs(b)))


def compare(batch, cars, runs, rtol=None, atol=0.0, hits=None, host=np.asarray):
    """Step both on `runs`; every actuation (drive, brake per wheel, tire angle) and the state
    above equal: bit for bit (rtol None), else within atol + rtol |scalar|. `hits`: a dict the
    batch's branch masks are summed into, per name (slot-steps). Returns the largest deviation
    of an actuation, relative (0.0 when bit for bit)."""
    n = len(cars)
    for c in cars:
        c.reset()
    batch.reset_batch(np.ones(n, bool))
    worst = 0.0
    for k, (reqs, sens, resets, dt) in enumerate(runs):
        if resets.any():
            batch.reset_batch(resets)
            for c, r in zip(cars, resets):
                if r:
                    c.reset()
        act = B.checked_batch(batch.step_batch(B.commands_to_batch("autoware", reqs),
                                               B.sensors_to_batch(sens), dt), n)
        got = {f: host(act[f]) for f in ("drive", "brake", "tire_angle")}
        if hits is not None:
            for name, m in batch.debug.items():
                hits[name] = hits.get(name, 0) + int(host(m).sum())
            hits["sensor_dropout"] = hits.get("sensor_dropout", 0) + sum(s is None for s in sens)
            hits["reset"] = hits.get("reset", 0) + int(resets.sum())
        for i, (c, q, s) in enumerate(zip(cars, reqs, sens)):
            want = c.step(q, s, dt)
            g = (got["drive"][i], got["brake"][i], got["tire_angle"][i])
            w = (want.drive, want.brake, want.tire_angle)
            if rtol is None:
                ok = all(_bits(a) == _bits(b) for a, b in zip(g, w))
            else:
                ok = all(_close(a, b, rtol, atol) for a, b in zip(g, w))
                for a, b in zip(g, w):
                    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
                    worst = max(worst, float(np.max(d / np.maximum(np.abs(b), 1.0))))
            assert ok, (k, i, "actuation", g, w)
            sb, sc = batch_state(batch, i, host), ecu_state(c)
            for key, val in sc.items():
                bv = sb[key]
                if isinstance(val, bool) or val is None or bv is None:
                    assert bv == val, (k, i, key, bv, val)
                elif rtol is None:
                    assert _bits(bv) == _bits(val), (k, i, key, bv, val)
                else:
                    assert _close(bv, val, rtol, max(atol, 1e-9)), (k, i, key, bv, val)
    return worst


# the branches each comparison must have taken (test_ev_batch.py asserts each hit > 0)
BRANCHES = (
    "abs_release", "abs_reapply", "abs_event", "tc_cut", "tc_active", "esc_oversteer",
    "esc_sideslip", "esc_cut", "hold", "park", "neutral", "reverse", "pushing", "braking",
    "regen_blend", "regen_capped", "regen_cut", "rsc_release", "rsc_runaway", "rsc_growing",
    "slip_event_start", "axle_mu_found", "axle_mu_followed", "axle_mu_raised",
    "axle_mu_raise", "axle_mu_forgotten", "brake_floor", "est_first", "est_standing", "est_braking",
    "est_driving", "est_mean", "est_slipping", "est_bound", "est_rising", "mu_limit", "mu_held",
    "mu_recover", "sensor_dropout", "reset")
STEERING = ("esc_understeer", "mu_understeer")


def folded(hits):
    """Per-wheel and per-axle counts summed under one name."""
    out = {}
    for k, v in hits.items():
        base = k.rstrip("0123").rstrip("_") if k[-1].isdigit() else k
        out[base] = out.get(base, 0) + v
    return out
