"""Test cars for the batched form (092 S3), pytest-free so the bench and Isaac's Python use
them too: a small VESC-like car in array form (LagBatch, Lag = its scalar form), the same car
in plain floats (LagScalar), a random run, and a bit-for-bit comparison."""
import math
import random

import numpy as np

from sim_vehicle_ecu import batch as B
from sim_vehicle_ecu import signals
from sim_vehicle_ecu.signals import Sensors, VescCommand

DT = 1 / 120


# A small VESC-like car, written once in array form (+ - * /, clip, where; one Python Random per
# slot, drawn in slot order as N single cars would): the contract's reference until the
# roboracer example's component takes its place (092 S3, with 118's W23).
class LagBatch:
    commands = "vesc"
    per_slot_params = True   # kt, servo_tau, servo_gain, noise: a number or an array [N]
    truth = ("speed",)
    driven = (False, False, True, True)

    def __init__(self, vehicle, params, seeds):
        n = len(seeds)
        arr = lambda k, d: np.broadcast_to(np.asarray(params.get(k, d), float), (n,)).copy()  # noqa
        self.kt = arr("kt", 0.05)          # N m per A at the wheel
        self.tau = arr("servo_tau", 0.05)
        self.gain = arr("servo_gain", 0.6)  # rad per unit servo from centre
        self.noise = arr("noise", 0.002)
        self.seeds = list(seeds)
        n = len(self.seeds)
        self.angle = np.zeros(n)
        self.rngs = [random.Random(s) for s in self.seeds]

    def reset_batch(self, mask):
        m = np.asarray(mask, bool)
        self.angle = np.where(m, 0.0, self.angle)
        for i in np.flatnonzero(m):
            self.rngs[i] = random.Random(self.seeds[i])

    def step_batch(self, c, s, dt):
        n = len(self.angle)
        servo = np.where(np.isnan(c["servo"]), 0.5, c["servo"])
        target = (servo - 0.5) * 2.0 * self.gain
        a = np.minimum(dt / self.tau, 1.0)
        # a slot with no noise draws nothing, as its single car does
        jitter = np.array([r.gauss(0.0, sd) if sd else 0.0
                           for r, sd in zip(self.rngs, self.noise.tolist())])
        self.angle = self.angle + (target - self.angle) * a + jitter
        current = np.where(c["motor"] == signals.MOTOR_MODES.index("current"), c["value"], 0.0)
        brake_a = np.where(c["motor"] == signals.MOTOR_MODES.index("brake"), c["value"], 0.0)
        stale = c["age"] > 0.5
        torque = np.where(stale, 0.0, np.clip(current * self.kt, -2.0, 2.0))
        # the sensors' speed slows the motor: back-EMF, only once there is a reading
        has = s["valid"] & s["truth_valid"]["speed"]   # a reading, and its speed in it
        speed = np.where(has, s["truth"]["speed"], 0.0)
        w = s["wheel_speeds"]   # the mean of four, added in the scalar order, not np.mean
        mean = np.where(s["valid"], (((w[:, 0] + w[:, 1]) + w[:, 2]) + w[:, 3]) / 4.0, 0.0)
        torque = torque - np.where(torque > 0, 0.01 * speed, 0.0)
        drive = np.zeros((n, 4))
        drive[:, 2] = drive[:, 3] = 0.5 * torque
        brake = np.zeros((n, 4))
        brake[:, 2] = brake[:, 3] = np.where(stale, 1.0, np.abs(brake_a) * self.kt)
        return {"drive": drive, "brake": brake, "tire_angle": self.angle,
                "telemetry": {"current_motor": current, "stale": stale, "wheel_mean": mean}}


class Lag(B.ScalarOf):
    batch = LagBatch


class LagScalar:
    """The same car in plain Python floats, written separately: the bit-for-bit reference."""
    commands, truth, driven = "vesc", ("speed",), (False, False, True, True)

    def __init__(self, vehicle, params, seed):
        self.kt, self.tau = float(params.get("kt", 0.05)), float(params.get("servo_tau", 0.05))
        self.gain, self.noise = float(params.get("servo_gain", 0.6)), float(params.get("noise", 0.002))
        self.seed = seed
        self.reset()

    def reset(self):
        self.angle, self.rng = 0.0, random.Random(self.seed)

    def step(self, c, s, dt):
        servo = 0.5 if c.servo is None else c.servo
        a = min(dt / self.tau, 1.0)
        self.angle = self.angle + ((servo - 0.5) * 2.0 * self.gain - self.angle) * a \
            + (self.rng.gauss(0.0, self.noise) if self.noise else 0.0)
        current = c.value if c.motor == "current" else 0.0
        brake_a = c.value if c.motor == "brake" else 0.0
        stale = c.age > 0.5
        torque = 0.0 if stale else min(max(current * self.kt, -2.0), 2.0)
        speed = s.truth["speed"] if s is not None and "speed" in s.truth else 0.0
        w = s.wheel_speeds if s is not None else None   # written out: sum() is compensated
        mean = (((w[0] + w[1]) + w[2]) + w[3]) / 4.0 if s is not None else 0.0   # on 3.12
        torque = torque - (0.01 * speed if torque > 0 else 0.0)
        b = 1.0 if stale else abs(brake_a) * self.kt
        return signals.Actuation((0.0, 0.0, 0.5 * torque, 0.5 * torque), (0.0, 0.0, b, b),
                                 self.angle, {"current_motor": current, "stale": stale,
                                              "wheel_mean": mean})


def vesc_run(n, steps, seed=7):
    """Commands and sensors for n slots: random modes, missing servo and motor, stale ages,
    a first step without a reading, and resets of random slots mid-run."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(steps):
        cmds, sens = [], []
        for i in range(n):
            mode = rng.choice([None, "current", "brake", "duty"])
            cmds.append(VescCommand(motor=None if mode is None else str(mode),
                                    value=float(rng.normal(0, 20)),
                                    servo=None if rng.random() < 0.1 else float(rng.random()),
                                    age=float(rng.choice([0.0, 0.01, 0.9, math.inf]))))
            sens.append(None if k == 0 or rng.random() < 0.05 else
                        Sensors(tuple(float(x) for x in rng.normal(10, 1, 4)),
                                float(rng.normal()), float(rng.normal()), float(rng.normal()),
                                {"speed": float(rng.normal(3, 1))} if rng.random() > 0.1 else {}))
        resets = rng.random(n) < 0.03
        out.append((cmds, sens, resets))
    return out


def same(a, b):
    """Actuations equal bit for bit (floats compared by their bits, telemetry by value)."""
    pack = lambda v: np.array(v, dtype=np.float64).tobytes()   # noqa: E731
    return (pack(a.drive) == pack(b.drive) and pack(a.brake) == pack(b.brake)
            and pack([a.tire_angle]) == pack([b.tire_angle])
            and (a.telemetry or {}) == (b.telemetry or {}))


def run_both(batch, singles, steps, kind="vesc", truth=("speed",), runs=None):
    n = len(singles)
    for c in singles:
        c.reset()
    batch.reset_batch(np.ones(n, bool))
    for cmds, sens, resets in (runs or vesc_run(n, steps)):
        if resets.any():
            batch.reset_batch(resets)
            for c, r in zip(singles, resets):
                if r:
                    c.reset()
        act = B.checked_batch(batch.step_batch(B.commands_to_batch(kind, cmds),
                                               B.sensors_to_batch(sens, truth), DT), n)
        got = B.batch_to_actuations(act)
        want = [c.step(cmd, s, DT) for c, cmd, s in zip(singles, cmds, sens)]
        for i, (g, w) in enumerate(zip(got, want)):
            assert same(g, w), (i, g, w)
