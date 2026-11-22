"""`ev` in batched form: N cars' onboard computers stepped at once on arrays (sim_vehicle_ecu
.batch's contract; numpy or torch, on the CPU or a GPU, through sim_vehicle_ecu.xp).

    b = batch.load_batch("ev", vehicle, params, seeds, array_api="torch", device="cuda")

It is `Ecu` with the `ev` powertrain (ecu.py, estimator.py, esc.py, powertrain.py, brake.py,
actuators.py), written once more over arrays: every branch of the scalar form is a mask, every
expression the scalar's in its order, so slot i's actuation is the scalar car's bit for bit
(test/test_ev_batch.py drives both). Where it cannot be:

- The tire angle's sin, cos and tan (the estimator's wheel places, the ESC's and the understeer
  test's bicycle yaw rate) are `xp.sin_cos`, not `math`'s: within a few ulp of them, the same
  in every namespace. A car that steers differs from the scalar one in the last bits; one that
  does not (a tire angle of 0) does not differ at all.
- The first-order lags' `1 - exp(-dt / tau)` are computed on the host with `math.exp`, per slot
  and per `dt` (cached), as the scalar computes them: the same bits.
- The wheels' sums are written out in the scalar's order, from 0 as Python 3.10's `sum()`
  adds; Python 3.12's `sum()` is compensated, so on 3.12 the scalar form itself has other last
  bits.
- An `autoware` command that is not `valid` (none yet) is `Request()`: the scalar `Ecu` takes
  none (it reads `request.tire_angle`).

`params` are calibration overrides by field name, as `ev`'s, per slot (`per_slot_params`):
numbers and bools may differ between slots; the powertrain's structure (`drive_axle`,
`front_drive_share`, `diff_type`, `brake_distribution`) may not.

The standard library only at import: numpy (or torch) comes in through the namespace.
"""
import math

from . import xp as xpmod
from .actuators import RPM
from .calibration import G, Calibration
from .estimator import MU_TIME_CONSTANT, SIDESLIP_MIN_SPEED
from .powertrain import AXLE_MU_TIME_CONSTANT, ROLLING_RAMP, EvPowertrain
from .signals import GEAR_DRIVE, GEAR_NEUTRAL, GEAR_PARK, GEAR_REVERSE

# What gives the powertrain its shape: one value for every slot.
STRUCTURAL = ("powertrain", "drive_axle", "front_drive_share", "diff_type",
              "brake_distribution")
# The calibration's numbers and bools the batch reads, as arrays [N].
FIELDS = (
    "wheel_base", "wheel_radius", "max_steer_angle", "mass", "wheel_mass", "tire_friction",
    "gear_ratio", "motor_peak_torque", "motor_peak_power", "motor_time_constant",
    "regen_max_torque", "regen_max_power", "regen_fade_low", "regen_fade_high",
    "max_brake_torque", "hold_brake_torque", "brake_dead_time", "brake_time_constant",
    "brake_floor", "abs_enabled", "abs_slip_release", "abs_slip_reapply", "abs_release_rate",
    "abs_apply_rate", "abs_min_speed", "tc_enabled", "tc_slip_limit", "tc_slip_reapply",
    "tc_release_rate", "tc_apply_rate", "regen_slip_control", "regen_slip_release",
    "regen_slip_reapply", "regen_runaway_slip", "regen_runaway_decel", "regen_slip_growth",
    "regen_spin_settle", "regen_release_rate", "regen_apply_rate", "regen_mu_recover_rate",
    "diff_preload", "diff_power_lock", "diff_coast_lock", "esc_enabled", "esc_min_speed",
    "esc_understeer", "esc_dead_band", "esc_dead_band_share", "esc_gain", "esc_sideslip_limit",
    "esc_sideslip_gain", "esc_max_brake_torque", "esc_motor_cut_rate", "esc_motor_restore_rate",
    "steer_time_constant", "steer_rate_limit", "est_slip_offset", "est_sideslip_washout",
    "est_mu_margin", "est_mu_floor", "est_mu_max", "est_mu_recover_rate",
)
# xp.sin_cos is accurate for |x| <= 1.6 (review of 2026-11-23): the tire angle, clipped to
# max_steer_angle, must stay within it
SIN_COS_LIMIT = 1.6


def _derived(c):
    """The calibration's own sub-expressions, as the scalar form evaluates them (same order, the
    same Python `**`), so a step need not redo them."""
    r2 = c.wheel_radius ** 2
    w_max = c.motor_max_rpm * RPM
    taper_from = 0.95 * w_max
    return {
        "half": c.wheel_tread / 2,
        "com0": c.com[0], "com2": c.com[2],
        "r2": r2,
        "mass_ff": c.mass + 4 * 0.5 * c.wheel_mass,        # feedforward's mass
        "damp4": 4 * c.wheel_damping_rate,                  # feedforward's driveline drag
        "damp": c.wheel_damping_rate,
        "k_roll": c.rolling_coefficient * c.mass * G,
        "k_drag": 0.5 * c.air_density * c.drag_coefficient * c.frontal_area,
        "mu_g": c.tire_friction * G,
        "w_max": w_max, "taper_from": taper_from, "taper_den": w_max - taper_from,
        "fade_den": c.regen_fade_high - c.regen_fade_low,
        "inertia": 0.5 * c.wheel_mass * r2,                 # differential's; force's I/r^2 part
        "grip": c.tire_friction * c.mass * G * c.wheel_radius / 2,
        "deficit_keep": 1.0 - c.regen_runaway_deficit,
        "rsc_release_eps": c.regen_slip_release + 1e-6,
        "abs_creep": 0.25 * c.abs_apply_rate, "tc_creep": 0.25 * c.tc_apply_rate,
        "rsc_creep": 0.25 * c.regen_apply_rate,
        "half_sideslip_limit": 0.5 * c.esc_sideslip_limit,
        "esc_target_margin": c.esc_target_margin,
        "neg_esc_gain": -c.esc_gain,
        "grip_share": c.regen_grip_share,
        "neg_reapply": -c.regen_slip_reapply,
    }


class _Lags:
    """Per `dt`, per slot, the scalar form's `1 - exp(...)` factors, computed on the host with
    math.exp exactly as it computes them (the same bits), then kept on the batch's device."""

    def __init__(self, xp, cals):
        self.xp, self.cals, self.cache = xp, cals, {}

    def __call__(self, dt):
        got = self.cache.get(dt)
        if got is None:
            if len(self.cache) > 64:
                self.cache.clear()
            rows = {k: [] for k in ("steer", "gain", "slipping", "bound", "esc", "motor",
                                    "brake")}
            for c in self.cals:
                tau = c.steer_time_constant
                rows["steer"].append(1 - math.exp(-dt / tau) if tau > 0 else 0.0)
                for k, g in (("gain", c.est_speed_gain), ("slipping", c.est_speed_gain_slipping),
                             ("bound", c.est_speed_gain_bound)):
                    rows[k].append(1 - math.exp(-g * dt))
                tau = c.brake_time_constant
                rows["esc"].append(1 - math.exp(-dt / tau) if tau > 0 else 1.0)
                rows["brake"].append(1 - math.exp(-dt / tau) if tau > 0 else 0.0)
                tau = c.motor_time_constant
                rows["motor"].append(1 - math.exp(-dt / tau) if tau > 0 else 0.0)
            got = {k: self.xp.asarray(v, self.xp.float64) for k, v in rows.items()}
            n = len(self.cals)
            # divisors as arrays: torch on CUDA divides by a Python number as a product with
            # its reciprocal, which is not the quotient's bits
            got["dt"], got["dt2"], got["dt4"] = (self.xp.full(n, x) for x in (dt, 2 * dt, 4 * dt))
            got["mu"] = 1 - math.exp(-dt / MU_TIME_CONSTANT)
            got["axle_mu"] = 1 - math.exp(-dt / AXLE_MU_TIME_CONSTANT)
            self.cache[dt] = got
        return got


class _DeadTime:
    """actuators.DeadTime for N slots and both axles: what went in `delay` s earlier, at the
    steps' own times (0 before that). A ring of (time, value) per slot, one write per step for
    every slot; the output is the newest entry at or before `t - delay + 1e-9`, as the scalar's
    queue pops them. Its length follows the smallest `dt` seen (a host decision: no sync)."""

    def __init__(self, xp, n, delay, delay_max, width):
        self.xp, self.n, self.delay, self.delay_max, self.width = xp, n, delay, delay_max, width
        self.rows = xp.arange(n)
        self.k, self.head = 0, 0
        self.times = xp.full((n, 0), xp.inf)
        self.values = xp.zeros((n, 0, width))
        self.t = xp.zeros(n)

    def reset(self, m):
        xp = self.xp
        self.t = xp.where(m, 0.0, self.t)
        if self.k:
            self.times = xp.where(m[:, None], xp.inf, self.times)

    def _grow(self, k):
        xp = self.xp
        times, values = xp.full((self.n, k), xp.inf), xp.zeros((self.n, k, self.width))
        if self.k:
            order = xp.index([(self.head + j) % self.k for j in range(self.k)])
            times[:, :self.k] = self.times[:, order]
            values[:, :self.k] = self.values[:, order]
        self.times, self.values, self.head, self.k = times, values, self.k, k

    def step(self, value, dt):
        """`value` [N, width] -> [N, width]."""
        xp = self.xp
        self.t = self.t + dt
        if self.delay_max <= 0:
            return value
        need = int(self.delay_max / dt) + 3
        if need > self.k:
            self._grow(need)
        self.times[:, self.head] = self.t
        self.values[:, self.head] = value
        self.head = (self.head + 1) % self.k
        due = self.times <= ((self.t - self.delay) + 1e-9)[:, None]
        key = xp.where(due, self.times, -xp.inf)
        j = key.argmax(1)
        found = due[self.rows, j]
        out = xp.where(found[:, None], self.values[self.rows, j], 0.0)
        return xp.where((self.delay <= 0)[:, None], value, out)


def _mn(xp, a, b):
    """Python's min(a, b): a unless b < a."""
    return xp.where(b < a, b, a)


def _mx(xp, a, b):
    """Python's max(a, b): a unless b > a."""
    return xp.where(b > a, b, a)


def _sum0(terms):
    """Python 3.10's sum(): from 0, left to right."""
    out = 0.0 + terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def _modulate(xp, factor, slip, enabled, release, reapply, release_rate, apply_rate, creep, dt):
    """brake.SlipModulator.step over N slots."""
    up = _mn(xp, 1.0, factor + apply_rate * dt)
    return xp.where(~enabled, up,
                    xp.where(slip > release, _mx(xp, 0.0, factor - release_rate * dt),
                             xp.where(slip < reapply, up, _mn(xp, 1.0, factor + creep * dt))))


class EvBatch:
    """`ev` (Ecu with the ev powertrain) over N slots; see the module's doc."""

    commands = "autoware"
    truth = ()
    array_api = ("numpy", "torch")
    per_slot_params = True

    def __init__(self, vehicle, params, seeds, xp=None):
        self.xp = xp = xp if xp is not None else xpmod.namespace("numpy")
        n = self.n = len(seeds)
        self.seeds = list(seeds)
        self.cals = cals = _calibrations(vehicle, params or {}, n)
        from .batch import BatchError
        for k, c in enumerate(cals):
            if not abs(c.max_steer_angle) <= SIN_COS_LIMIT:
                raise BatchError(f"ev: slot {k}: max_steer_angle {c.max_steer_angle} rad is past "
                                 f"{SIN_COS_LIMIT}, where xp.sin_cos (the batch's sine and cosine "
                                 "of the tire angle) stops being accurate; run the scalar form")
        c0 = cals[0]
        pt = EvPowertrain(c0)
        self.front_share = pt.front_share
        self.drive_share = pt.drive_share
        self.driven = pt.driven
        self.locked, self.lsd = c0.diff_type == "locked", c0.diff_type == "lsd"
        self.dynamic_split = c0.brake_distribution == "dynamic_axle_load"
        p = {}
        base = Calibration()
        for f in FIELDS:
            vals = [getattr(c, f) for c in cals]
            if isinstance(getattr(base, f), bool):   # a switch, however the override spells it
                p[f] = xp.asarray([bool(v) for v in vals], xp.bool)
            else:
                p[f] = xp.asarray([float(v) for v in vals], xp.float64)
        rows = [_derived(c) for c in cals]
        for k in rows[0]:
            p[k] = xp.asarray([r[k] for r in rows], xp.float64)
        self.p = p
        floors = [c.brake_floor for c in cals]
        self.brake_floor = floors[0] if len(set(floors)) == 1 else floors
        self.lags = _Lags(xp, cals)
        self.dead = _DeadTime(xp, n, p["brake_dead_time"], max(c.brake_dead_time for c in cals),
                              2)
        # divisors as arrays (_Lags says why)
        self.g = xp.full(n, G)
        s = self.front_share
        self.share_t = (xp.full(n, s), xp.full(n, 1 - s)) if 0.0 < s < 1.0 else None
        self.debug = {}
        self._fresh()

    # --- state ------------------------------------------------------------------------------

    def _fresh(self):
        xp, n = self.xp, self.n
        z = lambda: xp.zeros(n)                # noqa: E731
        f = lambda: xp.zeros(n, xp.bool)       # noqa: E731
        one = lambda: xp.ones(n)               # noqa: E731
        self.tire_angle = z()
        self.has_last, self.last_braking = f(), f()
        self.speed, self.started, self.sideslip = z(), f(), z()
        self.friction = xp.copy(self.p["tire_friction"])
        self.esc_brake = [z() for _ in range(4)]
        self.motor_factor = one()
        self.motor_torque = z()
        self.pressure = [z(), z()]
        self.abs_f = [one() for _ in range(4)]
        self.tc_f, self.rsc_f = one(), one()
        self.axle_mu, self.axle_known = z(), f()
        self.mu_measured, self.mu_known = z(), f()
        self.slip_event, self.regen_capped, self.regen_excess = f(), f(), f()
        self.last_spins = [z() for _ in range(1 if self.locked else sum(self.driven))]
        self.spins_known = f()
        self.last_slower, self.slower_known = z(), f()
        self.since_spin = xp.full(n, xp.inf)
        self.abs_event = f()
        self.wind = [z(), z()]

    def reset_batch(self, mask):
        """Every controller and actuator of the slots in `mask` at rest (Ecu.reset)."""
        xp = self.xp
        if xp.name == "numpy":
            import numpy as np
            m = np.asarray(mask, dtype=bool)
            if m.shape != (self.n,):
                from .batch import BatchError
                raise BatchError(f"reset_batch: a mask of shape {m.shape}, not ({self.n},)")
        else:
            m = xp.asarray(mask, xp.bool)
            if tuple(m.shape) != (self.n,):
                from .batch import BatchError
                raise BatchError(f"reset_batch: a mask of shape {tuple(m.shape)}, not "
                                 f"({self.n},)")
        keep = {k: getattr(self, k) for k in _STATE + _LISTS}
        self._fresh()
        for k in _STATE:
            setattr(self, k, xp.where(m, getattr(self, k), keep[k]))
        for k in _LISTS:
            setattr(self, k, [xp.where(m, a, b) for a, b in zip(getattr(self, k), keep[k])])
        self.dead.reset(m)

    # --- the parts --------------------------------------------------------------------------

    def _road_load(self, v):
        xp, p = self.xp, self.p
        roll = p["k_roll"] * _mn(xp, 1.0, xp.abs(v) / ROLLING_RAMP)
        drag = p["k_drag"] * v * v
        s = roll + drag
        return xp.where(v > 0, -s, xp.where(v < 0, s, 0.0))

    def _motor_limit(self, omega):
        xp, p = self.xp, self.p
        w = xp.abs(omega)
        pos = w > 0
        t = xp.where(pos, _mn(xp, p["motor_peak_torque"],
                              p["motor_peak_power"] / xp.where(pos, w, 1.0)),
                     p["motor_peak_torque"])
        t = xp.where(w > p["taper_from"], t * ((p["w_max"] - w) / p["taper_den"]), t)
        return xp.where(w >= p["w_max"], 0.0, t)

    def _regen_limit(self, omega, speed, motor_limit):
        xp, p = self.xp, self.p
        w = xp.abs(omega)
        pos = w > 0
        t = _mn(xp, p["regen_max_torque"], motor_limit)
        t = xp.where(pos, _mn(xp, t, p["regen_max_power"] / xp.where(pos, w, 1.0)), t)
        v = xp.abs(speed)
        lo, hi = p["regen_fade_low"], p["regen_fade_high"]
        t = xp.where(v < hi, t * ((v - lo) / p["fade_den"]), t)
        return xp.where(v <= lo, 0.0, t)

    def _clamp_front(self, front):
        xp = self.xp
        return _mn(xp, _mx(xp, front, 0.05), 0.95)

    def _brake_split(self, decel):
        xp, p = self.xp, self.p
        shift = p["com2"] * _mx(xp, decel, 0.0) / self.g if self.dynamic_split else 0.0
        return self._clamp_front((p["com0"] + shift) / p["wheel_base"])

    def _axle_load_share(self, accel):
        xp, p, fs = self.xp, self.p, self.front_share
        front = self._clamp_front((p["com0"] - p["com2"] * accel / self.g) / p["wheel_base"])
        if 0.0 < fs < 1.0:
            return xp.ones(self.n)
        return front if fs >= 1.0 else 1.0 - front

    def _bicycle(self, speed, tan):
        xp, p = self.xp, self.p
        v = _mx(xp, xp.abs(speed), 0.1)
        return speed * tan / (p["wheel_base"] * (1 + p["esc_understeer"] * v * v))

    def _driven_load(self, accel_x):
        xp, p = self.xp, self.p
        front = self._clamp_front((p["com0"] - p["com2"] * accel_x / self.g)
                                  / p["wheel_base"])
        share = _sum0([(front if i < 2 else 1 - front) / 2
                       for i, d in enumerate(self.driven) if d])
        return _mx(xp, share, 0.05)

    # --- the step ---------------------------------------------------------------------------

    def step_batch(self, commands, sensors, dt):
        if self.xp.name == "numpy":
            import numpy as np
            with np.errstate(all="ignore"):   # both sides of every branch are computed
                return self._step(commands, sensors, dt)
        return self._step(commands, sensors, dt)

    def _step(self, commands, sensors, dt):
        xp, p, n = self.xp, self.p, self.n
        dt = float(dt)
        lag = self.lag_now = self.lags(dt)
        dbg = self.debug = {}
        f64 = xp.float64

        # the request (Request() where there is none yet)
        valid_c = commands.get("valid")
        valid_c = xp.ones(n, xp.bool) if valid_c is None else xp.asarray(valid_c, xp.bool)
        acc = xp.where(valid_c, xp.asarray(commands["acceleration"], f64), 0.0)
        vel = xp.where(valid_c, xp.asarray(commands["velocity"], f64), 0.0)
        angle_req = xp.where(valid_c, xp.asarray(commands["tire_angle"], f64), 0.0)
        gear = xp.where(valid_c, xp.asarray(commands["gear"], xp.int64), GEAR_DRIVE)
        # the sensors (sv: a reading this step)
        sv = xp.asarray(sensors["valid"], xp.bool)
        ws = xp.asarray(sensors["wheel_speeds"], f64)
        w = [xp.where(sv, ws[:, i], 0.0) for i in range(4)]
        yaw = xp.where(sv, xp.asarray(sensors["yaw_rate"], f64), 0.0)
        ax = xp.where(sv, xp.asarray(sensors["accel_x"], f64), 0.0)
        ay = xp.where(sv, xp.asarray(sensors["accel_y"], f64), 0.0)

        # 1. the power steering (actuators.steer)
        limit = p["max_steer_angle"]
        cur = self.tire_angle
        target = _mx(xp, -limit, _mn(xp, limit, angle_req))
        target = xp.where(p["steer_time_constant"] > 0, cur + (target - cur) * lag["steer"],
                          target)
        step = p["steer_rate_limit"] * dt
        target = xp.where(p["steer_rate_limit"] > 0,
                          _mx(xp, cur - step, _mn(xp, cur + step, target)), target)
        d = self.tire_angle = target
        sin_d, cos_d = xpmod.sin_cos(xp, d)
        tan_d = sin_d / cos_d

        # 2. what the brake module was doing last step (Ecu._activity)
        abs_active = ((self.abs_f[0] < 1.0) | (self.abs_f[1] < 1.0) | (self.abs_f[2] < 1.0)
                      | (self.abs_f[3] < 1.0))
        hl = self.has_last
        a_braking = hl & (self.last_braking | (self.motor_torque < 0))
        a_driving = hl & (self.motor_torque > 0)
        a_abs = hl & abs_active
        a_tc = hl & (self.tc_f < 1.0)

        # 3. the estimates (estimator.Estimator.step)
        r = p["wheel_radius"]
        road = [x * r for x in w]
        beta = self.sideslip
        half = p["half"]
        place = []
        for i, y in enumerate((half, -half, half, -half)):
            if i < 2:
                place.append((cos_d + beta * sin_d, yaw * (p["wheel_base"] * sin_d - y * cos_d)))
            else:
                place.append((xp.ones(n), -yaw * y))
        body = [(u - b) / k for u, (k, b) in zip(road, place)]
        mean = _sum0(body) / 4
        # _speed
        standing = ((xp.abs(body[0]) < 0.05) & (xp.abs(body[1]) < 0.05)
                    & (xp.abs(body[2]) < 0.05) & (xp.abs(body[3]) < 0.05)
                    & (xp.abs(self.speed) < 0.1))
        direction_e = xp.where(xp.where(self.speed != 0, self.speed, mean) >= 0, 1.0, -1.0)
        best = direction_e * body[0]
        worst = best
        for u in body[1:]:
            best = _mx(xp, best, direction_e * u)
            worst = _mn(xp, worst, direction_e * u)
        free = [u for u, dr in zip(body, self.driven) if not dr]
        drive_candidate = _sum0(free) / len(free) if free else direction_e * worst
        candidate = xp.where(a_braking, direction_e * best,
                             xp.where(a_driving, drive_candidate, mean))
        predicted = self.speed + ax * dt
        slipping = a_abs | a_tc
        below = direction_e * (candidate - predicted) < 0
        k = xp.where(slipping, lag["slipping"], lag["gain"])
        k = xp.where(slipping & a_braking & below, lag["bound"], k)
        k = xp.where(slipping & a_braking & ~below, lag["gain"], k)
        moving = predicted + (candidate - predicted) * k
        started = self.started
        speed_new = xp.where(started, xp.where(standing, mean, moving), mean)
        v = self.speed = xp.where(sv, speed_new, self.speed)
        self.started = started | sv
        dbg.update(est_first=sv & ~started, est_standing=sv & started & standing,
                   est_braking=sv & started & ~standing & a_braking,
                   est_driving=sv & started & ~standing & ~a_braking & a_driving,
                   est_mean=sv & started & ~standing & ~a_braking & ~a_driving,
                   est_slipping=sv & started & ~standing & slipping & ~a_braking,
                   est_bound=sv & started & ~standing & slipping & a_braking & below,
                   est_rising=sv & started & ~standing & slipping & a_braking & ~below)
        slip = []
        for u, (kk, b) in zip(road, place):
            ground = kk * v + b
            slip.append((u - ground) / (xp.abs(ground) + p["est_slip_offset"]))
        # _sideslip
        fast = xp.abs(v) >= SIDESLIP_MIN_SPEED
        rate = ay / xp.where(fast, v, 1.0) - yaw
        tau = p["est_sideslip_washout"]
        leak = xp.where(tau > 0, self.sideslip * dt / xp.where(tau > 0, tau, 1.0), 0.0)
        ss = xp.where(fast, self.sideslip + (rate * dt - leak), 0.0)
        self.sideslip = ss = xp.where(sv, ss, self.sideslip)
        # _friction
        reached = xpmod.hypot(xp, ax, ay) / self.g
        want = self._bicycle(v, tan_d)
        band = p["esc_dead_band"] + p["esc_dead_band_share"] * xp.abs(want)
        understeer = ((xp.abs(v) >= p["esc_min_speed"]) & (xp.abs(want) > band)
                      & (want * yaw >= 0) & (xp.abs(yaw) < xp.abs(want) - band))
        at_limit = (a_abs | a_tc | (xp.abs(ss) > p["half_sideslip_limit"]) | understeer)
        tc_only = a_tc & ~a_abs
        push_x = xp.where(v >= 0, ax, -ax)
        held = at_limit & tc_only & (push_x <= 0.0)
        reached_l = xp.where(tc_only, push_x / self.g / self._driven_load(ax), reached)
        target_mu = _mn(xp, _mx(xp, reached_l + p["est_mu_margin"], p["est_mu_floor"]),
                        p["est_mu_max"])
        fr = self.friction
        at = fr + (target_mu - fr) * lag["mu"]
        away = _mn(xp, p["tire_friction"], fr + p["est_mu_recover_rate"] * dt)
        away = _mx(xp, away, _mn(xp, reached, p["tire_friction"]))
        fr_new = xp.where(at_limit, xp.where(held, fr, at), away)
        mu = self.friction = xp.where(sv, fr_new, fr)
        dbg.update(mu_limit=sv & at_limit & ~held, mu_held=sv & held, mu_recover=sv & ~at_limit,
                   mu_understeer=sv & understeer)

        # 4. the stability control (esc.Esc.step)
        esc_on = p["esc_enabled"] & (xp.abs(v) >= p["esc_min_speed"])
        vv = _mx(xp, xp.abs(v), 0.1)
        cap = p["esc_target_margin"] * mu * G / vv
        r_ref = _mx(xp, -cap, _mn(xp, cap, self._bicycle(v, tan_d)))
        err = yaw - r_ref
        band = p["esc_dead_band"] + p["esc_dead_band_share"] * xp.abs(r_ref)
        out_band = xp.abs(err) > band
        moment = xp.where(out_band, p["neg_esc_gain"] * (err - xp.where(err < 0, -band, band)),
                          0.0)
        oversteer = out_band & (yaw * err > 0) & (xp.abs(yaw) > xp.abs(r_ref))
        lim = p["esc_sideslip_limit"]
        wide = xp.abs(ss) > lim
        moment = xp.where(wide, moment + p["esc_sideslip_gain"] * (ss - xp.where(ss < 0, -lim,
                                                                                  lim)), moment)
        oversteer = oversteer | wide
        acting = esc_on & (moment != 0.0)
        oversteer = acting & oversteer
        left = moment > 0
        wheel = xp.where(left, xp.where(oversteer, 0, 2), xp.where(oversteer, 1, 3))
        force = xp.abs(moment) / half
        torque = xp.where(acting, _mn(xp, force * r, p["esc_max_brake_torque"]), 0.0)
        for i in range(4):
            tgt = xp.where(acting & (wheel == i), torque, 0.0)
            b = self.esc_brake[i] + (tgt - self.esc_brake[i]) * lag["esc"]
            self.esc_brake[i] = xp.where((b < 1.0) & (tgt == 0.0), 0.0, b)
        mf = self.motor_factor
        self.motor_factor = xp.where(oversteer, _mx(xp, 0.0, mf - p["esc_motor_cut_rate"] * dt),
                                     _mn(xp, 1.0, mf + p["esc_motor_restore_rate"] * dt))
        dbg.update(esc_oversteer=acting & oversteer, esc_understeer=acting & ~oversteer,
                   esc_sideslip=esc_on & wide, esc_cut=self.motor_factor < 1.0)

        # 5. the powertrain (powertrain.EvPowertrain.torques)
        ratio = p["gear_ratio"]
        direction = xp.where(gear == GEAR_REVERSE, -1.0, 1.0)
        spins_w = [x for x, dr in zip(w, self.driven) if dr]
        omega = xp.where(sv, _sum0(spins_w) / len(spins_w), v / r) * ratio
        park, neutral = gear == GEAR_PARK, gear == GEAR_NEUTRAL
        hold = park | ((xp.abs(vel) < 0.01) & (acc <= 0.01) & (xp.abs(v) < 0.1))
        motor_prev = self.motor_torque
        self._regen_slip(direction, v, w, slip, ax, sv, hold, dt, lag)
        capped_now = xp.zeros(n, xp.bool)
        idle = hold | neutral
        self.abs_event = xp.where(idle | (xp.abs(v) < p["abs_min_speed"]), False,
                                  self.abs_event)
        push = (p["mass_ff"] * acc - self._road_load(direction * v)
                + p["damp4"] * (direction * v) / p["r2"])
        pushing = ~idle & (push >= 0)
        braking = ~idle & ~pushing
        self.abs_event = xp.where(pushing, False, self.abs_event)
        motor_limit = self._motor_limit(omega)
        request_push = _mn(xp, push * r / ratio, motor_limit)
        brake_force = -push
        self.abs_event = xp.where(braking & abs_active, True, self.abs_event)
        available = xp.where(self.abs_event, 0.0, self._regen_limit(omega, v, motor_limit))
        regen_force = _mn(xp, brake_force, available * ratio / r)
        decel = _mn(xp, -acc, p["mu_g"])
        own = self._regen_share(brake_force, decel)
        excess = braking & (regen_force > own)
        cap = (p["grip_share"] * self.axle_mu * self._axle_load_share(ax) * p["mass"] * G)
        top = _mx(xp, cap, own)
        capping = braking & self.axle_known & sv & (regen_force > top)
        regen_force = xp.where(capping, top, regen_force)
        capped_now = capping
        cut = (regen_force > own) & (self.rsc_f < 1.0)
        regen_force = xp.where(cut, own + (regen_force - own) * self.rsc_f, regen_force)
        request_brake = -regen_force * r / ratio
        split_f, split_r = self._split_friction(brake_force, regen_force, decel)
        request = xp.where(idle, 0.0, xp.where(pushing, request_push, request_brake))
        hold_t = p["hold_brake_torque"]
        fric_f = xp.where(hold, hold_t, xp.where(braking, split_f, 0.0))
        fric_r = xp.where(hold, hold_t, xp.where(braking, split_r, 0.0))
        self.regen_excess, self.regen_capped = excess, capped_now
        dbg.update(hold=hold, neutral=neutral & ~hold, reverse=direction < 0, park=park,
                   pushing=pushing, braking=braking,
                   regen_blend=braking & (regen_force > 0) & ((split_f > 0) | (split_r > 0)),
                   regen_capped=capping, regen_cut=braking & cut,
                   abs_event=braking & self.abs_event)
        # the motor
        mt = xp.where(p["motor_time_constant"] > 0,
                      motor_prev + (request - motor_prev) * lag["motor"], request)
        mt = self.motor_torque = _mx(xp, -motor_limit, _mn(xp, motor_limit, mt))
        delivered = xp.where(mt > 0, mt * self.motor_factor, mt)
        # traction control
        tc_on = sv & p["tc_enabled"] & (delivered > 0)
        spin = None
        for x, dr in zip(slip, self.driven):
            if dr:
                s_ = direction * x
                spin = s_ if spin is None else _mx(xp, spin, s_)
        self.tc_f = _modulate(xp, self.tc_f, xp.where(tc_on, spin, 0.0), tc_on,
                              p["tc_slip_limit"], p["tc_slip_reapply"], p["tc_release_rate"],
                              p["tc_apply_rate"], p["tc_creep"], dt)
        dbg.update(tc_cut=tc_on & (spin > p["tc_slip_limit"]), tc_active=self.tc_f < 1.0)
        delivered = xp.where(tc_on, delivered * self.tc_f, delivered)
        axle = direction * delivered * ratio
        drive = [axle * s for s in self.drive_share]
        for k, (a, b) in enumerate(((0, 1), (2, 3))):
            if self.driven[a]:
                t = self._differential(drive[a] + drive[b], direction, w[a], w[b], dt, k, sv)
                drive[a] = xp.where(sv, drive[a] - t, drive[a])
                drive[b] = xp.where(sv, drive[b] + t, drive[b])
        # the friction brakes: dead time, then the pressure lag, per axle
        fric = xp.zeros((n, 2))
        fric[:, 0], fric[:, 1] = fric_f, fric_r
        delayed = self.dead.step(fric, dt)
        floor = p["brake_floor"]
        for i in range(2):
            dl = delayed[:, i]
            pr = self.pressure[i]
            pr = xp.where(p["brake_time_constant"] > 0, pr + (dl - pr) * lag["brake"], dl)
            zero = (pr < floor) & (dl < floor)
            dbg[f"brake_floor_{i}"] = zero & (pr > 0)
            self.pressure[i] = xp.where(zero, 0.0, pr)
        # ABS per wheel
        abs_on = p["abs_enabled"] & sv & ~hold & (xp.abs(v) > p["abs_min_speed"])
        brake = []
        last_braking = xp.zeros(n, xp.bool)
        for i in range(4):
            pressure = self.pressure[0 if i < 2 else 1] + self.esc_brake[i]
            lock = xp.where(sv, -direction * slip[i], 0.0)
            on = pressure > 1.0
            if self.driven[i]:
                on = on | (mt < 0)
            en = abs_on & on
            dbg[f"abs_release_{i}"] = en & (lock > p["abs_slip_release"])
            dbg[f"abs_reapply_{i}"] = en & (lock < p["abs_slip_reapply"]) & (self.abs_f[i] < 1.0)
            self.abs_f[i] = _modulate(xp, self.abs_f[i], lock, en, p["abs_slip_release"],
                                      p["abs_slip_reapply"], p["abs_release_rate"],
                                      p["abs_apply_rate"], p["abs_creep"], dt)
            b = _mn(xp, pressure * self.abs_f[i], p["max_brake_torque"])
            brake.append(b)
            last_braking = last_braking | (b > floor)
        self.last_braking = last_braking
        self.has_last = xp.ones(n, xp.bool)
        out_d, out_b = xp.zeros((n, 4)), xp.zeros((n, 4))
        for i in range(4):
            out_d[:, i], out_b[:, i] = drive[i], brake[i]
        return {"drive": out_d, "brake": out_b, "tire_angle": xp.copy(d)}

    def _regen_share(self, brake, decel):
        xp, s = self.xp, self.front_share
        share = self._brake_split(decel)
        if s >= 1.0:
            return brake * share
        if s <= 0.0:
            return brake * (1 - share)
        return _mn(xp, brake * share / self.share_t[0], brake * (1 - share) / self.share_t[1])

    def _split_friction(self, brake, regen, decel):
        xp, r = self.xp, self.p["wheel_radius"]
        share = self._brake_split(decel)
        front, rear = brake * share, brake * (1 - share)
        rf, rr = regen * self.front_share, regen * (1 - self.front_share)
        excess = _mx(xp, 0.0, rf - front) + _mx(xp, 0.0, rr - rear)
        front = _mx(xp, 0.0, front - rf)
        rear = _mx(xp, 0.0, rear - rr)
        cut = _mn(xp, front, excess)
        front, excess = front - cut, excess - cut
        rear = rear - _mn(xp, rear, excess)
        return front * r / 2, rear * r / 2

    def _differential(self, axle_torque, direction, spin_left, spin_right, dt, axle, sv):
        xp, p = self.xp, self.p
        if not (self.locked or self.lsd):
            return xp.zeros(self.n)
        inertia = p["inertia"]
        gap = spin_left - spin_right
        damping = inertia * gap / self.lag_now["dt2"]
        grip = p["grip"]
        want = damping + self.wind[axle]
        wind = _mx(xp, -grip, _mn(xp, grip, self.wind[axle] + inertia * gap / self.lag_now["dt4"]))
        if self.lsd:
            lock = xp.where(axle_torque * direction >= 0, p["diff_power_lock"],
                            p["diff_coast_lock"])
            limit = p["diff_preload"] + lock * xp.abs(axle_torque)
            self.debug[f"diff_slipping_{axle}"] = sv & (xp.abs(want) > limit)
            want = _mx(xp, -limit, _mn(xp, limit, want))
            wind = _mx(xp, -limit, _mn(xp, limit, wind))
        self.wind[axle] = xp.where(sv, wind, self.wind[axle])
        self.debug[f"diff_transfer_{axle}"] = sv & (want != 0)
        return want

    def _regen_slip(self, direction, speed, w, slip, accel, sv, hold, dt, lag):
        """EvPowertrain.regen_slip over N slots (sv: wheels and accel known)."""
        xp, p = self.xp, self.p
        dbg = self.debug
        r = p["wheel_radius"]
        on = p["regen_slip_control"] & sv & ~hold & (xp.abs(speed) > p["abs_min_speed"])
        regen = (self.motor_torque < 0) & self.regen_excess
        slips = [-direction * x for x, dr in zip(slip, self.driven) if dr]
        spins = [direction * x * r for x, dr in zip(w, self.driven) if dr]
        if self.locked:
            lock = _sum0(slips) / len(slips)
            spins = [_sum0(spins) / len(spins)]
        else:
            lock = slips[0]
            for s_ in slips[1:]:
                lock = _mx(xp, lock, s_)
        lock = xp.where(sv, lock, 0.0)
        fs = self.front_share
        pr = self.pressure
        if 0.0 < fs < 1.0:
            undriven = 0.0
            own = pr[0] + pr[1]
        else:
            undriven = 2 * pr[1 if fs >= 1.0 else 0] / r
            own = pr[0 if fs >= 1.0 else 1]
        carried = self._road_load(direction * speed) - p["mass"] * direction * accel - undriven
        applied = (-self.motor_torque * p["gear_ratio"] + 2 * own) / r
        self.since_spin = xp.where(lock < p["neg_reapply"], 0.0, self.since_spin + dt)
        settled = self.since_spin >= p["regen_spin_settle"]
        short = ~sv | (settled & (carried < p["deficit_keep"] * applied))
        have = sv & self.spins_known
        rates = [(u - u0) / lag["dt"] for u, u0 in zip(spins, self.last_spins)]
        rate = _sum0(rates) / len(rates)
        least = rates[0]
        for x in rates[1:]:
            least = _mn(xp, least, x)
        slower = direction * accel - least
        runaway = have & short & (lock > p["regen_runaway_slip"]) & (slower > p["regen_runaway_decel"])
        # not while the excess falls (a wheel settling from a wheelspin; scensim issue 106)
        falling = self.slower_known & (slower < self.last_slower)
        growing = have & settled & (slower > p["regen_slip_growth"]) & ~falling
        self.last_slower = xp.where(have, slower, 0.0)
        self.slower_known = have
        self.last_spins = [xp.where(sv, u, u0) for u, u0 in zip(spins, self.last_spins)]
        self.spins_known = sv
        rsc_active = self.rsc_f < 1.0
        slipped = (lock > p["regen_slip_release"]) & (short | growing | rsc_active)
        limit = runaway | slipped
        rsc_in = xp.where(limit, _mx(xp, lock, p["rsc_release_eps"]),
                          _mn(xp, lock, p["regen_slip_release"]))
        en = on & regen
        self.rsc_f = _modulate(xp, self.rsc_f, rsc_in, en, p["regen_slip_release"],
                               p["regen_slip_reapply"], p["regen_release_rate"],
                               p["regen_apply_rate"], p["rsc_creep"], dt)
        cont = on   # on implies a reading (accel known)
        self.slip_event = self.slip_event & cont
        weight = self._axle_load_share(accel) * p["mass"] * G
        start = cont & regen & limit & ~self.slip_event
        self.slip_event = self.slip_event | start
        self.mu_known = self.mu_known & ~start
        in_event = cont & self.slip_event
        measure = in_event & (self.motor_torque < 0) & have
        spin = _sum0(spins) / len(spins)
        nd = sum(self.driven)
        force = applied + nd * (p["damp"] * spin + p["inertia"] * rate) / p["r2"]
        mu = _mx(xp, force, 0.0) / weight
        first = measure & ~self.mu_known
        later = measure & self.mu_known
        mu_first = _mx(xp, mu, p["est_mu_floor"])
        mm = self.mu_measured + (mu - self.mu_measured) * lag["axle_mu"]
        self.mu_measured = xp.where(first, mu_first, xp.where(later, mm, self.mu_measured))
        self.axle_mu = xp.where(first, mu_first,
                                xp.where(later, _mx(xp, self.axle_mu, self.mu_measured),
                                         self.axle_mu))
        self.axle_known = self.axle_known | first
        self.mu_known = self.mu_known | first
        still = in_event & (self.rsc_f < 1.0)
        self.slip_event = self.slip_event & ~(in_event & ~still)
        post = cont & ~still & self.axle_known
        am = xp.where(post & ~self.regen_capped,
                      self.axle_mu + p["regen_mu_recover_rate"] * dt, self.axle_mu)
        raise_ = post & regen & (lock < p["regen_slip_reapply"])
        raised = raise_ & (applied / weight > am)
        am = xp.where(raise_, _mx(xp, am, applied / weight), am)
        forget = post & (am >= p["tire_friction"])
        self.axle_mu = am
        self.axle_known = self.axle_known & ~forget
        dbg.update(rsc_release=en & (rsc_in > p["regen_slip_release"]), rsc_active=self.rsc_f < 1.0,
                   rsc_runaway=en & runaway, rsc_growing=en & growing & slipped,
                   slip_event_start=start, axle_mu_found=first, axle_mu_followed=later,
                   axle_mu_raise=raise_, axle_mu_raised=raised,
                   axle_mu_forgotten=post & forget, axle_mu_known=self.axle_known)


# the attributes reset_batch takes from a fresh state where its mask is set
_STATE = ("tire_angle", "has_last", "last_braking", "speed", "started", "sideslip", "friction",
          "motor_factor", "motor_torque", "tc_f", "rsc_f", "axle_mu", "axle_known",
          "mu_measured", "mu_known", "slip_event", "regen_capped", "regen_excess",
          "spins_known", "last_slower", "slower_known", "since_spin", "abs_event")
# and its per-wheel and per-axle lists
_LISTS = ("esc_brake", "pressure", "abs_f", "last_spins", "wind")


def _calibrations(vehicle, params, n):
    """One Calibration per slot from `params` (values: one for all, or arrays [N] per slot), each
    as `ev`'s scalar component makes it; refused when its structure differs between slots."""
    from .batch import BatchError

    def at(v, i):
        if hasattr(v, "shape") and len(getattr(v, "shape", ())) == 1 and v.shape[0] == n:
            x = v[i]
            return x.item() if hasattr(x, "item") else x
        return v

    cache, out = {}, []
    for i in range(n):
        over = {k: at(v, i) for k, v in params.items()}
        over["powertrain"] = "ev"
        key = repr(sorted(over.items()))
        c = cache.get(key)
        if c is None:
            c = cache[key] = Calibration.from_spec(vehicle, over)
            EvPowertrain(c)   # its checks, as the scalar's
        out.append(c)
    for f in STRUCTURAL:
        vals = {getattr(c, f) for c in out}
        if len(vals) > 1:
            raise BatchError(f"ev: calibration field {f!r} differs between slots "
                             f"({sorted(map(str, vals))}); it shapes the powertrain, one value "
                             "for every slot")
    return out


__all__ = ["EvBatch"]
