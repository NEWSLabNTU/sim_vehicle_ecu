"""The actuators the onboard computers drive, as a simulator with a drive-none vehicle lacks
them: the motor's envelope and torque lag, the brake hydraulics' dead time and pressure lag,
the steering motor's lag and rate limit. Pure."""
import math
from collections import deque

RPM = 2 * math.pi / 60  # rad/s per rpm
# A releasing brake below this is released: PhysX keeps a slow wheel in its sticky-tire hold
# while any brake torque at all is applied, so a pressure decaying exponentially towards 0
# held the car for seconds (seen in Isaac: 6 s at full drive torque with 1e-10 N m left).
# 50 N m is ~0.03 m/s^2 of braking per wheel; with the lags, the hold releases in ~0.3 s.
BRAKE_FLOOR = 50.0  # N m


def lag(current, target, dt, time_constant):
    """One step of a first-order lag (exact for a target held over `dt`)."""
    if time_constant <= 0:
        return target
    return current + (target - current) * (1 - math.exp(-dt / time_constant))


class DeadTime:
    """A pure delay: `step(value, dt)` returns what went in `delay` s earlier (`initial`
    before that), at the steps' own times."""

    def __init__(self, delay, initial=0.0):
        self.delay = delay
        self.initial = initial
        self.reset()

    def reset(self):
        self.t = 0.0
        self.queue = deque()
        self.out = self.initial

    def step(self, value, dt):
        self.t += dt
        if self.delay <= 0:
            self.out = value
            return value
        self.queue.append((self.t, value))
        while self.queue and self.queue[0][0] <= self.t - self.delay + 1e-9:
            self.out = self.queue.popleft()[1]
        return self.out


def motor_limit(c, omega):
    """The motor's torque envelope at `omega` (rad/s at the motor shaft, either sign), N m:
    peak torque up to base speed (peak power / peak torque), then peak power / omega, tapering
    linearly to 0 over the last 5 % below the maximum speed, 0 past it."""
    w = abs(omega)
    w_max = c.motor_max_rpm * RPM
    if w >= w_max:
        return 0.0
    t = c.motor_peak_torque
    if w > 0:
        t = min(t, c.motor_peak_power / w)
    taper_from = 0.95 * w_max
    if w > taper_from:
        t *= (w_max - w) / (w_max - taper_from)
    return t


def steer(current, request, dt, limit, time_constant=0.0, rate_limit=0.0):
    """The steering actuator's tire angle after `dt`: `request` clamped to +-limit, through
    the lag and rate limit when set."""
    target = max(-limit, min(limit, request))
    if time_constant > 0:
        target = current + (target - current) * (1 - math.exp(-dt / time_constant))
    if rate_limit > 0:
        step = rate_limit * dt
        target = max(current - step, min(current + step, target))
    return target
