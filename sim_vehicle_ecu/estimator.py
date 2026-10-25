"""What the brake module estimates from its own sensors: the car's speed, each wheel's slip,
the sideslip and the road's friction. Pure.

A brake module has no speed-over-ground sensor, no slip sensor and no friction sensor; it has
four wheel speeds, a yaw rate and two accelerations, and works the rest out (van Zanten,
"Bosch ESP Systems: 5 Years of Experience", SAE 2000-01-1633; Rajamani, *Vehicle Dynamics
and Control*, ch. 8 and 14). This one does it the textbook way:

- **Reference speed**: predicted from the longitudinal acceleration, then pulled toward what
  the wheels say: all four when the car is rolling freely, the undriven ones under drive (a
  spinning driven wheel says too much), the fastest under braking (a braked wheel says too
  little). The pull is `est_speed_gain` (1/s), and `est_speed_gain_slipping` while ABS or
  traction control works a wheel, when none of the wheels can be trusted; under ABS the
  fastest wheel is a bound from below, so the estimate rises to it at once and sinks toward
  it only at `est_speed_gain_bound`. The first reading after a reset is the wheels' mean.
- **Slip**: each wheel's road speed against the reference, over `|v| + est_slip_offset`:
  PhysX's own definition (its tire adds its minimum slip denominator, 4 m/s on the rig, to
  the speed), the one the ABS and TC thresholds were tuned against when they read PhysX's
  slip. A `max(|v|, 4)` read up to twice that at town speeds (scenario-world-bridge issue
  009).
- **Sideslip**: the kinematic integral `beta' = a_y / v - r` (exact for small angles, the
  acceleration being the body's in its own frame), with a washout so a sensor's bias does
  not build up: `est_sideslip_washout` s.
- **Friction**: a car shows its road's friction only at the limit. When it is there (ABS or
  traction control working a wheel, a sideslip past half the ESC's limit, or turning less than
  the steering asks), the estimate follows the acceleration it reaches (plus
  `est_mu_margin`; under traction control alone, over the driven axle's share of the load);
  away from the limit it recovers toward the design friction at `est_mu_recover_rate` /s, and
  never below what the car is reaching.
"""
import math
from typing import NamedTuple

from .calibration import G
from .esc import bicycle_yaw_rate
from .powertrain import Wheel

# Below this speed the sideslip means nothing and is held at 0, m/s.
SIDESLIP_MIN_SPEED = 1.0
# The friction estimate's smoothing at the limit, s: the accelerations are a step's.
MU_TIME_CONSTANT = 0.1


class Estimate(NamedTuple):
    speed: float  # the reference speed, m/s along x
    wheels: tuple  # four powertrain.Wheel: the speed sensor's spin, the estimated slip
    sideslip: float  # rad, + the velocity to the left of the nose
    friction: float  # the road's friction as the car has found it


class Activity(NamedTuple):
    """What the brake module itself was doing last step: how far its wheels can be trusted."""

    braking: bool = False  # any friction brake or regen on
    driving: bool = False  # the motor pushing
    abs_active: bool = False
    tc_active: bool = False


class Estimator:
    def __init__(self, calibration, driven):
        """`driven`: per wheel, whether the motor drives it."""
        self.c = calibration
        self.driven = tuple(driven)
        self.reset()

    def reset(self):
        self.speed = 0.0
        self.started = False
        self.sideslip = 0.0
        self.friction = self.c.tire_friction
        self.at_limit = False
        self.out = Estimate(0.0, None, 0.0, self.friction)

    def step(self, sensors, dt, tire_angle, activity):
        """This step's `Estimate` from `sensors` (None: no reading yet, as before the first
        physics step: the speed held, no wheels)."""
        c = self.c
        if sensors is None:
            self.out = Estimate(self.speed, None, self.sideslip, self.friction)
            return self.out
        r = c.wheel_radius
        road = [w * r for w in sensors.wheel_speeds]
        if self.started:
            self.speed = self._speed(road, sensors.accel_x, dt, activity)
        else:  # the first reading: the wheels, as a module waking up takes them
            self.speed = sum(road) / len(road)
            self.started = True
        v = self.speed
        denominator = abs(v) + c.est_slip_offset
        wheels = tuple(Wheel(w, (u - v) / denominator)
                       for w, u in zip(sensors.wheel_speeds, road))
        self._sideslip(v, sensors, dt)
        self._friction(v, sensors, dt, tire_angle, activity)
        self.out = Estimate(v, wheels, self.sideslip, self.friction)
        return self.out

    def _speed(self, road, accel_x, dt, activity):
        c = self.c
        mean = sum(road) / len(road)
        if all(abs(u) < 0.05 for u in road) and abs(self.speed) < 0.1:
            return mean  # standing: the wheels are the truth, and the integral's bias goes
        direction = 1.0 if (self.speed or mean) >= 0 else -1.0
        if activity.braking:
            candidate = direction * max(direction * u for u in road)
        elif activity.driving:
            free = [u for u, d in zip(road, self.driven) if not d]
            candidate = (sum(free) / len(free) if free
                         else direction * min(direction * u for u in road))
        else:
            candidate = mean
        predicted = self.speed + accel_x * dt
        slipping = activity.abs_active or activity.tc_active
        k = c.est_speed_gain_slipping if slipping else c.est_speed_gain
        if slipping and activity.braking and direction * (candidate - predicted) < 0:
            # under ABS even the fastest wheel is slower than the car: a bound from below,
            # followed down only as far as the acceleration's bias needs
            k = c.est_speed_gain_bound
        elif slipping and activity.braking:
            k = c.est_speed_gain  # a wheel back up to the road: the car is at least that fast
        return predicted + (candidate - predicted) * (1 - math.exp(-k * dt))

    def _sideslip(self, v, sensors, dt):
        if abs(v) < SIDESLIP_MIN_SPEED:
            self.sideslip = 0.0
            return
        rate = sensors.accel_y / v - sensors.yaw_rate
        tau = self.c.est_sideslip_washout
        self.sideslip += rate * dt - (self.sideslip * dt / tau if tau > 0 else 0.0)

    def _friction(self, v, sensors, dt, tire_angle, activity):
        c = self.c
        reached = math.hypot(sensors.accel_x, sensors.accel_y) / G
        self.at_limit = (activity.abs_active or activity.tc_active
                         or abs(self.sideslip) > 0.5 * c.esc_sideslip_limit
                         or self._understeer(v, sensors.yaw_rate, tire_angle))
        if self.at_limit:
            if activity.tc_active and not activity.abs_active:
                # only the driven axle is at its limit: its share of the load carries `reached`
                reached = abs(sensors.accel_x) / G / self._driven_load(sensors.accel_x)
            target = max(reached + c.est_mu_margin, c.est_mu_floor)
            self.friction += (target - self.friction) * (1 - math.exp(-dt / MU_TIME_CONSTANT))
        else:
            self.friction = min(c.tire_friction, self.friction + c.est_mu_recover_rate * dt)
            self.friction = max(self.friction, min(reached, c.tire_friction))

    def _understeer(self, v, yaw_rate, tire_angle):
        """Turning clearly less than the steering asks, the same way: the front tires at their
        limit."""
        c = self.c
        if abs(v) < c.esc_min_speed:
            return False
        want = bicycle_yaw_rate(c, v, tire_angle)
        band = c.esc_dead_band + c.esc_dead_band_share * abs(want)
        return abs(want) > band and want * yaw_rate >= 0 and abs(yaw_rate) < abs(want) - band

    def _driven_load(self, accel_x):
        """The driven wheels' share of the car's weight at `accel_x`: the static split by the
        centre of mass, shifted toward the rear under acceleration."""
        c = self.c
        front = (c.com[0] - c.com[2] * accel_x / G) / c.wheel_base
        front = min(max(front, 0.05), 0.95)
        share = sum((front if i < 2 else 1 - front) / 2 for i, d in enumerate(self.driven) if d)
        return max(share, 0.05)
