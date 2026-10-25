"""Electronic stability control: one wheel's brake against a yaw rate the tires cannot hold.
Part of the brake module. Pure.

The usual design (Rajamani, *Vehicle Dynamics and Control*, ch. 8; van Zanten's Bosch ESP):
a target yaw rate from the bicycle model at the present speed and tire angle, capped by what
the friction can carry (|r| <= mu g / v), against the measured one. Past a dead band:

- oversteer (turning more than the target, same sense): a yaw moment against the turn, from
  the outer front wheel's brake;
- understeer (turning less): a yaw moment into the turn, from the inner rear wheel's brake.

A sideslip past `esc_sideslip_limit` adds to the oversteer moment whatever the yaw rate says
(a car sliding wide at a modest yaw rate). While it works against oversteer the motor's torque
is cut, as an ESC asks the drive unit to. The moment becomes a brake torque on the one wheel
(F = M / (tread / 2), at the wheel radius), through a first-order lag for the hydraulics; the
powertrain adds it to the wheel's brake ahead of ABS, so ABS still keeps that wheel from
locking.

The speed, the sideslip and the friction are the brake module's estimates
(`sim_vehicle_ecu.estimator`); the yaw rate is its sensor's.
"""
import math
from typing import NamedTuple

from .calibration import G
from .signals import FRONT_LEFT, FRONT_RIGHT, REAR_LEFT, REAR_RIGHT


class EscOut(NamedTuple):
    brake: tuple  # extra brake torque per wheel, N m
    motor_factor: float  # on the motor's delivered torque, 0..1
    moment: float  # the yaw moment asked, N m (+: left)
    target: float  # the target yaw rate, rad/s


IDLE = EscOut((0.0,) * 4, 1.0, 0.0, 0.0)


def bicycle_yaw_rate(c, speed, tire_angle):
    """The bicycle model's yaw rate at `speed` and `tire_angle`, rad/s, with the target's
    understeer, uncapped."""
    v = max(abs(speed), 0.1)
    return speed * math.tan(tire_angle) / (c.wheel_base * (1 + c.esc_understeer * v * v))


class Esc:
    def __init__(self, calibration):
        self.c = calibration
        self.reset()

    def reset(self):
        self.brake = [0.0] * 4
        self.motor_factor = 1.0
        self.out = IDLE

    def target(self, speed, tire_angle, mu):
        """The bicycle model's yaw rate, capped at esc_target_margin * mu g / speed."""
        v = max(abs(speed), 0.1)
        cap = self.c.esc_target_margin * mu * G / v
        return max(-cap, min(cap, bicycle_yaw_rate(self.c, speed, tire_angle)))

    def step(self, speed, yaw_rate, sideslip, tire_angle, mu, dt):
        c = self.c
        if not c.esc_enabled or abs(speed) < c.esc_min_speed:
            return self._settle(0.0, None, 0.0, dt, False)
        r_ref = self.target(speed, tire_angle, mu)
        err = yaw_rate - r_ref
        band = c.esc_dead_band + c.esc_dead_band_share * abs(r_ref)
        moment = 0.0
        oversteer = False
        if abs(err) > band:
            excess = err - math.copysign(band, err)
            moment = -c.esc_gain * excess  # against the error
            # the same sense as the yaw rate and more of it: oversteer
            oversteer = yaw_rate * err > 0 and abs(yaw_rate) > abs(r_ref)
        beta = sideslip
        if abs(beta) > c.esc_sideslip_limit:
            # sliding wide: beta's sign is against the turn's in a spin (r > 0, beta < 0), so
            # a moment of beta's sign stops it
            moment += c.esc_sideslip_gain * (beta - math.copysign(c.esc_sideslip_limit, beta))
            oversteer = True
        if moment == 0.0:
            return self._settle(0.0, None, r_ref, dt, False)
        # A moment to the left (+) brakes a left wheel, to the right a right wheel; the front
        # one against oversteer, the rear one against understeer.
        if moment > 0:
            wheel = FRONT_LEFT if oversteer else REAR_LEFT
        else:
            wheel = FRONT_RIGHT if oversteer else REAR_RIGHT
        return self._settle(moment, wheel, r_ref, dt, oversteer)

    def _settle(self, moment, wheel, r_ref, dt, oversteer):
        c = self.c
        torque = 0.0
        if wheel is not None:
            force = abs(moment) / (c.wheel_tread / 2)
            torque = min(force * c.wheel_radius, c.esc_max_brake_torque)
        a = 1 - math.exp(-dt / c.brake_time_constant) if c.brake_time_constant > 0 else 1.0
        for i in range(4):
            target = torque if i == wheel else 0.0
            self.brake[i] += (target - self.brake[i]) * a
            if self.brake[i] < 1.0 and target == 0.0:
                self.brake[i] = 0.0
        if oversteer:
            self.motor_factor = max(0.0, self.motor_factor - c.esc_motor_cut_rate * dt)
        else:
            self.motor_factor = min(1.0, self.motor_factor + c.esc_motor_restore_rate * dt)
        self.out = EscOut(tuple(self.brake), self.motor_factor, moment, r_ref)
        return self.out
