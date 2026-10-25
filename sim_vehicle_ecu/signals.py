"""What crosses the onboard computers' boundary: the by-wire request in, the car's own sensor
readings in, the actuation out. Pure.

Every per-wheel tuple is in one order: front left, front right, rear left, rear right.
"""
from typing import NamedTuple

FRONT_LEFT, FRONT_RIGHT, REAR_LEFT, REAR_RIGHT = range(4)
IS_FRONT = (True, True, False, False)

# Autoware's GearCommand / GearReport values, which the by-wire request carries
GEAR_NEUTRAL, GEAR_DRIVE, GEAR_REVERSE, GEAR_PARK = 1, 2, 20, 22


class Request(NamedTuple):
    """The by-wire request: what the vehicle interface asks of the car."""

    acceleration: float = 0.0  # m/s^2, along the gear's direction
    velocity: float = 0.0  # m/s, the planner's target (0: stopping, so a standing car holds)
    tire_angle: float = 0.0  # bicycle tire angle, rad, + left
    gear: int = GEAR_DRIVE


class Sensors(NamedTuple):
    """What the car's own sensors read: the four wheel speed sensors and the brake module's
    yaw-rate and acceleration sensor. Body frame: x forward, y left, z up."""

    wheel_speeds: tuple = (0.0,) * 4  # rad/s, + rolling forward
    yaw_rate: float = 0.0  # rad/s, + turning left
    accel_x: float = 0.0  # m/s^2
    accel_y: float = 0.0  # m/s^2, + to the left


class Actuation(NamedTuple):
    """What the onboard computers and their actuators put on the wheels."""

    drive: tuple  # N m per wheel, signed along x
    brake: tuple  # N m per wheel, a magnitude (it opposes the wheel's spin)
    tire_angle: float  # the steering actuator's bicycle tire angle, rad
