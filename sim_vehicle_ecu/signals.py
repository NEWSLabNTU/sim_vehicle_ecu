"""What crosses the onboard computers' boundary: the by-wire request in, the car's own sensor
readings in, the actuation out. Pure.

Every per-wheel tuple is in one order: front left, front right, rear left, rear right.
"""
import math
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
    # the ground truth a control component declared (sim_vehicle_ecu.component.TRUTH_INPUTS),
    # as the host delivers it (latency, noise); None for every other component
    truth: object = None


class Actuation(NamedTuple):
    """What the onboard computers and their actuators put on the wheels."""

    drive: tuple  # N m per wheel, signed along x
    brake: tuple  # N m per wheel, a magnitude (it opposes the wheel's spin)
    tire_angle: float  # the steering actuator's bicycle tire angle, rad
    # what the real controller reports (a VESC: erpm, duty, current_motor, ...), in its own
    # units, for the vehicle interface to publish and the run to record; None: nothing
    telemetry: object = None


# A VESC's motor command modes, as its driver's command topics name them (vesc_driver:
# commands/motor/<topic>): the value's unit is ERPM, duty (-1 to 1), A, A and rad.
MOTOR_MODES = ("speed", "duty", "current", "brake", "position")


class AckermannCommand(NamedTuple):
    """The last ``ackermann_msgs/AckermannDrive`` the vehicle interface received (the
    ``ackermann`` commands, scensim roadmap 091): the message's own semantics, kept as they are.
    A 0 ``acceleration`` or ``steering_angle_velocity`` means "as fast as possible"."""

    speed: float = 0.0  # m/s, the desired forward speed (negative: backwards)
    acceleration: float = 0.0  # m/s^2, a magnitude
    jerk: float = 0.0  # m/s^3, a magnitude
    steering_angle: float = 0.0  # rad, the virtual bicycle angle, + left
    steering_angle_velocity: float = 0.0  # rad/s, a magnitude
    age: float = math.inf  # s since it arrived; inf: none has


class VescCommand(NamedTuple):
    """What a VESC was last told (the ``vesc`` commands, scensim roadmap 091), after its
    driver's limits: the last motor command (the last message on any of the motor topics wins,
    as on the driver), and the servo's position. The component decides what an old command
    means (a real VESC stops after its own timeout)."""

    motor: str = None  # one of MOTOR_MODES; None: no motor command yet
    value: float = 0.0  # in that mode's unit
    servo: float = None  # 0 to 1; None: no servo command yet (the servo stays where it is)
    age: float = math.inf  # s since the motor command arrived; inf: none has
