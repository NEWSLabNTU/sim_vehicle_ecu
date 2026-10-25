"""The drive unit's controller (VCU) and the brake module's longitudinal part, with their
actuators: how the by-wire request becomes wheel torques. Pure.

A powertrain turns the request (acceleration, velocity, gear), the car's speed and its
wheels (spin, longitudinal slip) into a drive torque and a brake torque for every wheel. The
speed and the slips are the brake module's estimates (`sim_vehicle_ecu.estimator`), not a
simulator's: `Ecu` hands them in. The physics then applies the torques with no engine model of
its own (PhysX's "drive none").

Models (`powertrain.model` in the spec):

- `ideal`: acb's contract on CARLA. The requested acceleration is realized as the force
  `m a + F_resist(v)`, split evenly over the four wheels as drive torque, capped per wheel; a
  negative force is braking (never the motor), split between the axles by the dynamic axle
  load. A standing car with no request, or in park, holds on its hold brake. No lag, no motor
  map: the car does what is asked within its limits.
- `ev`: an electric drive unit modelled on a real one (the Tesla Model 3 RWD's, in the
  shipped spec). The same feedforward `m a + F_resist(v)` becomes a motor torque request
  through the gear reduction; the motor delivers it through its torque/power envelope (peak
  torque to base speed, then constant power, nothing past its maximum speed) and a
  first-order torque lag. Deceleration is regenerative first (the motor as a generator,
  limited by its regen torque and power cap, fading out at low speed); the friction brakes
  cover the rest, through a hydraulic dead time and a first-order pressure lag, split by the
  dynamic axle load. ABS releases and reapplies each wheel's brake on its longitudinal slip;
  traction control cuts motor torque when a driven wheel spins. No hidden speed loop: what
  the model cannot deliver, Autoware's longitudinal controller sees and corrects, as on a
  real car.
- `physx` (reserved): PhysX's own DriveStandard (engine, gears, clutch, differential), with
  the request mapped to throttle and brake through accel/brake maps as a real vehicle
  interface would.
- `fmu` (reserved): an FMI 2.0 co-simulation unit (a supplier's powertrain model), stepped
  each substep.

The reserved models raise NotImplementedError from `make`, so a spec naming them fails loudly
at load instead of driving a different car.

The road load in the feedforward (`road_load`) is the car's estimate of it, from its
calibration; the simulator applies the road's own to the body.
"""
import math
from typing import NamedTuple

from .actuators import BRAKE_FLOOR, DeadTime, lag, motor_limit
from .brake import SlipModulator
from .calibration import G
from .signals import GEAR_NEUTRAL, GEAR_PARK, GEAR_REVERSE, IS_FRONT

MODELS = ("ideal", "physx", "ev", "fmu")
DRIVE_AXLES = ("rear", "front", "all")
# Below this speed rolling resistance ramps to zero, so a standing car is not pushed about.
ROLLING_RAMP = 0.5  # m/s


class Torques(NamedTuple):
    """Per wheel (FL, FR, RL, RR), N m. `drive` is signed along x; `brake` is a magnitude
    (the physics opposes the wheel's spin with it)."""

    drive: tuple
    brake: tuple


class Wheel(NamedTuple):
    """One wheel as the brake module sees it."""

    rotation_speed: float = 0.0  # rad/s, positive rolling forward: the wheel speed sensor
    slip: float = 0.0  # longitudinal slip, estimated: + faster than the road, - slower


def road_load(c, v):
    """The road load the car expects at a speed `v` (m/s along x), N, signed against `v`:
    rolling resistance (Crr m g, ramped in below ROLLING_RAMP) and aerodynamic drag
    (rho Cd A v^2 / 2)."""
    roll = c.rolling_coefficient * c.mass * G * min(1.0, abs(v) / ROLLING_RAMP)
    drag = 0.5 * c.air_density * c.drag_coefficient * c.frontal_area * v * v
    return -math.copysign(roll + drag, v) if v else 0.0


def feedforward(c, accel, speed_along):
    """The tractive force (N, along the gear's direction) that gives `accel` at
    `speed_along`: the car and its spinning wheels times the acceleration, plus the road load
    and the driveline's drag it must overcome."""
    mass = c.mass + 4 * 0.5 * c.wheel_mass  # four wheels at 0.5 m r^2: m_w / 2 each, reflected
    damping = 4 * c.wheel_damping_rate * speed_along / (c.wheel_radius ** 2)
    return mass * accel - road_load(c, speed_along) + damping


def regen_limit(c, omega, speed):
    """The most braking torque the motor may regenerate at `omega` (rad/s, motor) and road
    `speed` (m/s), N m: its regen torque limit and power cap, within the motor's envelope,
    faded linearly to 0 from `regen_fade_high` down to `regen_fade_low` m/s."""
    w = abs(omega)
    t = min(c.regen_max_torque, motor_limit(c, omega))
    if w > 0:
        t = min(t, c.regen_max_power / w)
    v = abs(speed)
    lo, hi = c.regen_fade_low, c.regen_fade_high
    if v <= lo:
        return 0.0
    if v < hi:
        t *= (v - lo) / (hi - lo)
    return t


class Powertrain:
    """The interface: `torques(req, speed, dt, wheels=None, esc=None) -> Torques`. `speed` is
    the car's forward speed (m/s), `wheels` the four `Wheel`s (None: none known yet), `esc`
    the stability control's request (`sim_vehicle_ecu.esc`), which only `ev` acts on: `ideal`
    has no brake hydraulics or ABS to put it through."""

    name = ""

    def __init__(self, calibration):
        self.c = calibration

    def torques(self, req, speed, dt, wheels=None, esc=None):  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self):
        """A new episode: forget any internal state (none for `ideal`)."""


def _holding(req, speed):
    return (req.gear == GEAR_PARK
            or (abs(req.velocity) < 0.01 and req.acceleration <= 0.01 and abs(speed) < 0.1))


def _direction(req):
    return -1.0 if req.gear == GEAR_REVERSE else 1.0


class IdealPowertrain(Powertrain):
    name = "ideal"

    def torques(self, req, speed, dt, wheels=None, esc=None):
        c = self.c
        r = c.wheel_radius
        if _holding(req, speed):
            return Torques((0.0,) * 4, (c.hold_brake_torque,) * 4)
        if req.gear == GEAR_NEUTRAL:
            return Torques((0.0,) * 4, (0.0,) * 4)
        direction = _direction(req)
        # Along the gear's direction: positive pushes that way, negative slows it down.
        push = feedforward(c, req.acceleration, direction * speed)
        if push >= 0:
            return Torques((direction * min(c.max_drive_torque, push * r / 4),) * 4, (0.0,) * 4)
        # Brakes, never the motor: decelerating to a stop does not drive it backwards. Split
        # by the dynamic axle loads, so a hard stop reaches the tires' limit on all four
        # together instead of locking the unloaded rear first.
        front = c.brake_split(-req.acceleration)
        total = -push * r
        f = min(total * front / 2, c.max_brake_torque)
        b = min(total * (1 - front) / 2, c.max_brake_torque)
        return Torques((0.0,) * 4, (f, f, b, b))


class EvPowertrain(Powertrain):
    name = "ev"

    def __init__(self, calibration):
        super().__init__(calibration)
        c = calibration
        if c.drive_axle not in DRIVE_AXLES:
            raise ValueError(f"drive_axle {c.drive_axle!r}: one of {', '.join(DRIVE_AXLES)}")
        for k in ("gear_ratio", "motor_peak_torque", "motor_peak_power", "motor_max_rpm"):
            if not getattr(c, k) > 0:
                raise ValueError(f"ev powertrain: {k} must be positive")
        self.front_share = {"rear": 0.0, "front": 1.0, "all": c.front_drive_share}[c.drive_axle]
        s = self.front_share
        # Each wheel's share of the axle torque: an open differential halves it.
        self.drive_share = (s / 2, s / 2, (1 - s) / 2, (1 - s) / 2)
        self.driven = tuple(x > 0 for x in self.drive_share)
        self.brake_delay = (DeadTime(c.brake_dead_time), DeadTime(c.brake_dead_time))
        self.abs = [SlipModulator(c.abs_slip_release, c.abs_slip_reapply,
                                  c.abs_release_rate, c.abs_apply_rate) for _ in range(4)]
        self.tc = SlipModulator(c.tc_slip_limit, c.tc_slip_reapply,
                                c.tc_release_rate, c.tc_apply_rate)
        self.reset()

    def reset(self):
        self.motor_torque = 0.0  # N m at the motor shaft, along the gear's direction
        self.pressure = [0.0, 0.0]  # friction brake torque per wheel, front and rear axle
        for d in self.brake_delay:
            d.reset()
        for m in self.abs:
            m.reset()
        self.tc.reset()
        self.abs_event = False  # ABS has worked a wheel in this stop: no regen until it ends
        self.debug = {}

    @property
    def slipping(self):
        """ABS or traction control is working a wheel."""
        return self.tc.active or any(m.active for m in self.abs)

    def motor_speed(self, speed, wheels):
        """Motor shaft speed, rad/s: the driven wheels' mean spin (the car's speed when the
        wheels are not known) through the reduction."""
        if wheels:
            spins = [w.rotation_speed for w, d in zip(wheels, self.driven) if d]
            wheel = sum(spins) / len(spins)
        else:
            wheel = speed / self.c.wheel_radius
        return wheel * self.c.gear_ratio

    def torques(self, req, speed, dt, wheels=None, esc=None):
        """`esc`: a `sim_vehicle_ecu.esc.EscOut` of this step, its brake added to each wheel's
        ahead of ABS and its factor on the motor's torque."""
        c = self.c
        r, ratio = c.wheel_radius, c.gear_ratio
        direction = _direction(req)
        omega = self.motor_speed(speed, wheels)
        hold = _holding(req, speed)
        regen_force = 0.0
        friction = (0.0, 0.0)  # per wheel friction torque requests, front and rear axle
        if hold or req.gear == GEAR_NEUTRAL or abs(speed) < c.abs_min_speed:
            self.abs_event = False
        if hold or req.gear == GEAR_NEUTRAL:
            request = 0.0
            if hold:
                friction = (c.hold_brake_torque, c.hold_brake_torque)
        else:
            push = feedforward(c, req.acceleration, direction * speed)
            if push >= 0:
                self.abs_event = False
                request = min(push * r / ratio, motor_limit(c, omega))
            else:
                brake = -push
                # Regeneration first, unless ABS has worked a wheel in this stop: the brake
                # controller then hands all braking to the friction brakes until the stop
                # ends (handing it back mid-stop would leave a gap the size of the
                # hydraulics' lag every time ABS lets go).
                if any(m.active for m in self.abs):
                    self.abs_event = True
                available = 0.0 if self.abs_event else regen_limit(c, omega, speed)
                regen_force = min(brake, available * ratio / r)
                request = -regen_force * r / ratio
                # The axle split follows the deceleration the car can reach: an EBD works
                # from the measured deceleration, which the tires cap near mu g.
                decel = min(-req.acceleration, c.tire_friction * G)
                friction = self.split_friction(brake, regen_force, decel)
        # The motor: the inverter's torque lag, within the envelope at the present speed.
        limit = motor_limit(c, omega)
        self.motor_torque = lag(self.motor_torque, request, dt, c.motor_time_constant)
        self.motor_torque = max(-limit, min(limit, self.motor_torque))
        delivered = self.motor_torque
        if esc is not None and delivered > 0:
            delivered *= esc.motor_factor
        # Traction control: a driven wheel spinning past its slip limit cuts motor torque.
        if wheels and c.tc_enabled and delivered > 0:
            spin = max(direction * w.slip for w, d in zip(wheels, self.driven) if d)
            delivered *= self.tc.step(spin, dt)
        else:
            self.tc.step(0.0, dt, enabled=False)
        axle = direction * delivered * ratio
        drive = tuple(axle * s for s in self.drive_share)
        # The friction brakes: dead time, then the pressure's first-order build-up, per axle.
        for i in range(2):
            delayed = self.brake_delay[i].step(friction[i], dt)
            self.pressure[i] = lag(self.pressure[i], delayed, dt, c.brake_time_constant)
            if self.pressure[i] < BRAKE_FLOOR and delayed < BRAKE_FLOOR:
                self.pressure[i] = 0.0
        # ABS per wheel on its slip against the direction of travel (regen counts as braking
        # on a driven wheel); off below abs_min_speed, where the slip means little.
        abs_on = (c.abs_enabled and wheels is not None and not hold
                  and abs(speed) > c.abs_min_speed)
        brake = []
        for i in range(4):
            pressure = self.pressure[0 if IS_FRONT[i] else 1] + (esc.brake[i] if esc else 0.0)
            lock = -direction * wheels[i].slip if wheels else 0.0
            braking = pressure > 1.0 or (self.driven[i] and self.motor_torque < 0)
            factor = self.abs[i].step(lock, dt, enabled=abs_on and braking)
            brake.append(min(pressure * factor, c.max_brake_torque))
        self.debug = {"motor_torque": self.motor_torque, "delivered": delivered,
                      "motor_limit": limit, "omega": omega, "regen_force": regen_force,
                      "friction_request": friction, "pressure": tuple(self.pressure),
                      "abs": tuple(m.factor for m in self.abs), "tc": self.tc.factor}
        return Torques(drive, tuple(brake))

    def split_friction(self, brake, regen, decel):
        """Per wheel friction torque (front, rear) for a total braking force `brake` (N) of
        which the motor regenerates `regen` on the driven axle(s): the dynamic axle load
        distribution of the total, with regen taking its axle's share first; regen beyond
        that share (mild braking) comes off the other axle's friction."""
        c = self.c
        share = c.brake_split(decel)
        front, rear = brake * share, brake * (1 - share)
        regen_front, regen_rear = regen * self.front_share, regen * (1 - self.front_share)
        excess = max(0.0, regen_front - front) + max(0.0, regen_rear - rear)
        front = max(0.0, front - regen_front)
        rear = max(0.0, rear - regen_rear)
        cut = min(front, excess)
        front, excess = front - cut, excess - cut
        rear -= min(rear, excess)
        return front * c.wheel_radius / 2, rear * c.wheel_radius / 2


def make(calibration):
    """The powertrain `calibration.powertrain` names."""
    name = calibration.powertrain or "ideal"
    if name == "ideal":
        return IdealPowertrain(calibration)
    if name == "ev":
        return EvPowertrain(calibration)
    if name in MODELS:
        raise NotImplementedError(
            f"powertrain {name!r} is reserved and not implemented yet; 'ideal' and 'ev' are")
    raise ValueError(f"unknown powertrain {name!r}; known: {', '.join(MODELS)}")
