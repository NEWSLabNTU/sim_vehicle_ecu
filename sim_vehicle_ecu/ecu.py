"""The car's onboard computers as one unit: `Ecu.step(request, sensors, dt) -> Actuation`. Pure.

Each step, in the order a car's would act on one sample:

1. the power steering (EPS): the requested tire angle through the steering actuator;
2. the brake module's estimates (`estimator`): reference speed, slips, sideslip, friction,
   from the sensors and what the module was doing last step;
3. the stability control (`esc`): one wheel's brake and a motor cut, from the estimates and
   the yaw rate sensor;
4. the drive unit's controller and the brake module's longitudinal part (`powertrain`):
   motor torque and each wheel's brake, through ABS and traction control.

The host steps it every physics substep with the request the vehicle interface holds and what
the car's sensors read, and applies the actuation to the wheels. It reads nothing else.
"""
from .actuators import BRAKE_FLOOR, steer
from .esc import Esc
from .estimator import Activity, Estimator
from .powertrain import make
from .signals import Actuation


class Ecu:
    def __init__(self, calibration, estimator=None):
        """`estimator`: the brake module's (default `Estimator`); a test may hand in another
        with the same `step(sensors, dt, tire_angle, activity)` and `reset()`."""
        self.c = calibration
        self.powertrain = make(calibration)
        self.esc = Esc(calibration)
        driven = getattr(self.powertrain, "driven", (True,) * 4)
        self.estimator = estimator or Estimator(calibration, driven)
        self.reset()

    def reset(self):
        """A new episode (a spawn, a teleport): every controller and actuator at rest."""
        self.tire_angle = 0.0
        self.powertrain.reset()
        self.esc.reset()
        self.estimator.reset()
        self.last = None  # the last step's Actuation
        self.debug = {}

    def step(self, request, sensors, dt):
        """The actuation for one step of `dt` s. `sensors`: this step's `Sensors`, or None
        before the car has any reading (its first physics step)."""
        c = self.c
        self.tire_angle = steer(self.tire_angle, request.tire_angle, dt, c.max_steer_angle,
                                c.steer_time_constant, c.steer_rate_limit)
        est = self.estimator.step(sensors, dt, self.tire_angle, self._activity())
        yaw_rate = sensors.yaw_rate if sensors is not None else 0.0
        esc = self.esc.step(est.speed, yaw_rate, est.sideslip, self.tire_angle, est.friction, dt)
        torques = self.powertrain.torques(request, est.speed, dt, est.wheels, esc)
        self.last = Actuation(torques.drive, torques.brake, self.tire_angle)
        self.debug = {**getattr(self.powertrain, "debug", {}),
                      "esc": (tuple(round(b) for b in esc.brake), esc.motor_factor),
                      "estimate": {"speed": est.speed, "sideslip": est.sideslip,
                                   "friction": est.friction,
                                   "slip": est.wheels and tuple(w.slip for w in est.wheels)}}
        return self.last

    def _activity(self):
        """What the brake module was doing last step."""
        if self.last is None:
            return Activity()
        pt = self.powertrain
        motor = getattr(pt, "motor_torque", None)
        if motor is None:  # ideal: no motor of its own, the torques are the request
            braking = any(b > BRAKE_FLOOR for b in self.last.brake)
            driving = any(d != 0.0 for d in self.last.drive)
        else:  # the motor's torque is along the gear's direction: < 0 is regen
            braking = any(b > BRAKE_FLOOR for b in self.last.brake) or motor < 0
            driving = motor > 0
        abs_ = getattr(pt, "abs", None)
        tc = getattr(pt, "tc", None)
        return Activity(braking=braking, driving=driving,
                        abs_active=bool(abs_) and any(m.active for m in abs_),
                        tc_active=tc is not None and tc.active)
