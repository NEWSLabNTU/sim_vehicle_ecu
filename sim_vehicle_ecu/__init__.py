"""sim_vehicle_ecu: a simulated car's onboard computers, below its vehicle interface.

The drive unit's controller (VCU), the brake module (brake distribution, ABS, traction control,
ESC, and the estimates they work from) and the power steering, with the actuators a
drive-none physics vehicle lacks (motor, brake hydraulics, steering motor). Pure Python, the
standard library only: no simulator, no ROS.

    from sim_vehicle_ecu import Calibration, Ecu, Request, Sensors
    ecu = Ecu(Calibration.from_spec(spec.doc))
    act = ecu.step(Request(acceleration=1.0, velocity=10.0), Sensors(wheel_speeds=...), 1 / 60)

Modules: `signals` (Request, Sensors, Actuation), `calibration`, `ecu`, `estimator`,
`powertrain`, `esc`, `brake`, `actuators`.
"""
from .calibration import Calibration
from .ecu import Ecu
from .signals import (GEAR_DRIVE, GEAR_NEUTRAL, GEAR_PARK, GEAR_REVERSE, Actuation, Request,
                      Sensors)

__all__ = ["Actuation", "Calibration", "Ecu", "GEAR_DRIVE", "GEAR_NEUTRAL", "GEAR_PARK",
           "GEAR_REVERSE", "Request", "Sensors"]
