# sim_vehicle_ecu

A simulated car's onboard computers, below its vehicle interface: the drive unit's controller
(VCU), the brake module (brake distribution, ABS, traction control, ESC, and the estimates
they work from) and the power steering (EPS), with the actuators a drive-none physics vehicle
lacks: the motor's envelope and torque lag, the brake hydraulics, the steering motor.

Pure Python 3, the standard library only: no simulator and no ROS, so Isaac Sim's Python 3.12
and the system's both import it. It is part of
[scenario-world-bridge](https://github.com/NEWSLabNTU/scenario-world-bridge) (roadmap 055);
the Isaac vehicle bridge, [autoware_isaac_bridge](https://github.com/NEWSLabNTU/autoware_isaac_bridge)
(aib), steps it every physics substep.

## Where it sits

| layer | real car | here |
|---|---|---|
| planning, high-level control | the autopilot (Autoware): acceleration, tire angle | Autoware |
| vehicle interface | the drive-by-wire adapter | a vehicle bridge (aib on Isaac) |
| onboard computers | VCU, brake module, EPS | **this repository** |
| the car's body, tires, ground | physics | the simulator (PhysX) |

The onboard computers see what a real car's see: the by-wire request and the car's own
sensors (four wheel speeds, a yaw rate, two accelerations). They never read the simulator's
ground truth. The car's speed over the ground, each wheel's slip, the sideslip and the road's
friction are their own estimates (`estimator`), as on a real brake module.

CARLA does not use it: CARLA 0.9.16 takes throttle, brake and steering, not per-wheel torque,
and its car has its own model.

## Use

```python
from sim_vehicle_ecu import Calibration, Ecu, Request, Sensors

ecu = Ecu(Calibration.from_spec(spec.doc, overrides={"esc_enabled": True}))
# every physics substep:
act = ecu.step(Request(acceleration, velocity, tire_angle, gear),
               Sensors(wheel_speeds, yaw_rate, accel_x, accel_y), dt)
act.drive, act.brake   # N m per wheel: front left, front right, rear left, rear right
act.tire_angle         # the steering actuator's bicycle tire angle, rad
ecu.reset()            # a spawn or a teleport
```

`Sensors` is None before the car has a reading (its first physics step). `spec.doc` is a
loaded [sim_vehicle_spec](https://github.com/NEWSLabNTU/sim_vehicle_spec) spec's document
(version 2); `Calibration.from_spec` reads its `geometry`, `inertia`, `wheels`, `tires`,
`brakes` (with `abs`, and `esc` when there is one), `powertrain` (with `traction_control` and `differential`),
`resistance` and `steering`, and the defaults hold for the rest. The defaults are the shipped
Tesla Model 3 RWD's, so `Calibration()` is that car.

## What is in it

| module | what |
|---|---|
| `signals` | `Request`, `Sensors`, `Actuation`; the wheel order and the gears |
| `calibration` | `Calibration`, from a spec and overrides |
| `ecu` | `Ecu`: steering, estimates, ESC, powertrain, in that order every step |
| `estimator` | reference speed, wheel slip, sideslip, friction, from the sensors |
| `powertrain` | `ideal` (acb's contract on CARLA) and `ev` (the Model 3 RWD drive unit: motor envelope, regen blended with the friction brakes, EBD, ABS, traction control; each driven axle's differential: open, locked or a clutch-type LSD) |
| `esc` | stability control: one wheel's brake and a motor cut against a yaw the tires cannot hold |
| `brake` | the slip modulator ABS and traction control share |
| `actuators` | the motor's envelope, first-order lags, the hydraulics' dead time, the steering actuator |

## Develop

```bash
just test    # unit tests
just check   # tests and pyflakes
```

## License

Apache-2.0.
