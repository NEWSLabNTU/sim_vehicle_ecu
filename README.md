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
| `powertrain` | `ideal` (acb's contract on CARLA) and `ev` (the Model 3 RWD drive unit: motor envelope, regen blended with the friction brakes and slip-controlled, EBD, ABS, traction control; each driven axle's differential: open, locked or a clutch-type LSD) |
| `esc` | stability control: one wheel's brake and a motor cut against a yaw the tires cannot hold |
| `brake` | the slip modulator ABS and traction control share |
| `actuators` | the motor's envelope, first-order lags, the hydraulics' dead time, the steering actuator |

## Regen slip control

An EV brakes with its motor first, on the driven axle. On a slick road that alone can lock the
axle: the drift car's 2812 N of regen on a rear that carries about 1630 N on ice turned the rear
wheels backwards with ABS off (autoware-drift issue 009). As a real EV's brake module blends
regen, `ev` does three things, all from the module's own estimates (the wheel slips, the wheel
speeds, the acceleration sensor), never the simulator's:

- **Slip control on regen.** The regen beyond the driven axle's share of the brakes (by the
  brake distribution) is cut while a driven wheel slips, by the modulator ABS and TC use, and
  the friction brakes take what it gives up. Regen within the axle's own share is left alone:
  cutting it would only hand it to the same axle's friction brakes. It releases past
  `regen_slip_release` (0.25), or when a driven wheel runs away: past `regen_runaway_slip`
  (0.06), slowing `regen_runaway_decel` (3 g) faster than the car, and its axle carrying
  `regen_runaway_deficit` (30 %) less than the brakes put on it -- a tyre past its peak, on any
  road (ice peaks at 0.05, asphalt at 0.25). Each of the three alone also fires on dry asphalt
  (a wheel taking up a new torque, or coming down from a wheelspin), the three together did not
  on Haruna or Tsukuba. A locked axle's slip is its two wheels' mean.
- **The axle's grip found.** During a slip event the axle's tyre force is measured from the
  wheels' own balance (brake and regen torque and the driveline's drag, less what the wheels'
  inertia gives up), which does not lag the tyre as the car's deceleration does; the event's
  highest, over the axle's load, is the axle's friction found (`axle_mu`).
- **Regen held within its share.** From then on regen beyond the axle's own share stays within
  `regen_grip_share` (0.7) of that grip. It is never held below what the axle carries without
  slipping, and forgotten at `regen_mu_recover_rate` (0.001 /s) while regen asks less.

On a grippy road regen never slips the driven axle, so none of it acts and the car is the one
before, bit for bit. `regen_slip_control: false` turns it off; a spec may set the fields in
`powertrain.regen_slip_control` (`enabled`, `slip_release`, `slip_reapply`, `runaway_slip`,
`runaway_decel`, `runaway_deficit`, `release_rate`, `apply_rate`, `grip_share`,
`mu_recover_rate`).

## Develop

```bash
just test    # unit tests
just check   # tests and pyflakes
```

## License

Apache-2.0.
