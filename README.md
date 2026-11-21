# sim_vehicle_ecu

A simulated car's onboard computers, below its vehicle interface: the drive unit's controller
(VCU), the brake module (brake distribution, ABS, traction control, ESC, and the estimates
they work from) and the power steering (EPS), with the actuators a drive-none physics vehicle
lacks: the motor's envelope and torque lag, the brake hydraulics, the steering motor.

Pure Python 3, the standard library only: no simulator and no ROS, so Isaac Sim's Python 3.12
and the system's both import it. It is part of
[scensim](https://github.com/NEWSLabNTU/scensim) (roadmap 055);
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
Tesla Model 3 RWD's on CARLA 0.9.16's body, so `Calibration()` is that car. A spec with
`bodies` is calibrated as the body its simulator drives, `spec.effective(spec, body).doc`;
`from_spec` refuses the spec's own document, since the wheel radius and the reduction are the
body's.

## Your own control component

A car may run its owner's model of its control component instead (scensim roadmap 091): a
RoboRacer's speed controller and servo, say. It is a class in the owner's own package, found by
its entry point in group `scensim.vehicle_components`, and a spec names it:

```yaml
powertrain: {model: component, component: my_racer.vesc_servo, params: {erpm_gain: 4614.0}}
```

```python
from sim_vehicle_ecu import component

car = component.for_spec(spec_doc, seed=run_seed)   # ideal, ev, or the spec's own component
act = component.checked(car.step(command, sensors, dt))
```

The interface (`commands`, `__init__(vehicle, params, seed)`, `reset()`, `step(command, sensors,
dt)`) and its rules are in `sim_vehicle_ecu/component.py` and in scensim's
docs/design/roboracer-indoor-twin.md. It sees the car's own sensors; ground truth only as it
declares it (`truth = ("speed", ...)`, from `TRUTH_INPUTS`: what its user also measured on the
real car), in `sensors.truth`, through a `TruthFeed` with each input's latency and noise.
`ideal` and `ev` are components too (`Ecu` with that
powertrain), declared in the same group. The component must be pure Python: Isaac Sim's
interpreter imports it.

## A compiled model: the `fmu` component

A model that is not Python (C, C++, Simulink, Modelica, a vendor's binary) is exported as an FMI
2.0 co-simulation unit, and the shipped `fmu` component steps it. Its params name the unit,
relative to the spec's directory (the host passes it as `base`), and map the unit's variables to
the command, the sensors and the actuation:

```yaml
powertrain:
  model: component
  component: fmu
  params:
    fmu: models/racer.fmu
    commands: vesc
    inputs: {speed_ref: command.value, servo: command.servo, wheel_rl: "sensors.wheel_speeds[2]"}
    outputs: {"drive[2]": torque_rl, tire_angle: steer, telemetry.erpm: erpm}
    parameters: {kp: 0.8}
    seed: seed
```

The sources, targets and rules are in `sim_vehicle_ecu/fmu.py`. Every name is checked against the
unit's modelDescription.xml when the component is made. A unit that cannot take a variable step
size refuses a changed dt. It needs nothing beyond the standard library (ctypes): a Linux x86_64
binary (`binaries/linux64/`), unpacked once per sha256 under `SCENSIM_FMU_CACHE` (default
`~/.cache/scensim/fmu`). Each car is its own instance of the unit, so one process can step
many, unless the unit says `canBeInstantiatedOnlyOncePerProcess`. **A `.fmu` is native code**: its
binary runs inside the world server's process with your rights, as any component's code does, so
take a unit only from a source you would take a program from. A unit's log messages are kept
as their format strings: a ctypes callback cannot read C varargs.

## What is in it

| module | what |
|---|---|
| `signals` | `Request`, `Sensors`, `Actuation`; the wheel order and the gears |
| `component` | the control component interface: the shipped `ideal`, `ev` and `fmu`, a user's own by entry point, the host's checks |
| `fmu` | the `fmu` component: an FMI 2.0 co-simulation unit, stepped through ctypes |
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
  cutting it would only hand it to the same axle's friction brakes. It releases on a lost grip:
  a driven wheel running away (past `regen_runaway_slip`, 0.06, slowing `regen_runaway_decel`,
  3 g, faster than the car, and its axle carrying `regen_runaway_deficit`, 30 %, less than the
  brakes put on it) -- a tyre past its peak, on any road (ice peaks at 0.05, asphalt at 0.25) --
  or a driven wheel past `regen_slip_release` (0.25) whose axle is that short or whose slip is
  still growing (slowing `regen_slip_growth`, 1 g, faster than the car: a tyre that keeps its
  force past the peak still locks when the brakes ask more than it carries). A slip alone is
  not one: wet mud's tyre peaks at 0.2 and keeps its force past it, so a rear braking at 0.3
  slip there holds it, carrying what the brakes put on it, and cutting regen would only move
  the braking to the front (autoware-drift issue 011). Once released, the slip alone, as ABS.
  The car's deceleration lags a reversal of the tyres' force, so neither test is read until
  the driven wheels have been out of a drive slip for `regen_spin_settle` (0.1 s): a wheel
  coming down from a wheelspin into regen slows as fast as a runaway one. Each test alone also
  fires on dry asphalt (a wheel taking up a new torque, or coming down from a wheelspin),
  together they did not on Haruna or Tsukuba. A locked axle's slip is its two wheels' mean.
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
`runaway_decel`, `runaway_deficit`, `slip_growth`, `spin_settle`, `release_rate`,
`apply_rate`, `grip_share`, `mu_recover_rate`).

## The batched form

`sim_vehicle_ecu.batch` steps N cars (slots) at once on arrays, for a batched executor (scensim
roadmap 092 S3; the contract is scensim's docs/design/roboracer-indoor-twin.md, section 2.6):

```python
from sim_vehicle_ecu import batch
b = batch.load_batch("ev", vehicle, params, seeds=[1, 2, 3])
b.reset_batch(np.ones(3, bool))
act = b.step_batch(commands, sensors, dt)   # dicts of arrays, slot first
```

A component with a batched form declares `batch = <class>` (built as `cls(vehicle, params,
seeds)`), and `batch.ScalarOf` gives its scalar form from it, so the two are one code. Every
other component, `ideal`, `ev` and `fmu` included, runs as `LoopBatch`: N instances, each stepped
by `step`. That is correct, but it is one Python step per slot, and `vectorised = False`, so
`require_vectorised` refuses it by name. The batched form equals N scalar steps bit for bit when
it computes with `+ - * /`, sqrt, min/max/clip, where and comparisons; numpy's transcendental
kernels can differ from `math` in the last bit. numpy is imported by `batch` only.

## Develop

```bash
just test    # unit tests
just check   # tests and pyflakes
```

It is also a ROS package (`ament_python`, `package.xml` at the root), so a colcon workspace
that holds it builds and installs it like any other: `colcon build --packages-select
sim_vehicle_ecu` puts the module in the install's `lib/python3.10/site-packages`, with no
dependency beyond Python. Isaac Sim's Python imports it from the source tree as before.

## License

Apache-2.0. The FMI 2.0 headers in `test/fmi2/`, used only to build the test unit, are the
Modelica Association's, under the 2-Clause BSD License (`test/fmi2/LICENSE.txt`).
