"""The car's control component: the unit a host steps every physics substep (scensim roadmap 091).

A component is what sits between the vehicle interface and the wheels: on the shipped cars the
onboard computers (`Ecu`: steering, the brake module's estimates, ESC, the powertrain), on a
user's car their own model of it (a RoboRacer's VESC and servo, say). The host (aib) asks
for one by name and steps it; it never knows which it has.

The interface (docs/design/roboracer-indoor-twin.md in scensim, section 2):

    class MyCar:
        commands = "autoware"          # what it takes: one of COMMANDS
        driven = (False, False, True, True)  # optional: the wheels it drives (default all)
        brake_floor = 50.0             # optional, N m: a brake below this is released

        def __init__(self, vehicle: dict, params: dict, seed: int): ...
        def reset(self): ...
        def step(self, command, sensors, dt) -> Actuation: ...

- `vehicle`: the spec's document as the simulator's body sees it (sim_vehicle_spec, v2);
  `params`: the spec's `powertrain.params` (or a run's override), as is; `seed`: the run's.
- `step` gets the last command (None before the first), the car's own sensor readings after
  the last step (`signals.Sensors`, None on the first step after a reset) and the step's
  length in s, and returns `signals.Actuation` in SI units, wheels FL, FR, RL, RR. It sees no
  ground truth, does no I/O, and is deterministic given its seed.
- `reset` comes before an episode's first step, after a teleport and after a respawn.

Components are found by name: the entry points of group `GROUP` in the importing interpreter
(a user's package declares its own), and the shipped ones, `ideal` and `ev` (`Ecu` with that
powertrain), which a spec names as `powertrain.model` without an entry point. Pure: the
standard library only.
"""
import math
from importlib import metadata

from .calibration import Calibration
from .ecu import Ecu

GROUP = "scensim.vehicle_components"
# What a component may take (signals: Request for `autoware`; the others come with their
# vehicle interfaces, scensim roadmap 091 step 5).
COMMANDS = ("autoware", "ackermann", "vesc")
# The shipped components a spec names by `powertrain.model` alone.
SHIPPED = ("ideal", "ev")


class ComponentError(RuntimeError):
    """A component that cannot be found, loaded or used; the message names the fix."""


class EcuComponent:
    """The shipped onboard computers (`Ecu`) as a component: Autoware's by-wire request in.
    `params` are calibration overrides by field name (an ego config's `ecu:`)."""

    commands = "autoware"
    powertrain = None  # the model it forces; None: the spec's

    def __init__(self, vehicle, params=None, seed=0):
        overrides = dict(params or {})
        if self.powertrain:
            overrides["powertrain"] = self.powertrain
        self._use(Calibration.from_spec(vehicle, overrides))

    @classmethod
    def from_calibration(cls, calibration):
        """One on a ready calibration (aib's rig, which applies its own overrides)."""
        self = cls.__new__(cls)
        self._use(calibration)
        return self

    def _use(self, c):
        self.calibration = c
        self.ecu = Ecu(c)
        self.brake_floor = c.brake_floor
        if c.powertrain == "ideal" or c.drive_axle == "all":
            self.driven = (True,) * 4
        else:
            front = c.drive_axle == "front"
            self.driven = (front, front, not front, not front)

    @property
    def debug(self):
        return self.ecu.debug

    def reset(self):
        self.ecu.reset()

    def step(self, command, sensors, dt):
        return self.ecu.step(command, sensors, dt)


class Ideal(EcuComponent):
    powertrain = "ideal"


class Ev(EcuComponent):
    powertrain = "ev"


def _entry_point(name):
    eps = metadata.entry_points()
    group = eps.select(group=GROUP) if hasattr(eps, "select") else eps.get(GROUP, ())
    found = [ep for ep in group if ep.name == name]
    if not found:
        known = sorted({ep.name for ep in group} | set(SHIPPED))
        raise ComponentError(
            f"no control component {name!r} in entry point group {GROUP} (known: "
            f"{', '.join(known) or 'none'}); fix: install the package that declares it into a "
            "workspace the pilot overlays, built, and check its entry_points")
    if len({ep.value for ep in found}) > 1:
        raise ComponentError(f"control component {name!r} is declared twice in {GROUP}: "
                             + ", ".join(sorted({ep.value for ep in found})))
    return found[0]


def resolve(name):
    """The class (or factory) a component name stands for."""
    if name == "ideal":
        return Ideal
    if name == "ev":
        return Ev
    ep = _entry_point(name)
    try:
        return ep.load()
    except Exception as e:  # noqa: BLE001 - whatever the user's import raises
        raise ComponentError(f"control component {name!r} ({ep.value}) does not import: "
                             f"{type(e).__name__}: {e}") from e


def check(component, name="component"):
    """Refuse an object that is not a component (ComponentError); returns it."""
    commands = getattr(component, "commands", None)
    if commands not in COMMANDS:
        raise ComponentError(f"{name}: commands = {commands!r}, not one of "
                             f"{', '.join(COMMANDS)}")
    for m in ("reset", "step"):
        if not callable(getattr(component, m, None)):
            raise ComponentError(f"{name}: no {m}()")
    driven = getattr(component, "driven", (True,) * 4)
    if not (isinstance(driven, tuple) and len(driven) == 4):
        raise ComponentError(f"{name}: driven must be four booleans (FL, FR, RL, RR)")
    return component


def load(name, vehicle, params=None, seed=0):
    """The component `name` for the car `vehicle` (a spec document), ready to step."""
    cls = resolve(name)
    try:
        component = cls(vehicle, dict(params or {}), int(seed))
    except ComponentError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ComponentError(f"control component {name!r} refused its car or params: "
                             f"{type(e).__name__}: {e}") from e
    return check(component, name)


def for_spec(vehicle, overrides=None, seed=0, component=None):
    """The component a spec's car runs: its `powertrain.model` (`ideal`, `ev`: the shipped
    onboard computers, `overrides` their calibration overrides), or for `component` the entry
    point `powertrain.component` with `powertrain.params`. `component`: a run's
    `{entry_point, params}` in place of the spec's."""
    pt = vehicle.get("powertrain") or {}
    if component:
        return load(component["entry_point"], vehicle, component.get("params"), seed)
    if pt.get("model") == "component":
        return load(pt["component"], vehicle, pt.get("params"), seed)
    return load(pt.get("model") or "ideal", vehicle, overrides, seed)


def checked(act, name="component"):
    """The actuation, or ComponentError when it has the wrong shape or a value that is not
    finite: a host stops the run rather than drive a different car."""
    try:
        drive, brake, angle = tuple(act.drive), tuple(act.brake), float(act.tire_angle)
    except Exception as e:  # noqa: BLE001
        raise ComponentError(f"{name}: step() returned {act!r}, not an Actuation") from e
    if len(drive) != 4 or len(brake) != 4:
        raise ComponentError(f"{name}: step() returned {len(drive)} drive and {len(brake)} "
                             "brake torques, not 4 each (FL, FR, RL, RR)")
    values = [float(v) for v in drive + brake] + [angle]
    if not all(math.isfinite(v) for v in values):
        raise ComponentError(f"{name}: step() returned a value that is not finite: {act!r}")
    if any(b < 0 for b in brake):
        raise ComponentError(f"{name}: step() returned a negative brake torque: {brake}")
    return act


def timing(component, command, sensors, dt, calls=1000):
    """The mean and maximum seconds of `calls` steps on a canned input, after a reset (the
    host's cost check; the budget is 60 / physics_hz ms per call)."""
    import time

    component.reset()
    worst, total = 0.0, 0.0
    for _ in range(calls):
        t0 = time.perf_counter()
        component.step(command, sensors, dt)
        d = time.perf_counter() - t0
        total += d
        worst = max(worst, d)
    component.reset()
    return total / calls, worst


def budget(physics_hz):
    """The per-call budget, s: 0.1 ms at 600 Hz, scaling with the rate (60 / f ms)."""
    return 0.06 / float(physics_hz)


__all__ = ["COMMANDS", "ComponentError", "EcuComponent", "Ev", "GROUP", "Ideal", "SHIPPED",
           "budget", "check", "checked", "for_spec", "load", "resolve", "timing"]
