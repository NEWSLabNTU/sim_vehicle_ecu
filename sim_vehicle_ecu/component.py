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
- `step` gets the last command (`COMMAND_TYPES`: for `autoware` None before the first; for
  `ackermann` and `vesc` always one, its `age` infinite before the first), the car's own sensor
  readings after the last step (`signals.Sensors`, None on the first step after a reset) and
  the step's length in s, and returns `signals.Actuation` in SI units, wheels FL, FR, RL, RR. It sees no
  ground truth, does no I/O, and is deterministic given its seed.
- `reset` comes before an episode's first step, after a teleport and after a respawn.
- Ground truth: none by default. A component may declare inputs from `TRUTH_INPUTS` it needs
  (`truth = ("speed", ...)`), only quantities its user also measured on the real car (motion
  capture, say); it gets them in `sensors.truth`, a mapping of the declared names only, through
  a `TruthFeed` (each with its own latency and noise, set per run).

Components are found by name: the entry points of group `GROUP` in the importing interpreter
(a user's package declares its own), and the shipped ones, `ideal` and `ev` (`Ecu` with that
powertrain), which a spec names as `powertrain.model` without an entry point; and `fmu`
(sim_vehicle_ecu.fmu: an FMI 2.0 co-simulation unit, named as a component: `component: fmu`).
Pure: the standard library only.
"""
import math
import random
from importlib import metadata

from .calibration import Calibration
from .ecu import Ecu
from .ev_batch import EvBatch

GROUP = "scensim.vehicle_components"
# What a component may take, and the command its `step` gets for each (signals): the vehicle
# interface's profile gives one (scensim roadmap 091).
COMMANDS = ("autoware", "ackermann", "vesc")
COMMAND_TYPES = {"autoware": "Request", "ackermann": "AckermannCommand", "vesc": "VescCommand"}
# The shipped components a spec names by `powertrain.model` alone.
SHIPPED = ("ideal", "ev", "fmu")
# The ground truth a component may declare (scensim roadmap 091: only what the user also measured
# on the real car), with its unit and shape.
TRUTH_INPUTS = {
    "pose": "m, m, rad: x, y, yaw in the map frame",
    "yaw": "rad: heading in the map frame",
    "velocity": "m/s, m/s: the body's velocity, x and y in the body frame",
    "speed": "m/s: forward speed over the ground",
    "yaw_rate": "rad/s: the body's, true",
    "wheel_slip": "1 x 4: each wheel's longitudinal slip (FL, FR, RL, RR)",
    "sideslip": "rad: the body's sideslip angle",
}
TRUTH_SETTINGS = ("latency_s", "noise_std")


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
    """The scalar form stays `Ecu` itself (its `debug`, `from_calibration`); `batch` is the
    vectorised form `load_batch("ev", ...)` builds (sim_vehicle_ecu.ev_batch), equal to it slot
    by slot (test/test_ev_batch.py)."""

    powertrain = "ev"
    batch = EvBatch


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
    if name == "fmu":
        from .fmu import Fmu  # an FMI 2.0 unit (roadmap 091 W4)
        return Fmu
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
    truth = getattr(component, "truth", ())
    unknown = [t for t in (truth if isinstance(truth, (tuple, list)) else [truth])
               if t not in TRUTH_INPUTS]
    if not isinstance(truth, (tuple, list)) or unknown:
        raise ComponentError(f"{name}: truth = {truth!r}: a tuple of {', '.join(TRUTH_INPUTS)}")
    driven = getattr(component, "driven", (True,) * 4)
    if not (isinstance(driven, tuple) and len(driven) == 4):
        raise ComponentError(f"{name}: driven must be four booleans (FL, FR, RL, RR)")
    return component


def load(name, vehicle, params=None, seed=0, base=None):
    """The component `name` for the car `vehicle` (a spec document), ready to step. `base`: the
    spec's directory, for a component that names files relative to it (`takes_base = True`: the
    shipped `fmu`)."""
    cls = resolve(name)
    try:
        if getattr(cls, "takes_base", False):
            component = cls(vehicle, dict(params or {}), int(seed), base=base)
        else:
            component = cls(vehicle, dict(params or {}), int(seed))
    except ComponentError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ComponentError(f"control component {name!r} refused its car or params: "
                             f"{type(e).__name__}: {e}") from e
    return check(component, name)


def for_spec(vehicle, overrides=None, seed=0, component=None, base=None):
    """The component a spec's car runs: its `powertrain.model` (`ideal`, `ev`: the shipped
    onboard computers, `overrides` their calibration overrides), or for `component` the entry
    point `powertrain.component` with `powertrain.params`. `component`: a run's
    `{entry_point, params}` in place of the spec's. `base`: the spec's directory (relative files
    in a component's params, as `fmu`'s unit)."""
    pt = vehicle.get("powertrain") or {}
    if component:
        return load(component["entry_point"], vehicle, component.get("params"), seed, base)
    if pt.get("model") == "component":
        return load(pt["component"], vehicle, pt.get("params"), seed, base)
    return load(pt.get("model") or "ideal", vehicle, overrides, seed, base)


class TruthFeed:
    """The declared ground truth as a component gets it: each input the truth `latency_s` ago
    (the newest sample at or before then; the first before that), plus Gaussian noise of
    `noise_std` from the run's seed, per element. `settings`: {input: {latency_s, noise_std}},
    only for declared inputs. Pure; deterministic for a seed."""

    def __init__(self, names, settings=None, seed=0):
        self.names = tuple(names)
        settings = dict(settings or {})
        extra = sorted(set(settings) - set(self.names))
        if extra:
            raise ComponentError(f"truth settings for {extra}, which the component does not "
                                 f"declare (it declares {list(self.names) or 'none'})")
        self.settings = {}
        for n in self.names:
            s = dict(settings.get(n) or {})
            bad = sorted(set(s) - set(TRUTH_SETTINGS))
            if bad:
                raise ComponentError(f"truth.{n}: unknown {bad}; "
                                     f"known: {', '.join(TRUTH_SETTINGS)}")
            lat, std = float(s.get("latency_s", 0.0)), float(s.get("noise_std", 0.0))
            if lat < 0 or std < 0:
                raise ComponentError(f"truth.{n}: latency_s and noise_std are >= 0")
            self.settings[n] = (lat, std)
        self.seed = seed
        self.reset()

    def reset(self):
        self.random = random.Random(self.seed)
        self.history = []  # (t, {name: value})

    def __call__(self, t, values):
        """The declared inputs at time `t` (s), given `values` ({name: value} of this step's
        truth, at least the declared names); None when nothing is declared."""
        if not self.names:
            return None
        self.history.append((t, {n: values[n] for n in self.names}))
        horizon = t - max(lat for lat, _ in self.settings.values())
        while len(self.history) > 1 and self.history[1][0] <= horizon + 1e-12:
            self.history.pop(0)
        out = {}
        for n in self.names:
            lat, std = self.settings[n]
            v = self.history[0][1][n]
            for ts, vals in self.history:
                if ts <= t - lat + 1e-12:
                    v = vals[n]
            if std > 0:
                v = (tuple(x + self.random.gauss(0.0, std) for x in v) if isinstance(v, tuple)
                     else v + self.random.gauss(0.0, std))
            out[n] = v
        return out


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


__all__ = ["COMMAND_TYPES", "COMMANDS", "ComponentError", "EcuComponent", "Ev", "GROUP", "Ideal", "SHIPPED",
           "TRUTH_INPUTS", "TRUTH_SETTINGS", "TruthFeed", "budget", "check", "checked",
           "for_spec", "load", "resolve", "timing"]
