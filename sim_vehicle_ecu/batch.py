"""The batched form of a control component: N cars (slots) stepped at once on arrays (scensim
roadmap 092 S3; docs/design/roboracer-indoor-twin.md in scensim, section 2.6).

    batch = load_batch("ev", vehicle, params, seeds=[1, 2, 3])
    batch.reset_batch(np.ones(3, bool))
    act = batch.step_batch(commands, sensors, dt)   # dicts of arrays, slot-major

The arrays are the scalar NamedTuples' fields, struct of arrays, slot first:

- commands, by `commands` type: `autoware` {acceleration, velocity, tire_angle, gear}, `ackermann`
  {speed, acceleration, jerk, steering_angle, steering_angle_velocity, age}, `vesc` {motor (int:
  index into MOTOR_MODES, -1 none), value, servo (NaN: none), age}; each [N], with `valid` [N]
  (False where the scalar form gets None: `autoware` before its first command).
- sensors: {wheel_speeds [N, 4], yaw_rate, accel_x, accel_y [N], valid [N] (False where the
  scalar form gets None: the first step after a reset), truth {name: [N] or [N, k]}, declared
  names only}.
- actuation: {drive [N, 4], brake [N, 4], tire_angle [N], telemetry {key: [N]} (optional)}.
- `dt` is one float: one world step for every slot. Slot i's seed is `seeds[i]`, the seed a
  single component for that car would get.

A component without a batched form runs as `LoopBatch`: N instances, each stepped by `step`.
It is correct and slow, and says so (`vectorised = False`, `loop_of`); `require_vectorised`
refuses it by name for an executor that needs throughput. A component with one declares
`batch = <class>` (constructed as `cls(vehicle, params, seeds)`); `ScalarOf` makes its scalar
form from it, so the two cannot drift.

Bit for bit: a batched form equals N scalar steps exactly only when both compute with + - * /,
sqrt, min/max/clip, where and comparisons. numpy's exp, log, tanh and atan2 may use their own
vector kernels, which can differ from Python's `math` in the last bit; a batched form that
needs one should say so.

numpy is imported here only: the scalar package stays the standard library.
"""
import math

import numpy as np

from . import component as comp
from . import signals

COMMAND_FIELDS = {
    "autoware": ("acceleration", "velocity", "tire_angle", "gear"),
    "ackermann": ("speed", "acceleration", "jerk", "steering_angle", "steering_angle_velocity",
                  "age"),
    "vesc": ("motor", "value", "servo", "age"),
}
SENSOR_FIELDS = ("wheel_speeds", "yaw_rate", "accel_x", "accel_y")
ACTUATION_FIELDS = ("drive", "brake", "tire_angle")
DTYPE = np.float64
# each declared truth input's width (component.TRUTH_INPUTS); None: one value per slot
TRUTH_WIDTH = {"pose": 3, "yaw": None, "velocity": 2, "speed": None, "yaw_rate": None,
               "wheel_slip": 4, "sideslip": None}


class BatchError(comp.ComponentError):
    """A batch that cannot be built, or arrays of the wrong shape; the message names the field."""


# --- conversions: scalar NamedTuples <-> arrays ---------------------------------------------

def commands_to_batch(kind, commands):
    """A list of scalar commands (None allowed for `autoware`) as the batch's dict."""
    n = len(commands)
    out = {f: np.zeros(n, dtype=np.int64 if f in ("gear", "motor") else DTYPE)
           for f in COMMAND_FIELDS[kind]}
    out["valid"] = np.array([c is not None for c in commands], dtype=bool)
    for i, c in enumerate(commands):
        if c is None:
            continue
        for f in COMMAND_FIELDS[kind]:
            v = getattr(c, f)
            if kind == "vesc" and f == "motor":
                v = -1 if v is None else signals.MOTOR_MODES.index(v)
            elif kind == "vesc" and f == "servo":
                v = math.nan if v is None else v
            out[f][i] = v
    return out


def batch_to_commands(kind, arrays):
    """The batch's commands as N scalar ones (None where `valid` is False)."""
    cls = getattr(signals, comp.COMMAND_TYPES[kind])
    valid = arrays.get("valid")
    n = len(next(iter(arrays[f] for f in COMMAND_FIELDS[kind])))
    out = []
    for i in range(n):
        if valid is not None and not valid[i]:
            out.append(None)
            continue
        kw = {}
        for f in COMMAND_FIELDS[kind]:
            v = arrays[f][i]
            if kind == "vesc" and f == "motor":
                v = None if int(v) < 0 else signals.MOTOR_MODES[int(v)]
            elif kind == "vesc" and f == "servo":
                v = None if math.isnan(v) else float(v)
            elif f in ("gear",):
                v = int(v)
            else:
                v = float(v)
            kw[f] = v
        out.append(cls(**kw))
    return out


def sensors_to_batch(sensors, truth_names=()):
    n = len(sensors)
    out = {"wheel_speeds": np.zeros((n, 4), DTYPE), "yaw_rate": np.zeros(n, DTYPE),
           "accel_x": np.zeros(n, DTYPE), "accel_y": np.zeros(n, DTYPE),
           "valid": np.array([s is not None for s in sensors], dtype=bool)}
    # every declared input has its array, valid slots or not (a slot without a reading: zeros)
    truth = {name: np.zeros((n, TRUTH_WIDTH[name]) if TRUTH_WIDTH[name] else n, DTYPE)
             for name in truth_names}
    for i, s in enumerate(sensors):
        if s is None:
            continue
        out["wheel_speeds"][i] = s.wheel_speeds
        out["yaw_rate"][i], out["accel_x"][i], out["accel_y"][i] = s.yaw_rate, s.accel_x, s.accel_y
        for name in truth_names:
            truth[name][i] = (s.truth or {})[name]
    if truth_names:
        out["truth"] = truth
    return out


def batch_to_sensors(arrays, truth_names=()):
    out = []
    for i in range(len(arrays["yaw_rate"])):
        if not arrays["valid"][i]:
            out.append(None)
            continue
        truth = None
        if truth_names:
            truth = {}
            for name in truth_names:
                v = arrays["truth"][name][i]
                truth[name] = tuple(float(x) for x in v) if np.ndim(v) else float(v)
        out.append(signals.Sensors(tuple(float(x) for x in arrays["wheel_speeds"][i]),
                                   float(arrays["yaw_rate"][i]), float(arrays["accel_x"][i]),
                                   float(arrays["accel_y"][i]), truth))
    return out


def actuations_to_batch(acts):
    out = {"drive": np.array([a.drive for a in acts], DTYPE).reshape(len(acts), 4),
           "brake": np.array([a.brake for a in acts], DTYPE).reshape(len(acts), 4),
           "tire_angle": np.array([a.tire_angle for a in acts], DTYPE)}
    tel = [a.telemetry for a in acts]
    if any(t is not None for t in tel):
        keys = [tuple(sorted(t)) if t is not None else None for t in tel]
        if len(set(keys)) != 1:
            raise BatchError("telemetry: the slots report different keys "
                             f"({sorted({k for k in keys if k}, key=str)})")
        out["telemetry"] = {k: np.array([t[k] for t in tel]) for k in keys[0]}
    return out


def batch_to_actuations(arrays):
    tel = arrays.get("telemetry")
    out = []
    for i in range(len(arrays["tire_angle"])):
        t = None if tel is None else {k: v[i].item() for k, v in tel.items()}
        out.append(signals.Actuation(tuple(float(x) for x in arrays["drive"][i]),
                                     tuple(float(x) for x in arrays["brake"][i]),
                                     float(arrays["tire_angle"][i]), t))
    return out


# --- checks ---------------------------------------------------------------------------------

def checked_batch(act, n, name="component"):
    """The actuation arrays, or BatchError naming the field: shape, finiteness, brake >= 0."""
    if not isinstance(act, dict):
        raise BatchError(f"{name}: step_batch() returned {type(act).__name__}, not a dict of "
                         f"arrays ({', '.join(ACTUATION_FIELDS)})")
    want = {"drive": (n, 4), "brake": (n, 4), "tire_angle": (n,)}
    for f, shape in want.items():
        a = act.get(f)
        if not isinstance(a, np.ndarray) or a.shape != shape:
            raise BatchError(f"{name}: actuation {f}: shape "
                             f"{getattr(a, 'shape', type(a).__name__)}, not {shape}")
        if not np.all(np.isfinite(a)):
            bad = np.argwhere(~np.isfinite(a))[0][0]
            raise BatchError(f"{name}: actuation {f}: not finite in slot {bad}")
    if np.any(act["brake"] < 0):
        bad = int(np.argwhere(act["brake"] < 0)[0][0])
        raise BatchError(f"{name}: actuation brake: negative in slot {bad}")
    for k, v in (act.get("telemetry") or {}).items():
        if np.shape(v)[:1] != (n,):
            raise BatchError(f"{name}: telemetry {k}: {np.shape(v)[:1]} slots, not {n}")
    return act


def require_vectorised(batch, name=None):
    """Refuse a batch that loops scalar steps, for an executor that needs throughput."""
    if not getattr(batch, "vectorised", False):
        raise BatchError(f"{name or getattr(batch, 'loop_of', 'component')}: has no batched "
                         "form (it runs as LoopBatch, one step per slot); fix: give the "
                         "component a `batch` class (sim_vehicle_ecu.batch), or run it on "
                         "seats")
    return batch


# --- the forms ------------------------------------------------------------------------------

class LoopBatch:
    """N scalar components as one batch: correct, one Python step per slot, flagged."""

    vectorised = False
    array_api = "numpy"
    dtype = DTYPE

    def __init__(self, components, name="component"):
        if not components:
            raise BatchError(f"{name}: a batch of no slots")
        self.components = list(components)
        first = self.components[0]
        self.commands = first.commands
        self.truth = tuple(getattr(first, "truth", ()))
        self.driven = getattr(first, "driven", (True,) * 4)
        self.n, self.loop_of = len(self.components), name

    def reset_batch(self, mask):
        for c, m in zip(self.components, _mask(mask, self.n)):
            if m:
                c.reset()

    def step_batch(self, commands, sensors, dt):
        cmds = batch_to_commands(self.commands, commands)
        sens = batch_to_sensors(sensors, self.truth)
        acts = [comp.checked(c.step(cmd, s, dt), f"{self.loop_of} slot {i}")
                for i, (c, cmd, s) in enumerate(zip(self.components, cmds, sens))]
        return actuations_to_batch(acts)


class ScalarOf:
    """The scalar component of a batched one: `step` is `step_batch` at N = 1, so the two forms
    are one code. A component author writes the batch class and declares
    `class MyCar(ScalarOf): batch = MyCarBatch`."""

    batch = None

    def __init__(self, vehicle, params=None, seed=0):
        if self.batch is None:
            raise BatchError(f"{type(self).__name__}: ScalarOf needs `batch = <class>`")
        self._b = self.batch(vehicle, dict(params or {}), [int(seed)])
        self.commands = self._b.commands
        self.truth = tuple(getattr(self._b, "truth", ()))
        self.driven = getattr(self._b, "driven", (True,) * 4)

    def reset(self):
        self._b.reset_batch(np.ones(1, bool))

    def step(self, command, sensors, dt):
        act = self._b.step_batch(commands_to_batch(self.commands, [command]),
                                 sensors_to_batch([sensors], self.truth), dt)
        return batch_to_actuations(act)[0]


def _mask(mask, n):
    m = np.asarray(mask, dtype=bool)
    if m.shape != (n,):
        raise BatchError(f"reset_batch: a mask of shape {m.shape}, not ({n},)")
    return m


def load_batch(name, vehicle, params=None, seeds=(0,), base=None):
    """N slots of the component `name`, one per seed: its `batch` class when it has one
    (vectorised), else a LoopBatch of N instances (`fmu`: N instances of the one unit, its
    canBeInstantiatedOnlyOncePerProcess refused as for two cars)."""
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise BatchError(f"{name}: a batch of no slots")
    given = isinstance(name, type)   # a class (a test's, a host's own) rather than a name
    cls = name if given else comp.resolve(name)
    name = cls.__name__ if given else name
    bcls = getattr(cls, "batch", None)
    if bcls is not None:
        try:
            b = bcls(vehicle, dict(params or {}), seeds)
        except comp.ComponentError:
            raise
        except Exception as e:  # noqa: BLE001
            raise BatchError(f"control component {name!r}'s batch refused its car or params: "
                             f"{type(e).__name__}: {e}") from e
        b.n = len(seeds)
        b.vectorised = getattr(b, "vectorised", True)
        return check_batch(b, name)
    if given:
        return LoopBatch([comp.check(cls(vehicle, dict(params or {}), s), name) for s in seeds],
                         name)
    return LoopBatch([comp.load(name, vehicle, params, s, base) for s in seeds], name)


def check_batch(b, name="component"):
    """Refuse an object that is not a batch (BatchError); returns it."""
    if getattr(b, "commands", None) not in comp.COMMANDS:
        raise BatchError(f"{name}: commands = {getattr(b, 'commands', None)!r}, not one of "
                         f"{', '.join(comp.COMMANDS)}")
    for m in ("reset_batch", "step_batch"):
        if not callable(getattr(b, m, None)):
            raise BatchError(f"{name}: no {m}()")
    truth = getattr(b, "truth", ())
    if not isinstance(truth, (tuple, list)) or any(t not in comp.TRUTH_INPUTS for t in truth):
        raise BatchError(f"{name}: truth = {truth!r}: a tuple of {', '.join(comp.TRUTH_INPUTS)}")
    return b


__all__ = ["ACTUATION_FIELDS", "BatchError", "COMMAND_FIELDS", "LoopBatch", "SENSOR_FIELDS",
           "ScalarOf", "actuations_to_batch", "batch_to_actuations", "batch_to_commands",
           "batch_to_sensors", "checked_batch", "commands_to_batch", "load_batch",
           "require_vectorised", "sensors_to_batch"]
