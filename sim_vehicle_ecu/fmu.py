"""The `fmu` control component: an FMI 2.0 co-simulation unit stepped as a car's component (scensim
roadmap 091 W4; docs/design/roboracer-indoor-twin.md, 2.5).

A compiled model (C, C++, Simulink, Modelica, a vendor's binary) is exported as an FMI 2.0
co-simulation unit (a `.fmu`), and this one shipped component steps it. A spec names it as any
component:

    powertrain:
      model: component
      component: fmu
      params:
        fmu: models/racer.fmu            # relative to the spec's directory
        commands: vesc                   # ackermann | vesc | autoware
        inputs:                          # FMU input <- source
          speed_ref: command.value
          servo: command.servo
          wheel_rl: sensors.wheel_speeds[2]
          wheel_rr: sensors.wheel_speeds[3]
        outputs:                         # target <- FMU output
          drive[2]: torque_rl
          drive[3]: torque_rr
          tire_angle: steer
          telemetry.erpm: erpm
        parameters: {kp: 0.8}            # FMU parameter: value, set before initialisation
        seed: seed                       # optional: the FMU parameter that gets the run's seed
        truth: [speed]                   # optional: declared ground truth (TRUTH_INPUTS)
        driven: [false, false, true, true]   # optional; brake_floor too

Sources: `command.<field>` (the command type's fields; for `vesc`, `command.motor` is the mode's
index in MOTOR_MODES, -1 before the first), `sensors.wheel_speeds[i]`, `sensors.yaw_rate`,
`sensors.accel_x`, `sensors.accel_y`, `sensors.truth.<name>` or `[i]` (declared names only),
`dt`, `time`. A source with no value yet (no sensors before the first step after a reset, a
servo not yet commanded) holds the input's last value, at first its start value. Targets:
`drive[i]`, `brake[i]`, `tire_angle` (floats), `telemetry.<key>` (the variable's type: float, int
or bool); one not mapped is 0 (no telemetry: None).

Every name is checked against the unit's modelDescription.xml when the component is made: a
variable it lacks, or one of the wrong causality (inputs from `input`, outputs from `output`,
parameters and the seed from `parameter`) is a ComponentError naming it.

The unit: FMI 2.0, co-simulation, a Linux x86_64 binary (binaries/linux64/<id>.so). It is unpacked
once per content (its sha256, `fmu_sha256`) under SCENSIM_FMU_CACHE (default
~/.cache/scensim/fmu). Each step is one fmi2DoStep(time, dt); a unit that cannot take a variable
step size raises if dt changes. reset() is fmi2Reset, then the parameters, the seed and the
initialisation again.

A unit is native code: it runs in the host's process (the world server's), with its user's
rights, like any component. Pure otherwise: the standard library (ctypes, xml.etree, zipfile). No state outside the instance but the
loaded libraries (one per unit, shared) and their instance counts, so one process can step many
cars (and a batched host later, roadmap 092 S3), except a unit that says
canBeInstantiatedOnlyOncePerProcess: its second instance is refused.
"""
import ctypes
import hashlib
import math
import os
import platform
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from . import signals
from .component import COMMANDS, ComponentError, TRUTH_INPUTS

_OK, _WARNING, _DISCARD, _ERROR, _FATAL, _PENDING = range(6)
_STATUS = ("ok", "warning", "discard", "error", "fatal", "pending")
_TYPES = ("Real", "Integer", "Boolean", "Enumeration")  # String is not mapped
_COMMAND_FIELDS = {c: getattr(signals, t)._fields
                   for c, t in (("autoware", "Request"), ("ackermann", "AckermannCommand"),
                                ("vesc", "VescCommand"))}
_SENSORS = {"wheel_speeds": 4, "yaw_rate": None, "accel_x": None, "accel_y": None}
_TRUTH_SIZE = {"pose": 3, "velocity": 2, "wheel_slip": 4}
_INDEX = re.compile(r"^([a-z_]+)(?:\[(\d+)\])?$")

# fmi2CallbackLogger is variadic: a ctypes callback cannot read the varargs (FMPy formats them with
# a compiled helper; this module stays standard library), so a unit's message is kept as its
# format string, and errors quote it
_LOGGER = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                           ctypes.c_char_p)
_ALLOC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t)
_FREE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_STEP_FINISHED = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int)


class _Callbacks(ctypes.Structure):
    _fields_ = [("logger", _LOGGER), ("allocateMemory", _ALLOC), ("freeMemory", _FREE),
                ("stepFinished", _STEP_FINISHED), ("componentEnvironment", ctypes.c_void_p)]


_libc = ctypes.CDLL(None)
_libc.calloc.restype, _libc.calloc.argtypes = ctypes.c_void_p, (ctypes.c_size_t, ctypes.c_size_t)
_libc.free.argtypes = (ctypes.c_void_p,)
_calloc = _ALLOC(lambda n, size: _libc.calloc(n, size))
_free = _FREE(lambda p: _libc.free(p))
_LIBRARIES = {}   # .so path -> CDLL, shared by the instances of one unit
_INSTANCES = {}   # .so path -> live instances (canBeInstantiatedOnlyOncePerProcess)


class Variable:
    def __init__(self, el):
        self.name = el.get("name")
        self.vr = int(el.get("valueReference"))
        self.causality = el.get("causality", "local")
        self.type = next((c.tag for c in el if c.tag in _TYPES + ("String",)), None)
        start = next((c.get("start") for c in el if c.tag == self.type), None)
        self.start = _parse(self.type, start) if start is not None else None


def _parse(kind, text):
    if kind == "Real":
        return float(text)
    if kind == "Boolean":
        return text.strip().lower() in ("true", "1")
    if kind in ("Integer", "Enumeration"):
        return int(text)
    return text


class ModelDescription:
    """What a unit's modelDescription.xml says that the component needs."""

    def __init__(self, xml_text, where="modelDescription.xml"):
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise ComponentError(f"{where}: not XML ({e})") from None
        if root.get("fmiVersion") != "2.0":
            raise ComponentError(f"{where}: fmiVersion {root.get('fmiVersion')!r}; the fmu "
                                 "component takes FMI 2.0 units only")
        cs = root.find("CoSimulation")
        if cs is None:
            raise ComponentError(f"{where}: not a co-simulation unit (model exchange only); "
                                 "export it for co-simulation")
        self.guid = root.get("guid")
        self.identifier = cs.get("modelIdentifier")
        self.variable_step = cs.get("canHandleVariableCommunicationStepSize", "false") == "true"
        self.once = cs.get("canBeInstantiatedOnlyOncePerProcess", "false") == "true"
        self.variables = {}
        mv = root.find("ModelVariables")
        for el in (mv if mv is not None else ()):
            if el.tag == "ScalarVariable":
                v = Variable(el)
                self.variables[v.name] = v

    def variable(self, name, causalities, what):
        v = self.variables.get(name)
        if v is None:
            raise ComponentError(f"fmu: {what} {name!r} is not a variable of the unit (it has "
                                 f"{', '.join(sorted(self.variables)) or 'none'})")
        if v.causality not in causalities:
            raise ComponentError(f"fmu: {what} {name!r} has causality {v.causality!r}, not "
                                 f"{' or '.join(causalities)}")
        if v.type not in _TYPES:
            raise ComponentError(f"fmu: {what} {name!r} is a {v.type}; only Real, Integer, "
                                 "Boolean and Enumeration variables are mapped")
        return v


def cache_dir():
    return Path(os.environ.get("SCENSIM_FMU_CACHE") or Path.home() / ".cache" / "scensim" / "fmu")


def unpack(fmu_path):
    """The unit at `fmu_path` unpacked (once per content) and its sha256: (directory, sha256).

    Into a temporary directory beside the cache's, renamed into place when complete: two
    processes making one unit at once (two seats, a world server and a testbench check) never
    write a binary another may already have loaded. The one whose rename loses uses the
    winner's."""
    import shutil
    import tempfile

    fmu_path = Path(fmu_path)
    try:
        data = fmu_path.read_bytes()
    except OSError as e:
        raise ComponentError(f"fmu: cannot read {fmu_path}: {e}") from None
    sha = hashlib.sha256(data).hexdigest()
    out = cache_dir() / sha
    if (out / ".unpacked").exists():
        return out, sha
    cache_dir().mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{sha[:12]}-", dir=cache_dir()))
    try:
        try:
            with zipfile.ZipFile(fmu_path) as z:
                root = str(tmp.resolve())
                for name in z.namelist():
                    target = str((tmp / name).resolve())
                    if target != root and not target.startswith(root + os.sep):
                        raise ComponentError(f"fmu: {fmu_path} holds a path outside itself: {name}")
                z.extractall(tmp)
        except zipfile.BadZipFile:
            raise ComponentError(f"fmu: {fmu_path} is not an FMU (not a zip archive)") from None
        (tmp / ".unpacked").write_text(str(fmu_path) + "\n")
        try:
            os.rename(tmp, out)
            tmp = None
        except OSError as e:
            if not (out / ".unpacked").exists():
                raise ComponentError(f"fmu: cannot unpack {fmu_path} into {out}: {e}; a directory "
                                     "there without its .unpacked marker is an interrupted "
                                     "unpack: remove it") from None
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    return out, sha


def _library(path, md):
    if path not in _LIBRARIES:
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as e:
            raise ComponentError(f"fmu: its binary does not load: {e}") from None
        sig = {
            "fmi2Instantiate": (ctypes.c_void_p, [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                                                  ctypes.c_char_p, ctypes.POINTER(_Callbacks),
                                                  ctypes.c_int, ctypes.c_int]),
            "fmi2FreeInstance": (None, [ctypes.c_void_p]),
            "fmi2SetupExperiment": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.c_double,
                                                   ctypes.c_double, ctypes.c_int, ctypes.c_double]),
            "fmi2EnterInitializationMode": (ctypes.c_int, [ctypes.c_void_p]),
            "fmi2ExitInitializationMode": (ctypes.c_int, [ctypes.c_void_p]),
            "fmi2Terminate": (ctypes.c_int, [ctypes.c_void_p]),
            "fmi2Reset": (ctypes.c_int, [ctypes.c_void_p]),
            "fmi2DoStep": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_double, ctypes.c_double,
                                          ctypes.c_int]),
        }
        for kind, ct in (("Real", ctypes.c_double), ("Integer", ctypes.c_int),
                         ("Boolean", ctypes.c_int)):
            for op in ("Get", "Set"):
                sig[f"fmi2{op}{kind}"] = (ctypes.c_int, [ctypes.c_void_p,
                                                         ctypes.POINTER(ctypes.c_uint),
                                                         ctypes.c_size_t, ctypes.POINTER(ct)])
        for name, (res, args) in sig.items():
            try:
                f = getattr(lib, name)
            except AttributeError:
                raise ComponentError(f"fmu: its binary has no {name} (not an FMI 2.0 "
                                     "co-simulation binary)") from None
            f.restype, f.argtypes = res, args
        _LIBRARIES[path] = lib
    return _LIBRARIES[path]


def _source(text, commands, truth):
    """A source's (kind, name, index), checked: ("command", field, None), ("sensors", field, i),
    ("truth", name, i), ("dt"|"time", None, None)."""
    if text in ("dt", "time"):
        return (text, None, None)
    head, _, rest = text.partition(".")
    if head == "command":
        fields = _COMMAND_FIELDS[commands]
        if rest not in fields:
            raise ComponentError(f"fmu: source {text!r}: a {commands} command has "
                                 f"{', '.join(fields)}")
        return ("command", rest, None)
    if head == "sensors":
        if rest.startswith("truth."):
            m = _INDEX.match(rest[len("truth."):])
            name = m.group(1) if m else None
            if name not in TRUTH_INPUTS or name not in truth:
                raise ComponentError(f"fmu: source {text!r}: sensors.truth.<name> needs the name "
                                     f"in params.truth (declared: {list(truth) or 'none'})")
            size, index = _TRUTH_SIZE.get(name), m.group(2)
            if (size is None) != (index is None) or (index is not None and int(index) >= size):
                raise ComponentError(f"fmu: source {text!r}: truth {name} is "
                                     + (f"{size} values, index it [0..{size - 1}]" if size
                                        else "one value, not indexed"))
            return ("truth", name, None if index is None else int(index))
        m = _INDEX.match(rest)
        if m and m.group(1) in _SENSORS:
            size, index = _SENSORS[m.group(1)], m.group(2)
            if (size is None) == (index is None) and (index is None or int(index) < size):
                return ("sensors", m.group(1), None if index is None else int(index))
        raise ComponentError(f"fmu: source {text!r}: one of sensors.wheel_speeds[0..3], "
                             "sensors.yaw_rate, sensors.accel_x, sensors.accel_y, "
                             "sensors.truth.<name>")
    raise ComponentError(f"fmu: source {text!r}: one of command.<field>, sensors.<field>, dt, time")


def _target(text):
    m = _INDEX.match(text)
    if m and m.group(1) in ("drive", "brake") and m.group(2) is not None and int(m.group(2)) < 4:
        return (m.group(1), int(m.group(2)))
    if text == "tire_angle":
        return ("tire_angle", None)
    if text.startswith("telemetry.") and len(text) > len("telemetry."):
        return ("telemetry", text[len("telemetry."):])
    raise ComponentError(f"fmu: output target {text!r}: one of drive[0..3], brake[0..3], "
                         "tire_angle, telemetry.<key>")


PARAMS = ("fmu", "commands", "inputs", "outputs", "parameters", "seed", "truth", "driven",
          "brake_floor")


class Fmu:
    """An FMI 2.0 co-simulation unit as a control component (see the module's doc)."""

    takes_base = True  # component.load passes the spec's directory (`base`) for `params.fmu`

    def __init__(self, vehicle, params, seed=0, base=None):
        params = dict(params or {})
        unknown = sorted(set(params) - set(PARAMS))
        if unknown:
            raise ComponentError(f"fmu: unknown params {unknown}; known: {', '.join(PARAMS)}")
        if "fmu" not in params:
            raise ComponentError("fmu: params.fmu names the unit (a .fmu, relative to the spec)")
        path = Path(os.path.expanduser(str(params["fmu"])))
        if not path.is_absolute():
            if base is None:
                raise ComponentError(f"fmu: params.fmu {str(path)!r} is relative and its host gave "
                                     "no spec directory; give an absolute path")
            path = Path(base) / path
        self.commands = params.get("commands")
        if self.commands not in COMMANDS:
            raise ComponentError(f"fmu: params.commands {self.commands!r}: one of "
                                 f"{', '.join(COMMANDS)}")
        truth = params.get("truth") or ()
        if isinstance(truth, str) or not all(t in TRUTH_INPUTS for t in truth):
            raise ComponentError(f"fmu: params.truth {truth!r}: a list of {', '.join(TRUTH_INPUTS)}")
        self.truth = tuple(truth)
        if "driven" in params:
            self.driven = tuple(bool(d) for d in params["driven"])
        if "brake_floor" in params:
            self.brake_floor = float(params["brake_floor"])
        self.seed = int(seed)
        self.path = path
        directory, self.fmu_sha256 = unpack(path)
        md_path = directory / "modelDescription.xml"
        try:
            self.md = ModelDescription(md_path.read_text(), f"{path}: modelDescription.xml")
        except OSError:
            raise ComponentError(f"fmu: {path} has no modelDescription.xml") from None
        if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
            raise ComponentError(f"fmu: runs linux64 binaries only, not {platform.system()} "
                                 f"{platform.machine()}")
        so = directory / "binaries" / "linux64" / f"{self.md.identifier}.so"
        if not so.exists():
            raise ComponentError(f"fmu: {path} has no binaries/linux64/{self.md.identifier}.so "
                                 "(export it for Linux x86_64)")
        self._so = str(so)
        self._resources = (directory / "resources").resolve().as_uri()
        md = self.md
        self.inputs = [(md.variable(n, ("input",), "input"), _source(str(s), self.commands,
                                                                    self.truth))
                       for n, s in dict(params.get("inputs") or {}).items()]
        self.outputs = [(_target(str(t)), md.variable(n, ("output",), "output"))
                        for t, n in dict(params.get("outputs") or {}).items()]
        self.parameters = [(md.variable(n, ("parameter",), "parameter"), v)
                           for n, v in dict(params.get("parameters") or {}).items()]
        self.seed_var = (md.variable(params["seed"], ("parameter",), "seed parameter")
                         if params.get("seed") else None)
        if self.seed_var is not None and self.seed_var.type not in ("Integer", "Real"):
            raise ComponentError(f"fmu: seed parameter {self.seed_var.name!r} is a "
                                 f"{self.seed_var.type}, not an Integer or Real")
        self._log = []
        self._callbacks = _Callbacks(_LOGGER(self._logger), _calloc, _free, _STEP_FINISHED(0),
                                     None)
        self._lib = _library(self._so, md)
        if md.once and _INSTANCES.get(self._so, 0) >= 1:
            raise ComponentError(f"fmu: {path} says canBeInstantiatedOnlyOncePerProcess, and an "
                                 "instance of it already runs in this process")
        self._c = self._lib.fmi2Instantiate(
            md.identifier.encode(), 1, (md.guid or "").encode(), self._resources.encode(),
            ctypes.byref(self._callbacks), 0, 1)  # logging on: many units log nothing otherwise
        if not self._c:
            raise ComponentError(f"fmu: {path}: fmi2Instantiate failed{self._log_tail()}")
        _INSTANCES[self._so] = _INSTANCES.get(self._so, 0) + 1
        self._initialize()

    # --- the unit's life ------------------------------------------------------------------------

    def _logger(self, env, name, status, category, message):
        if status >= _WARNING:
            self._log.append(f"{_STATUS[status] if status < 6 else status}: "
                             f"{(message or b'').decode(errors='replace')}")
            del self._log[:-20]

    def _log_tail(self):
        return f" ({'; '.join(self._log[-3:])})" if self._log else ""

    def _check(self, status, what):
        if status not in (_OK, _WARNING):
            raise ComponentError(f"fmu: {self.path}: {what} returned "
                                 f"{_STATUS[status] if 0 <= status < 6 else status}"
                                 f"{self._log_tail()}")

    def _set(self, var, value):
        vr = (ctypes.c_uint * 1)(var.vr)
        if var.type == "Real":
            self._check(self._lib.fmi2SetReal(self._c, vr, 1, (ctypes.c_double * 1)(float(value))),
                        f"fmi2SetReal({var.name})")
        elif var.type == "Boolean":
            self._check(self._lib.fmi2SetBoolean(self._c, vr, 1, (ctypes.c_int * 1)(int(bool(value)))),
                        f"fmi2SetBoolean({var.name})")
        else:
            self._check(self._lib.fmi2SetInteger(self._c, vr, 1, (ctypes.c_int * 1)(int(value))),
                        f"fmi2SetInteger({var.name})")

    def _get(self, var):
        """The variable's value: a float for Real, an int for Integer and Enumeration, a bool
        for Boolean (actuation targets take it as a float; telemetry keeps the type)."""
        vr = (ctypes.c_uint * 1)(var.vr)
        if var.type == "Real":
            out = (ctypes.c_double * 1)()
            self._check(self._lib.fmi2GetReal(self._c, vr, 1, out), f"fmi2GetReal({var.name})")
        elif var.type == "Boolean":
            out = (ctypes.c_int * 1)()
            self._check(self._lib.fmi2GetBoolean(self._c, vr, 1, out), f"fmi2GetBoolean({var.name})")
            return bool(out[0])
        else:
            out = (ctypes.c_int * 1)()
            self._check(self._lib.fmi2GetInteger(self._c, vr, 1, out), f"fmi2GetInteger({var.name})")
            return int(out[0])
        return float(out[0])

    def _initialize(self):
        self._check(self._lib.fmi2SetupExperiment(self._c, 0, 0.0, 0.0, 0, 0.0),
                    "fmi2SetupExperiment")
        for var, value in self.parameters:
            self._set(var, value)
        if self.seed_var is not None:
            self._set(self.seed_var, self.seed)
        self._check(self._lib.fmi2EnterInitializationMode(self._c), "fmi2EnterInitializationMode")
        self._check(self._lib.fmi2ExitInitializationMode(self._c), "fmi2ExitInitializationMode")
        self.time = 0.0
        self._dt = None
        self._held = {var.vr: var.start for var, _ in self.inputs}

    def reset(self):
        self._check(self._lib.fmi2Reset(self._c), "fmi2Reset")
        self._initialize()

    def close(self):
        """End the unit's instance (fmi2Terminate, fmi2FreeInstance); idempotent."""
        c, self._c = getattr(self, "_c", None), None
        if c:
            self._lib.fmi2Terminate(c)
            self._lib.fmi2FreeInstance(c)
            _INSTANCES[self._so] -= 1

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - at interpreter exit
            pass

    # --- one physics step -----------------------------------------------------------------------

    def _value(self, src, command, sensors, dt):
        kind, name, index = src
        if kind == "dt":
            return dt
        if kind == "time":
            return self.time
        if kind == "command":
            if command is None:
                return None
            v = getattr(command, name)
            if name == "motor":   # vesc: the mode's index, -1 before the first motor command
                return -1 if v is None else signals.MOTOR_MODES.index(v)
            return v
        if sensors is None:
            return None
        v = getattr(sensors, name) if kind == "sensors" else (sensors.truth or {}).get(name)
        if v is None:
            return None
        return v[index] if index is not None else v

    def step(self, command, sensors, dt):
        if not (isinstance(dt, (int, float)) and math.isfinite(dt) and dt > 0):
            raise ComponentError(f"fmu: step dt {dt!r} is not a positive number")
        if self._dt is None:
            self._dt = dt
        elif not self.md.variable_step and abs(dt - self._dt) > 1e-12 * max(1.0, self._dt):
            raise ComponentError(f"fmu: {self.path} cannot take a variable step size "
                                 f"(canHandleVariableCommunicationStepSize false): dt {dt} after "
                                 f"{self._dt}; run it at one physics rate")
        for var, src in self.inputs:
            v = self._value(src, command, sensors, dt)
            if v is None:
                v = self._held.get(var.vr)
                if v is None:
                    continue
            self._held[var.vr] = v
            self._set(var, v)
        self._check(self._lib.fmi2DoStep(self._c, self.time, dt, 1), f"fmi2DoStep(t={self.time})")
        self.time += dt
        drive, brake, angle, telemetry = [0.0] * 4, [0.0] * 4, 0.0, {}
        for (kind, where), var in self.outputs:
            v = self._get(var)
            if kind == "drive":
                drive[where] = float(v)
            elif kind == "brake":
                brake[where] = float(v)
            elif kind == "tire_angle":
                angle = float(v)
            else:
                telemetry[where] = v
        return signals.Actuation(tuple(drive), tuple(brake), angle,
                                 telemetry=telemetry or None)
