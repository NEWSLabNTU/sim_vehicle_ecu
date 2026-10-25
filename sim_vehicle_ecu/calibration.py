"""The onboard computers' calibration: what they know about their car. Pure.

`Calibration` is flat data, read from a vehicle spec (`from_spec`, sim_vehicle_spec's format,
version 2), then overridden by name for a run (`from_dict`'s `overrides`). The field names
are the ones aib's rig used before roadmap 055 (`abs_*`, `tc_*`, `esc_*`, ...), so a value
reads the same in both places.

Defaults are the shipped spec's (sim_vehicle_spec's vehicles/tesla_model3.yaml), so a bare
`Calibration()` is the same car; the spec, not these defaults, is the source of truth. The
estimator's fields (`est_*`) and the ESC's are not in the spec today: the defaults hold, and a
spec's `brakes.esc` is read when there is one.
"""
from dataclasses import asdict, dataclass, fields

G = 9.81


@dataclass
class Calibration:
    # The car as its controllers know it: base_link is the rear axle centre on the ground
    wheel_base: float = 3.005
    wheel_tread: float = 1.667
    wheel_radius: float = 0.370
    max_steer_angle: float = 0.70  # bicycle tire angle, rad
    mass: float = 1760.0
    com: tuple = (0.47 * 3.005, 0.0, 0.46)  # base_link
    wheel_mass: float = 20.0
    wheel_damping_rate: float = 0.25  # the driveline's drag per wheel, as the feedforward knows it
    tire_friction: float = 1.0  # the design road's friction (dry asphalt)
    # The drive unit: `ideal` realizes the request as m a + F_resist(v); `ev` is the
    # electric drive unit (sim_vehicle_ecu.powertrain)
    powertrain: str = "ev"
    max_drive_torque: float = 1500.0  # per wheel, N m (ideal)
    drive_axle: str = "rear"  # rear | front | all
    front_drive_share: float = 0.0  # the front axle's share of motor torque when "all"
    gear_ratio: float = 10.0  # motor to wheel, single speed
    motor_peak_torque: float = 340.0  # N m at the motor shaft
    motor_peak_power: float = 194000.0  # W
    motor_max_rpm: float = 15200.0
    motor_time_constant: float = 0.04  # s, the inverter/motor torque lag
    regen_max_torque: float = 130.0  # N m at the motor shaft
    regen_max_power: float = 60000.0  # W
    regen_fade_low: float = 1.0  # m/s: no regen at or below
    regen_fade_high: float = 3.0  # m/s: full regen at or above
    # The friction brakes
    max_brake_torque: float = 3000.0  # per wheel, N m
    hold_brake_torque: float = 1500.0  # per wheel, N m
    brake_distribution: str = "dynamic_axle_load"
    brake_dead_time: float = 0.08  # s, the hydraulics' dead time
    brake_time_constant: float = 0.06  # s, then a first-order pressure lag
    # ABS: per wheel, on longitudinal slip against the direction of travel
    abs_enabled: bool = True
    abs_slip_release: float = 0.25
    abs_slip_reapply: float = 0.15
    abs_release_rate: float = 8.0  # 1/s, of the brake pressure factor
    abs_apply_rate: float = 4.0  # 1/s
    abs_min_speed: float = 1.5  # m/s
    # Traction control: driven wheels' slip cuts motor torque
    tc_enabled: bool = True
    tc_slip_limit: float = 0.25
    tc_slip_reapply: float = 0.15
    tc_release_rate: float = 20.0  # 1/s, of the motor torque factor
    tc_apply_rate: float = 5.0  # 1/s
    # Stability control (sim_vehicle_ecu.esc): one wheel's brake against a yaw rate the
    # tires cannot hold
    esc_enabled: bool = False
    esc_min_speed: float = 3.0  # m/s: off below
    esc_understeer: float = 0.0  # s^2/m^2, the target's understeer: r = v tan(d) / (L (1 + K v^2))
    esc_target_margin: float = 0.85  # the target capped at this share of mu g / v
    esc_dead_band: float = 0.035  # rad/s
    esc_dead_band_share: float = 0.1  # of the target yaw rate, added to the dead band
    esc_gain: float = 12000.0  # N m of yaw moment per rad/s of yaw rate error
    esc_sideslip_limit: float = 0.07  # rad
    esc_sideslip_gain: float = 30000.0  # N m per rad of sideslip past the limit
    esc_max_brake_torque: float = 2000.0  # N m on the one wheel
    esc_motor_cut_rate: float = 5.0  # 1/s, of the motor torque factor, against oversteer
    esc_motor_restore_rate: float = 1.0  # 1/s
    # The road load the feedforward expects (the car's model of it, not the road's)
    rolling_coefficient: float = 0.0095  # Crr
    drag_coefficient: float = 0.23  # Cd
    frontal_area: float = 2.2  # m^2
    air_density: float = 1.2  # kg/m^3
    # The power steering's actuator
    steer_time_constant: float = 0.0  # s; 0: none
    steer_rate_limit: float = 0.0  # rad/s; 0: none
    # The brake module's estimates (sim_vehicle_ecu.estimator)
    est_speed_gain: float = 15.0  # 1/s: the reference speed's pull toward the wheels
    est_speed_gain_slipping: float = 1.5  # 1/s: the same while ABS or TC works a wheel
    est_speed_gain_bound: float = 0.1  # 1/s: under ABS, down toward the fastest wheel
    est_slip_offset: float = 4.0  # m/s: slip is over |v| + this, as PhysX's (its min denominator)
    est_sideslip_washout: float = 2.0  # s: the sideslip integral's leak toward 0
    est_mu_margin: float = 0.05  # added to the acceleration reached at the limit
    est_mu_floor: float = 0.1
    est_mu_recover_rate: float = 0.1  # 1/s, toward tire_friction away from the limit

    @classmethod
    def from_dict(cls, values, overrides=None):
        """`values`, then `overrides`; unknown names are refused."""
        known = {f.name for f in fields(cls)}
        merged = {**(values or {}), **(overrides or {})}
        unknown = set(merged) - known
        if unknown:
            raise ValueError(f"unknown calibration field(s) {sorted(unknown)}; "
                             f"known: {sorted(known)}")
        return cls(**{k: _tuple(v) for k, v in merged.items()})

    @classmethod
    def from_spec(cls, doc, overrides=None):
        """The calibration of a vehicle spec's document (sim_vehicle_spec: a loaded
        `Spec.doc`, validated), then `overrides` by field name."""
        return cls.from_dict(spec_values(doc), overrides)

    def to_dict(self):
        return {k: _list(v) for k, v in asdict(self).items()}

    def brake_split(self, decel):
        """The front axle's share of a braking force at `decel` (m/s^2). With
        `dynamic_axle_load`, the dynamic front load's, so all four tires reach their limit
        together (ideal distribution / EBD); with `static`, the static load's."""
        a, h = self.com[0], self.com[2]
        shift = h * max(decel, 0.0) / G if self.brake_distribution == "dynamic_axle_load" else 0.0
        front = (a + shift) / self.wheel_base
        return min(max(front, 0.05), 0.95)


def spec_values(doc):
    """{field: value} of a spec's document: what the spec says, the defaults for the rest."""
    g, i, w, t = doc["geometry"], doc["inertia"], doc["wheels"], doc["tires"]
    b, p = doc["brakes"], doc["powertrain"]
    rs = doc.get("resistance")
    abs_ = b.get("abs") or {}
    esc = b.get("esc") or {}
    tc = p.get("traction_control") or {}
    st = doc.get("steering") or {}
    base = Calibration()
    values = dict(
        wheel_base=g["wheel_base"], wheel_tread=g["wheel_tread"],
        wheel_radius=g["wheel_radius"], max_steer_angle=g["max_steer_angle"],
        mass=i["mass"],
        com=(i["com_front_fraction"] * g["wheel_base"], i.get("com_lateral", 0.0), i["com_height"]),
        wheel_mass=w["mass"], wheel_damping_rate=w.get("damping_rate", base.wheel_damping_rate),
        tire_friction=t["friction"],
        powertrain=p["model"], max_drive_torque=p["max_drive_torque"],
        max_brake_torque=b["max_brake_torque"],
        hold_brake_torque=b.get("hold_brake_torque", base.hold_brake_torque),
        brake_distribution=b.get("distribution", base.brake_distribution),
        brake_dead_time=b.get("dead_time", base.brake_dead_time),
        brake_time_constant=b.get("time_constant", base.brake_time_constant),
        steer_time_constant=st.get("time_constant", 0.0),
        steer_rate_limit=st.get("rate_limit", 0.0),
    )
    for key, field in (("enabled", "abs_enabled"), ("slip_release", "abs_slip_release"),
                       ("slip_reapply", "abs_slip_reapply"), ("release_rate", "abs_release_rate"),
                       ("apply_rate", "abs_apply_rate"), ("min_speed", "abs_min_speed")):
        values[field] = abs_.get(key, getattr(base, field))
    # The stability control: a spec's brakes.esc, by the field's name without its esc_
    # prefix; off when the spec has none.
    for f in fields(Calibration):
        if f.name.startswith("esc_"):
            values[f.name] = esc.get(f.name[4:], getattr(base, f.name))
    for key in ("drive_axle", "front_drive_share", "gear_ratio", "motor_peak_torque",
                "motor_peak_power", "motor_max_rpm", "motor_time_constant", "regen_max_torque",
                "regen_max_power"):
        values[key] = p.get(key, getattr(base, key))
    fade = p.get("regen_fade_speeds", (base.regen_fade_low, base.regen_fade_high))
    values["regen_fade_low"], values["regen_fade_high"] = fade
    for key, field in (("enabled", "tc_enabled"), ("slip_limit", "tc_slip_limit"),
                       ("slip_reapply", "tc_slip_reapply"), ("release_rate", "tc_release_rate"),
                       ("apply_rate", "tc_apply_rate")):
        values[field] = tc.get(key, getattr(base, field))
    # No `resistance` section: the car has no road load, and its feedforward expects none.
    for key in ("rolling_coefficient", "drag_coefficient", "frontal_area"):
        values[key] = (rs or {}).get(key, getattr(base, key) if rs is not None else 0.0)
    values["air_density"] = (rs or {}).get("air_density", base.air_density)
    out = {}
    for k, v in values.items():
        default = getattr(base, k)
        if isinstance(default, float) and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        out[k] = _tuple(v)
    return out


def _tuple(v):
    if isinstance(v, list):
        return tuple(_tuple(x) for x in v)
    return v


def _list(v):
    if isinstance(v, tuple):
        return [_list(x) for x in v]
    return v
