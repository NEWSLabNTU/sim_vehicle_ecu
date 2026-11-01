"""The regen slip control (`ev`): a rear-drive car with no ABS braking on ice through its motor
must not lock its rear, finds the rear's grip and holds regen within a share of it, and on a
grippy road changes nothing. A straight-line car on its own sensors: the body, a spooled rear
axle, free front wheels, and a tyre that peaks at a slip and falls off past it (PhysX's slip,
over |v| + 4 m/s). Pure."""
import unittest

from sim_vehicle_ecu import Calibration, Ecu, Request, Sensors
from sim_vehicle_ecu.calibration import G

DT = 1 / 60
SUB = 8


def drift_car(**over):
    """drift_mustang's numbers (sim_vehicle_spec): rear motor 150 N m regen through 6:1 on
    0.32 m wheels (2812 N at the tyres), a static 50:50 split, a spool, no ABS or TC."""
    c = dict(wheel_base=2.886, wheel_radius=0.32, mass=1450.0, com=(1.443, 0.0, 0.48),
             wheel_mass=22.0, gear_ratio=6.0, motor_peak_torque=700.0, motor_peak_power=330000.0,
             motor_max_rpm=16000.0, motor_time_constant=0.02, regen_max_torque=150.0,
             max_brake_torque=2500.0, brake_distribution="static", brake_dead_time=0.05,
             brake_time_constant=0.04, abs_enabled=False, tc_enabled=False, diff_type="locked",
             rolling_coefficient=0.0, drag_coefficient=0.0)
    c.update(over)
    return Calibration(**c)


def tyre(slip, mu, fz, peak, sliding=0.7):
    """Longitudinal force: linear to the peak slip, then falling to `sliding` of the peak at
    20 times it."""
    s = abs(slip)
    if s <= peak:
        f = s / peak
    else:
        f = max(sliding, 1.0 - (1.0 - sliding) * (s - peak) / (19 * peak))
    return (1 if slip > 0 else -1) * f * mu * fz


def drive(c, accel, seconds, mu, peak, v=13.0, ecu=None, sliding=0.7, spin=1.0, ramp=0.0):
    """Brake straight at the request `accel` for `seconds`: (ecu, the rear wheels' worst
    braking slip after the first 0.4 s, the speed, a log of (regen, rear friction) per step).
    `sliding`: the tyre's share past its peak; `spin`: the rear wheels' start speed over the
    road's (above 1: coming out of a wheelspin); `ramp`: seconds the request takes to reach
    `accel` (a planner's request: 0, a step)."""
    ecu = ecu or Ecu(c)
    r, J = c.wheel_radius, 0.5 * c.wheel_mass * c.wheel_radius ** 2
    w = [v / r] * 2 + [spin * v / r] * 2
    ax, t, worst, log = 0.0, 0.0, 0.0, []
    while t < seconds and v > 3.0:
        req = accel * min(1.0, (t + DT) / ramp) if ramp else accel
        act = ecu.step(Request(acceleration=req, velocity=0.0), Sensors(tuple(w), 0.0, ax, 0.0),
                       DT)
        rear_fz = c.mass * G * (c.com[0] - c.com[2] * -ax / G) / c.wheel_base
        loads = (0.5 * (c.mass * G - rear_fz),) * 2 + (0.5 * rear_fz,) * 2
        for _ in range(SUB):
            h = DT / SUB
            forces = [tyre((w[i] * r - v) / (v + 4.0), mu, loads[i], peak, sliding)
                      for i in range(4)]
            for i in range(4):
                torque = act.drive[i] - forces[i] * r - act.brake[i] * (1 if w[i] > 0 else -1)
                w[i] += torque / J * h
                if act.brake[i] and abs(w[i]) < 1e-3 and act.drive[i] == 0.0:
                    w[i] = 0.0
            spool = 0.5 * (w[2] + w[3])
            w[2] = w[3] = spool
            ax = sum(forces) / c.mass
            v += ax * h
        t += DT
        if t > 0.4:
            worst = max(worst, (v - w[2] * r) / (v + 4.0))
        d = ecu.debug
        log.append((d["regen_force"], d["pressure"][1] * 2 / r))
    return ecu, worst, v, log


class RegenSlipControlTest(unittest.TestCase):
    ICE = (0.265, 0.05)  # swb's ice: the peak friction and its slip

    def test_without_it_regen_locks_the_rear_on_ice(self):
        _, worst, _, _ = drive(drift_car(regen_slip_control=False), -1.8, 2.0, *self.ICE)
        self.assertGreater(worst, 0.5)  # issue 009: the rear turns far slower than the road

    def test_on_ice_the_rear_keeps_rolling_and_its_grip_is_found(self):
        c = drift_car()
        ecu, worst, _, log = drive(c, -1.8, 2.0, *self.ICE)
        pt = ecu.powertrain
        self.assertLess(worst, 0.05)  # rolling, within ice's peak slip
        self.assertIsNotNone(pt.axle_mu)
        self.assertAlmostEqual(pt.axle_mu, 0.265, delta=0.05)
        # regen held within its share of the rear's grip, the rest the friction brakes'
        rear_fz = c.mass * G * (c.com[0] - c.com[2] * 1.8 / G) / c.wheel_base
        regen, _ = log[-1]
        self.assertLess(regen, 1.05 * max(c.regen_grip_share * pt.axle_mu * rear_fz,
                                          0.5 * 1.8 * c.mass))

    def test_held_regen_brakes_the_front_by_the_distribution(self):
        # once the grip is known a mild stop is no longer all regen on the rear: the front's
        # friction brakes take their share of what regen gives up
        c = drift_car()
        ecu, _, _, _ = drive(c, -1.8, 1.0, *self.ICE)
        ecu, worst, _, _ = drive(c, -1.2, 1.0, *self.ICE, ecu=ecu)
        self.assertLess(worst, 0.05)
        self.assertGreater(ecu.powertrain.pressure[0], 50.0)

    def test_on_a_grippy_road_nothing_changes(self):
        # asphalt: regen never slips the rear, so the control never acts, bit for bit
        for accel in (-1.0, -2.0, -6.0):
            a, _, va, la = drive(drift_car(), accel, 1.5, 1.12, 0.25)
            b, _, vb, lb = drive(drift_car(regen_slip_control=False), accel, 1.5, 1.12, 0.25)
            self.assertEqual(la, lb)
            self.assertEqual(va, vb)
            self.assertIsNone(a.powertrain.axle_mu)
            self.assertEqual(a.powertrain.rsc.factor, 1.0)

    def test_a_slip_whose_axle_carries_its_brakes_is_not_a_lost_grip(self):
        # issue 011: wet mud's tyre peaks late and keeps its force past the peak, so a rear
        # braking through regen alone runs at 0.3 slip carrying all of it; the old 0.25
        # release cut it there and held regen within 0.7 of what it carried for the drive
        a, worst, _, la = drive(drift_car(), -1.4, 2.5, 0.35, 0.35, sliding=1.0, ramp=1.5)
        b, _, _, lb = drive(drift_car(regen_slip_control=False), -1.4, 2.5, 0.35, 0.35,
                            sliding=1.0, ramp=1.5)
        self.assertGreater(worst, 0.25)
        self.assertEqual(la, lb)
        self.assertIsNone(a.powertrain.axle_mu)
        self.assertEqual(a.powertrain.rsc.factor, 1.0)

    def test_a_lockup_on_a_road_that_keeps_its_force_is_released(self):
        # past the peak with no fall-off the tyre carries its grip, but regen beyond it still
        # locks the wheel slowly (no runaway, the axle 15 % short): the slip still growing
        c = drift_car()
        _, worst_off, _, _ = drive(drift_car(regen_slip_control=False), -1.8, 1.5, 0.29, 0.06,
                                   sliding=1.0)
        ecu, worst, _, _ = drive(c, -1.8, 1.5, 0.29, 0.06, sliding=1.0)
        self.assertGreater(worst_off, 0.5)
        self.assertLess(worst, 0.25)
        self.assertAlmostEqual(ecu.powertrain.axle_mu, 0.29, delta=0.05)

    def test_a_wheel_down_from_a_wheelspin_is_not_running_away(self):
        # issue 011: out of a power slide into regen, the rear crosses the road's speed and
        # slows 4-7 g faster than a car its tyre still pushed a moment ago; the car's
        # deceleration lags, the deficit reads past 30 %, and the grip was found at the floor
        for spin in (1.3, 1.6):
            a, _, _, la = drive(drift_car(), -1.2, 1.0, 0.3, 0.2, sliding=1.0, spin=spin)
            b, _, _, lb = drive(drift_car(regen_slip_control=False), -1.2, 1.0, 0.3, 0.2,
                                sliding=1.0, spin=spin)
            self.assertEqual(la, lb)
            self.assertIsNone(a.powertrain.axle_mu)

    def test_on_ice_out_of_a_wheelspin_the_grip_is_still_found(self):
        c = drift_car()
        ecu, worst, _, _ = drive(c, -1.8, 2.0, *self.ICE, spin=1.5)
        self.assertLess(worst, 0.05)
        self.assertAlmostEqual(ecu.powertrain.axle_mu, 0.265, delta=0.05)

    def test_regen_within_the_axles_own_share_is_not_cut(self):
        # a hard stop on ice: the rear's friction share alone is past its grip (no ABS), and
        # cutting regen below that share would only hand it to the same axle's friction brakes
        c = drift_car()
        ecu, _, _, log = drive(c, -2.4, 1.0, *self.ICE)
        brake = 2.4 * (c.mass + 2 * c.wheel_mass)
        regen, _ = log[-1]
        self.assertGreaterEqual(regen, 0.5 * brake - 60.0)

    def test_the_grip_found_is_kept_and_slowly_forgotten(self):
        c = drift_car()
        ecu, _, _, _ = drive(c, -1.8, 1.0, *self.ICE)
        pt = ecu.powertrain
        mu = pt.axle_mu
        pt.regen_capped = False
        for _ in range(60):  # coasting a second: no regen asked
            pt.regen_slip(1.0, 10.0, None, 0.0, False, DT)
        self.assertEqual(pt.axle_mu, mu)  # no wheels, no readings: kept
        ecu.reset()
        self.assertIsNone(ecu.powertrain.axle_mu)


class CalibrationTest(unittest.TestCase):
    def test_on_by_default_and_read_from_a_spec(self):
        import copy
        from test_calibration import MODEL3
        c = Calibration.from_spec(MODEL3)
        self.assertEqual((c.regen_slip_control, c.regen_grip_share), (True, 0.7))
        doc = copy.deepcopy(MODEL3)
        doc["powertrain"]["regen_slip_control"] = {"enabled": False, "grip_share": 0.5,
                                                  "runaway_decel": 20.0}
        c = Calibration.from_spec(doc)
        self.assertEqual((c.regen_slip_control, c.regen_grip_share, c.regen_runaway_decel),
                         (False, 0.5, 20.0))


if __name__ == "__main__":
    unittest.main()
