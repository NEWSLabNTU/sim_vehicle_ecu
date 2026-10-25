"""The onboard computers as one unit: what goes in, what comes out, and that it is its parts
in order. Pure."""
import math
import unittest

from sim_vehicle_ecu import Calibration, Ecu, Request, Sensors
from sim_vehicle_ecu import powertrain as pt
from sim_vehicle_ecu.esc import Esc
from sim_vehicle_ecu.estimator import Estimate

DT = 1 / 60


class Given:
    """A test estimator: returns the estimate it is given."""

    def __init__(self):
        self.estimate = Estimate(0.0, None, 0.0, 1.0)

    def reset(self):
        pass

    def step(self, sensors, dt, tire_angle, activity):
        return self.estimate


def rolling(c, v):
    return Sensors((v / c.wheel_radius,) * 4)


class EcuTest(unittest.TestCase):
    def test_before_any_reading_a_standing_car_holds(self):
        c = Calibration()
        ecu = Ecu(c)
        act = ecu.step(Request(), None, DT)
        self.assertEqual(act.drive, (0.0,) * 4)
        act = None
        for _ in range(60):
            act = ecu.step(Request(), None, DT)
        self.assertAlmostEqual(act.brake[0], c.hold_brake_torque, delta=1.0)

    def test_it_drives_on_its_sensors(self):
        c = Calibration()
        ecu = Ecu(c)
        act = None
        for _ in range(60):
            act = ecu.step(Request(acceleration=1.0, velocity=20.0), rolling(c, 10.0), DT)
        self.assertAlmostEqual(ecu.estimator.speed, 10.0, delta=0.01)
        self.assertAlmostEqual(sum(act.drive), pt.feedforward(c, 1.0, 10.0) * c.wheel_radius,
                               delta=2.0)

    def test_it_is_the_steering_then_the_esc_then_the_powertrain(self):
        c = Calibration(esc_enabled=True, steer_rate_limit=1.0)
        given = Given()
        ecu = Ecu(c, estimator=given)
        ref_pt, ref_esc = pt.make(c), Esc(c)
        wheels = tuple(pt.Wheel(20.0 / c.wheel_radius, s) for s in (0.0, 0.0, -0.3, 0.0))
        given.estimate = Estimate(20.0, wheels, -0.12, 0.4)
        req = Request(acceleration=-2.0, velocity=10.0, tire_angle=0.05)
        angle = 0.0
        for _ in range(40):
            act = ecu.step(req, Sensors(yaw_rate=0.6), DT)
            angle = min(0.05, angle + 1.0 * DT)
            esc = ref_esc.step(20.0, 0.6, -0.12, angle, 0.4, DT)
            want = ref_pt.torques(req, 20.0, DT, wheels, esc)
            self.assertAlmostEqual(act.tire_angle, angle)
            self.assertEqual((act.drive, act.brake), (want.drive, want.brake))
        self.assertGreater(act.brake[1], act.brake[0])  # the ESC's outer front

    def test_the_steering_actuator_holds_the_mechanical_stop(self):
        c = Calibration()
        act = Ecu(c).step(Request(tire_angle=2.0), None, DT)
        self.assertEqual(act.tire_angle, c.max_steer_angle)

    def test_reset_forgets(self):
        c = Calibration()
        ecu = Ecu(c)
        for _ in range(60):
            ecu.step(Request(acceleration=-3.0, tire_angle=0.2), rolling(c, 10.0), DT)
        ecu.reset()
        self.assertEqual((ecu.tire_angle, ecu.estimator.speed, ecu.powertrain.pressure),
                         (0.0, 0.0, [0.0, 0.0]))
        self.assertIsNone(ecu.last)

    def test_a_stop_on_low_grip_from_the_sensors_alone(self):
        # A quarter-car per wheel on mud-like grip (mu 0.25), braked hard: from the wheel speeds
        # and the deceleration alone, ABS keeps every wheel from locking and the car stops
        # near mu g.
        c = Calibration(regen_max_torque=0.0)
        ecu = Ecu(c)
        mu, r, v = 0.25, c.wheel_radius, 15.0
        J = 0.5 * c.wheel_mass * r * r
        fz = c.mass * 9.81 / 4
        omega = [v / r] * 4
        ax, t, worst = 0.0, 0.0, 0.0
        while v > 2.0 and t < 15.0:
            act = ecu.step(Request(acceleration=-8.0), Sensors(tuple(omega), 0.0, ax, 0.0), DT)
            force = 0.0
            for i in range(4):
                for _ in range(8):
                    s = (omega[i] * r - v) / max(v, 4.0)
                    tire = max(-1.0, min(1.0, s * 4612.5 * 9.81 / fz)) * fz * mu
                    omega[i] = max(0.0, omega[i] + (-tire * r - act.brake[i]) / J * (DT / 8))
                force += tire
                if t > 0.5:  # past the first release, as a real ABS's first cycle
                    worst = min(worst, (omega[i] * r - v) / max(v, 4.0))
            ax = force / c.mass
            v += ax * DT
            t += DT
        self.assertLess(t, 15.0)
        self.assertGreater(worst, -0.6)  # no wheel locked after the first cycle
        decel = (15.0 - 2.0) / t
        self.assertGreater(decel, 0.75 * mu * 9.81)
        self.assertLess(ecu.estimator.friction, 0.45)
        self.assertTrue(math.isfinite(ecu.estimator.speed))


if __name__ == "__main__":
    unittest.main()
