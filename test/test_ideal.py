"""The `ideal` powertrain (acb's contract), the steering actuator and the calibration's
brake split. Pure."""
import math
import unittest

from sim_vehicle_ecu import powertrain
from sim_vehicle_ecu.actuators import steer
from sim_vehicle_ecu.calibration import G, Calibration
from sim_vehicle_ecu.signals import GEAR_NEUTRAL, GEAR_PARK, GEAR_REVERSE, Request


class BrakeSplitTest(unittest.TestCase):
    def test_braking_moves_load_and_brake_force_forward(self):
        c = Calibration()
        static = c.brake_split(0.0)
        self.assertAlmostEqual(static, c.com[0] / c.wheel_base)
        self.assertGreater(c.brake_split(8.0), static + 0.1)
        self.assertAlmostEqual(Calibration(brake_distribution="static").brake_split(8.0), static)

    def test_a_hard_stop_can_reach_the_tire_limit(self):
        c = Calibration()
        need = c.mass * G * c.tire_friction * c.wheel_radius
        self.assertLess(need * c.brake_split(G) / 2, c.max_brake_torque)


class IdealPowertrainTest(unittest.TestCase):
    def setUp(self):
        # No road load or wheel damping: the force is m a alone
        self.c = Calibration(powertrain="ideal", rolling_coefficient=0.0,
                               drag_coefficient=0.0, wheel_damping_rate=0.0)
        self.pt = powertrain.make(self.c)
        self.mass = self.c.mass + 2 * self.c.wheel_mass  # four wheels at 0.5 m r^2 / r^2

    def test_it_is_ideal(self):
        self.assertEqual(self.pt.name, "ideal")

    def test_standing_still_holds(self):
        drive, brakes = self.pt.torques(Request(), 0.0, 0.01)
        self.assertEqual(drive, (0.0,) * 4)
        self.assertEqual(brakes, (self.c.hold_brake_torque,) * 4)
        _, brakes = self.pt.torques(Request(acceleration=1.0, velocity=5.0, gear=GEAR_PARK), 3.0, 0.01)
        self.assertEqual(brakes, (self.c.hold_brake_torque,) * 4)

    def test_acceleration_is_m_a_over_four_wheels(self):
        drive, brakes = self.pt.torques(Request(acceleration=1.0, velocity=5.0), 2.0, 0.01)
        for d in drive:
            self.assertAlmostEqual(d, self.mass * 1.0 * self.c.wheel_radius / 4)
        self.assertEqual(brakes, (0.0,) * 4)

    def test_drive_is_capped_and_reverse_drives_backwards(self):
        drive, _ = self.pt.torques(Request(acceleration=50.0, velocity=5.0), 2.0, 0.01)
        self.assertEqual(drive, (self.c.max_drive_torque,) * 4)
        drive, _ = self.pt.torques(Request(acceleration=1.0, velocity=-2.0, gear=GEAR_REVERSE), -1.0, 0.01)
        self.assertLess(drive[0], 0.0)

    def test_deceleration_brakes_split_by_dynamic_load(self):
        drive, (fl, fr, rl, rr) = self.pt.torques(Request(acceleration=-3.0, velocity=5.0), 7.4, 0.01)
        self.assertEqual(drive, (0.0,) * 4)
        self.assertEqual((fl, rl), (fr, rr))
        total = self.mass * 3.0 * self.c.wheel_radius
        self.assertAlmostEqual(2 * (fl + rl), total)
        self.assertAlmostEqual(fl / (fl + rl), self.c.brake_split(3.0))
        self.assertGreater(fl, rl)

    def test_brakes_are_capped(self):
        _, brakes = self.pt.torques(Request(acceleration=-100.0, velocity=5.0), 10.0, 0.01)
        self.assertEqual(brakes[0], self.c.max_brake_torque)

    def test_neutral_coasts(self):
        self.assertEqual(self.pt.torques(Request(acceleration=1.0, velocity=3.0, gear=GEAR_NEUTRAL), 3.0, 0.01),
                         ((0.0,) * 4, (0.0,) * 4))

    def test_the_road_load_is_in_the_feedforward(self):
        # The physics applies the road's load to the body, and the feedforward expects it:
        # ideal pushes against it, so a cruise at a = 0 still needs drive.
        c = Calibration(powertrain="ideal", wheel_damping_rate=0.0)
        drive, _ = powertrain.make(c).torques(Request(acceleration=0.0, velocity=20.0), 20.0, 0.01)
        load = -powertrain.road_load(c, 20.0)
        self.assertAlmostEqual(load, c.rolling_coefficient * c.mass * G
                               + 0.5 * 1.2 * 0.23 * 2.2 * 400.0)
        self.assertAlmostEqual(sum(drive), load * c.wheel_radius)

    def test_reserved_models_fail_loudly(self):
        for name in ("physx", "fmu"):
            with self.assertRaises(NotImplementedError):
                powertrain.make(Calibration(powertrain=name))
        with self.assertRaises(ValueError):
            powertrain.make(Calibration(powertrain="steam"))


class SteeringTest(unittest.TestCase):
    def test_clamped_to_the_limit(self):
        self.assertEqual(steer(0.0, 2.0, 0.01, 0.7), 0.7)
        self.assertEqual(steer(0.0, -2.0, 0.01, 0.7), -0.7)

    def test_lag_and_rate_limit(self):
        a = steer(0.0, 0.5, 0.1, 0.7, time_constant=0.1)
        self.assertAlmostEqual(a, 0.5 * (1 - math.exp(-1)))
        self.assertAlmostEqual(steer(0.0, 0.5, 0.1, 0.7, rate_limit=1.0), 0.1)


if __name__ == "__main__":
    unittest.main()
