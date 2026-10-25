"""The brake module's estimates from its own sensors: reference speed, slip, sideslip,
friction. Pure."""
import math
import unittest

from sim_vehicle_ecu.calibration import G, Calibration
from sim_vehicle_ecu.estimator import Activity, Estimator
from sim_vehicle_ecu.signals import Sensors

DT = 1 / 60
REAR = (False, False, True, True)


def sensors(c, road, yaw_rate=0.0, ax=0.0, ay=0.0):
    """Wheel speed sensors for road speeds `road` (m/s per wheel, or one for all)."""
    if not isinstance(road, (tuple, list)):
        road = (road,) * 4
    return Sensors(tuple(u / c.wheel_radius for u in road), yaw_rate, ax, ay)


class SpeedTest(unittest.TestCase):
    def setUp(self):
        self.c = Calibration()
        self.e = Estimator(self.c, REAR)

    def run_for(self, seconds, make, activity=Activity()):
        est = None
        for n in range(round(seconds / DT)):
            est = self.e.step(make(n * DT), DT, 0.0, activity)
        return est

    def test_rolling_freely_it_is_the_wheels(self):
        est = self.run_for(0.5, lambda t: sensors(self.c, 15.0))
        self.assertAlmostEqual(est.speed, 15.0, delta=0.01)
        for w in est.wheels:
            self.assertAlmostEqual(w.slip, 0.0, delta=0.001)

    def test_standing_is_zero(self):
        est = self.run_for(0.5, lambda t: sensors(self.c, 0.0, ax=0.02))
        self.assertEqual(est.speed, 0.0)

    def test_under_drive_the_undriven_wheels(self):
        # the rear wheels spin 50 % past the road; the fronts roll at it
        est = self.run_for(0.5, lambda t: sensors(self.c, (10.0, 10.0, 15.0, 15.0)),
                           Activity(driving=True))
        self.assertAlmostEqual(est.speed, 10.0, delta=0.01)
        self.assertAlmostEqual(est.wheels[2].slip, 5.0 / (10.0 + self.c.est_slip_offset),
                               delta=0.01)
        self.assertAlmostEqual(est.wheels[0].slip, 0.0, delta=0.01)

    def test_braking_with_every_wheel_slipping_follows_the_deceleration(self):
        # ABS holds all four wheels 20 % under the road while the car slows at 7 m/s^2: the
        # wheels alone would say 20 % too little, the acceleration carries the estimate
        self.run_for(0.5, lambda t: sensors(self.c, 20.0))
        start = self.e.speed
        est = self.run_for(1.0, lambda t: sensors(self.c, 0.8 * (start - 7.0 * t), ax=-7.0),
                           Activity(braking=True, abs_active=True))
        truth = start - 7.0 * 1.0  # the acceleration over 60 steps
        self.assertAlmostEqual(est.speed, truth, delta=0.4)
        for w in est.wheels:
            self.assertAlmostEqual(w.slip, -0.2 * truth / (truth + self.c.est_slip_offset),
                                   delta=0.05)

    def test_slip_is_over_the_speed_plus_physx_offset(self):
        est = self.run_for(0.5, lambda t: sensors(self.c, (1.0, 1.0, 3.0, 3.0)),
                           Activity(driving=True))
        self.assertAlmostEqual(est.speed, 1.0, delta=0.01)
        self.assertAlmostEqual(est.wheels[2].slip, 2.0 / (1.0 + self.c.est_slip_offset),
                               delta=0.01)

    def test_reverse(self):
        est = self.run_for(0.5, lambda t: sensors(self.c, -3.0))
        self.assertAlmostEqual(est.speed, -3.0, delta=0.01)


class SideslipTest(unittest.TestCase):
    def test_the_kinematic_integral(self):
        c = Calibration(est_sideslip_washout=0.0)
        e = Estimator(c, REAR)
        e.step(sensors(c, 10.0), DT, 0.0, Activity())
        # at 10 m/s, no yaw, the velocity turning left at 0.1 rad/s: a_y = v beta'
        for _ in range(round(0.5 / DT)):
            est = e.step(sensors(c, 10.0, ay=1.0), DT, 0.0, Activity())
        self.assertAlmostEqual(est.sideslip, 0.05, delta=0.002)

    def test_a_steady_turn_has_none(self):
        c = Calibration()
        e = Estimator(c, REAR)
        e.step(sensors(c, 10.0), DT, 0.0, Activity())
        for _ in range(round(2.0 / DT)):
            est = e.step(sensors(c, 10.0, yaw_rate=0.5, ay=5.0), DT, 0.1, Activity())
        self.assertAlmostEqual(est.sideslip, 0.0, delta=1e-6)

    def test_the_washout_forgets_a_bias(self):
        c = Calibration()
        e = Estimator(c, REAR)
        for _ in range(round(20.0 / DT)):
            est = e.step(sensors(c, 10.0, ay=0.05), DT, 0.0, Activity())
        # 0.005 rad/s of bias settles at bias x washout, not growing
        self.assertAlmostEqual(est.sideslip, 0.005 * c.est_sideslip_washout, delta=0.001)


class FrictionTest(unittest.TestCase):
    def test_at_the_limit_it_is_what_the_car_reaches_and_then_it_recovers(self):
        c = Calibration()
        e = Estimator(c, REAR)
        self.assertEqual(e.friction, 1.0)
        for _ in range(round(1.0 / DT)):
            est = e.step(sensors(c, 10.0, ax=-0.23 * G), DT, 0.0,
                         Activity(braking=True, abs_active=True))
        self.assertAlmostEqual(est.friction, 0.23 + c.est_mu_margin, delta=0.01)
        low = est.friction
        for _ in range(round(2.0 / DT)):
            est = e.step(sensors(c, 10.0), DT, 0.0, Activity())
        self.assertAlmostEqual(est.friction, low + 2.0 * c.est_mu_recover_rate, delta=0.01)

    def test_away_from_the_limit_never_below_what_the_car_reaches(self):
        c = Calibration()
        e = Estimator(c, REAR)
        e.friction = 0.3
        est = e.step(sensors(c, 10.0, ay=0.6 * G, yaw_rate=0.6 * G / 10.0), DT, 0.06, Activity())
        self.assertAlmostEqual(est.friction, 0.6, delta=0.01)

    def test_traction_control_alone_sees_the_driven_axle(self):
        c = Calibration()
        e = Estimator(c, REAR)
        for _ in range(round(1.0 / DT)):
            est = e.step(sensors(c, (5.0, 5.0, 7.0, 7.0), ax=2.0), DT, 0.0,
                         Activity(driving=True, tc_active=True))
        rear = 1 - (c.com[0] - c.com[2] * 2.0 / G) / c.wheel_base
        self.assertAlmostEqual(est.friction, 2.0 / G / rear + c.est_mu_margin, delta=0.01)

    def test_turning_less_than_asked_is_the_limit(self):
        c = Calibration()
        e = Estimator(c, REAR)
        d = math.radians(5.0)  # asks 10 tan(5 deg) / 3.005 = 0.29 rad/s at 10 m/s
        for _ in range(round(1.0 / DT)):
            est = e.step(sensors(c, 10.0, yaw_rate=0.15, ay=1.5), DT, d, Activity())
        self.assertTrue(e.at_limit)
        self.assertAlmostEqual(est.friction, 1.5 / G + c.est_mu_margin, delta=0.01)


if __name__ == "__main__":
    unittest.main()
