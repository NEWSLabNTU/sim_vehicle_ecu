"""The brake floor: a releasing brake below it is released (PhysX's sticky-tire hold). It is the
car's: a 5 kg RC car's whole braking is a fraction of the road car's 50 N m."""
import unittest

from sim_vehicle_ecu import powertrain as pt
from sim_vehicle_ecu.actuators import BRAKE_FLOOR
from sim_vehicle_ecu.calibration import Calibration
from sim_vehicle_ecu.signals import GEAR_DRIVE, Request

DT = 1 / 180
# AutoSDV car A's twin: 5 kg, 52 mm wheels, 1 N m brakes, a motor whose regen is tiny
SMALL = dict(powertrain="ev", mass=5.0, wheel_radius=0.052, wheel_mass=0.4, max_brake_torque=1.0,
             hold_brake_torque=0.5, gear_ratio=8.0, motor_peak_torque=0.15, motor_peak_power=60.0,
             motor_max_rpm=3000.0, regen_max_torque=0.03, regen_max_power=20.0, abs_enabled=False,
             brake_dead_time=0.05, brake_time_constant=0.05, wheel_damping_rate=0.002)


def brake_after(c, decel, speed=1.5, seconds=0.5):
    """The last step's per-wheel friction torques and the powertrain's debug state."""
    m = pt.make(c)
    out = None
    for _ in range(round(seconds / DT)):
        w = [pt.Wheel(speed / c.wheel_radius, 0.0)] * 4
        out = m.torques(Request(-decel, 0.0, 0.0, GEAR_DRIVE), speed, DT, w)
    return out.brake, m.debug


class BrakeFloor(unittest.TestCase):
    def test_the_default_is_the_road_cars(self):
        self.assertEqual(Calibration().brake_floor, BRAKE_FLOOR)

    def test_a_small_car_with_the_road_floor_gets_no_friction_brakes(self):
        # what the twin did: ~0.1 N m per wheel asked, all of it under 50 N m, so none applied
        brake, dbg = brake_after(Calibration(**SMALL), 2.5)
        self.assertGreater(max(dbg["friction_request"]), 0.05)
        self.assertEqual(max(brake), 0.0)

    def test_a_small_car_with_its_own_floor_gets_the_friction_it_asks(self):
        brake, dbg = brake_after(Calibration(**SMALL, brake_floor=0.005), 2.5)
        front, rear = dbg["friction_request"]
        self.assertGreater(front, 0.05)
        for got, want in zip(brake, (front, front, rear, rear)):
            self.assertAlmostEqual(got, want, delta=0.01 * want)


if __name__ == "__main__":
    unittest.main()
