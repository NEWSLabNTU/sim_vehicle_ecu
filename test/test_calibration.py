"""The calibration: from a spec's document, overrides by name, the defaults the shipped car's.
Pure."""
import copy
import unittest

from sim_vehicle_ecu.calibration import Calibration

# The physics sections of sim_vehicle_spec's vehicles/tesla_model3.yaml (spec version 2), as
# a loaded spec's document holds them.
MODEL3 = {
    "geometry": {"wheel_base": 3.005, "wheel_tread": 1.667, "wheel_radius": 0.37,
                 "wheel_width": 0.235, "max_steer_angle": 0.7},
    "inertia": {"mass": 1760.0, "com_front_fraction": 0.47, "com_height": 0.46},
    "wheels": {"mass": 20.0, "damping_rate": 0.25},
    "tires": {"friction": 1.0},
    "brakes": {"max_brake_torque": 3000.0, "hold_brake_torque": 1500.0,
               "distribution": "dynamic_axle_load", "dead_time": 0.08, "time_constant": 0.06,
               "abs": {"enabled": True, "slip_release": 0.25, "slip_reapply": 0.15,
                       "release_rate": 8.0, "apply_rate": 4.0, "min_speed": 1.5}},
    "powertrain": {"model": "ev", "max_drive_torque": 1500.0, "drive_axle": "rear",
                   "gear_ratio": 10.0, "motor_peak_torque": 340.0, "motor_peak_power": 194000.0,
                   "motor_max_rpm": 15200.0, "motor_time_constant": 0.04,
                   "regen_max_torque": 130.0, "regen_max_power": 60000.0,
                   "regen_fade_speeds": [1.0, 3.0],
                   "traction_control": {"enabled": True, "slip_limit": 0.25, "slip_reapply": 0.15,
                                        "release_rate": 20.0, "apply_rate": 5.0}},
    "resistance": {"rolling_coefficient": 0.0095, "drag_coefficient": 0.23, "frontal_area": 2.2,
                   "air_density": 1.2},
    "steering": {"ackermann": True, "time_constant": 0.0, "rate_limit": 0.0},
}


class FromSpecTest(unittest.TestCase):
    def test_the_shipped_car_is_the_defaults(self):
        c = Calibration.from_spec(MODEL3)
        base = Calibration()
        self.assertEqual(c.com, (0.47 * 3.005, 0.0, 0.46))
        self.assertEqual(c, base)

    def test_what_the_spec_says_is_read(self):
        doc = copy.deepcopy(MODEL3)
        doc["brakes"]["abs"]["slip_release"] = 0.3
        doc["powertrain"]["regen_fade_speeds"] = [2, 4]
        doc["powertrain"]["traction_control"]["enabled"] = False
        doc["steering"]["time_constant"] = 0.1
        c = Calibration.from_spec(doc)
        self.assertEqual(c.abs_slip_release, 0.3)
        self.assertEqual((c.regen_fade_low, c.regen_fade_high), (2.0, 4.0))
        self.assertIsInstance(c.regen_fade_low, float)
        self.assertFalse(c.tc_enabled)
        self.assertEqual(c.steer_time_constant, 0.1)

    def test_esc_is_off_unless_the_spec_has_it(self):
        self.assertFalse(Calibration.from_spec(MODEL3).esc_enabled)
        doc = copy.deepcopy(MODEL3)
        doc["brakes"]["esc"] = {"enabled": True, "gain": 9000}
        c = Calibration.from_spec(doc)
        self.assertTrue(c.esc_enabled)
        self.assertEqual(c.esc_gain, 9000.0)
        self.assertEqual(c.esc_sideslip_limit, Calibration().esc_sideslip_limit)

    def test_no_resistance_section_is_no_road_load(self):
        doc = copy.deepcopy(MODEL3)
        del doc["resistance"]
        c = Calibration.from_spec(doc)
        self.assertEqual((c.rolling_coefficient, c.drag_coefficient, c.frontal_area),
                         (0.0, 0.0, 0.0))

    def test_overrides_win_and_unknown_ones_are_refused(self):
        c = Calibration.from_spec(MODEL3, {"esc_enabled": True, "com": [1.4, 0.0, 0.5]})
        self.assertTrue(c.esc_enabled)
        self.assertEqual(c.com, (1.4, 0.0, 0.5))
        with self.assertRaises(ValueError) as e:
            Calibration.from_spec(MODEL3, {"esc_enable": True})
        self.assertIn("esc_enable", str(e.exception))

    def test_round_trip(self):
        c = Calibration(esc_enabled=True)
        self.assertEqual(Calibration.from_dict(c.to_dict()), c)


if __name__ == "__main__":
    unittest.main()
