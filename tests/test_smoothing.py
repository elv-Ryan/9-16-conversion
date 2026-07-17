import unittest

from vertical_focus.smoothing import SmoothCamera


class SmoothCameraTests(unittest.TestCase):
    def make_camera(self):
        return SmoothCamera(
            crop_width=0.31640625,
            response_time_s=0.34,
            deadband=0.007,
            max_speed_per_s=0.30,
            max_accel_per_s2=1.20,
            fast_response_time_s=0.12,
            fast_max_speed_per_s=0.78,
            fast_max_accel_per_s2=3.40,
            fast_error_threshold=0.055,
            fast_boost_s=0.34,
        )

    def test_normal_speed_and_crop_bounds(self):
        camera = self.make_camera()
        values = [camera.reset(0.25)]
        dt = 1.0 / 30.0
        for _ in range(180):
            values.append(camera.update(0.50, dt))
        speeds = [abs(b - a) / dt for a, b in zip(values, values[1:])]
        self.assertLessEqual(max(speeds), camera.fast_max_speed + 1e-6)
        half = 0.31640625 / 2.0
        self.assertTrue(all(half <= value <= 1.0 - half for value in values))
        self.assertGreater(values[-1], values[0])

    def test_deadband_stops_residual_drift_immediately(self):
        camera = self.make_camera()
        camera.reset(0.30)
        for _ in range(12):
            camera.update(0.70, 1 / 30, urgent=True)
        fixed_target = camera.x
        outputs = [camera.update(fixed_target + 0.002, 1 / 30) for _ in range(30)]
        self.assertTrue(all(abs(value - fixed_target) < 1e-12 for value in outputs))
        self.assertEqual(0.0, camera.v)

    def test_urgent_switch_catches_up_faster(self):
        normal = self.make_camera()
        urgent = self.make_camera()
        normal.reset(0.50)
        urgent.reset(0.50)
        for _ in range(12):
            normal.update(0.54, 1 / 30, urgent=False)
            urgent.update(0.54, 1 / 30, urgent=True)
        self.assertGreater(urgent.x, normal.x + 0.005)


if __name__ == "__main__":
    unittest.main()
