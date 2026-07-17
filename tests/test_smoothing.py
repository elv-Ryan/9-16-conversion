import unittest

from vertical_focus.smoothing import SmoothCamera


class SmoothCameraTests(unittest.TestCase):
    def test_speed_and_crop_bounds(self):
        camera = SmoothCamera(
            crop_width=0.31640625,
            response_time_s=0.30,
            deadband=0.004,
            max_speed_per_s=0.25,
            max_accel_per_s2=0.80,
        )
        values = [camera.reset(0.25)]
        dt = 1.0 / 30.0
        for _ in range(180):
            values.append(camera.update(0.80, dt))
        speeds = [abs(b - a) / dt for a, b in zip(values, values[1:])]
        self.assertLessEqual(max(speeds), 0.250001)
        half = 0.31640625 / 2.0
        self.assertTrue(all(half <= value <= 1.0 - half for value in values))
        self.assertGreater(values[-1], values[0])

    def test_deadband_removes_small_jitter(self):
        camera = SmoothCamera(
            crop_width=0.316,
            response_time_s=0.4,
            deadband=0.01,
            max_speed_per_s=0.2,
            max_accel_per_s2=0.5,
        )
        start = camera.reset(0.5)
        outputs = [camera.update(0.5 + (0.004 if i % 2 else -0.004), 1 / 30) for i in range(60)]
        self.assertAlmostEqual(start, outputs[-1], places=6)


if __name__ == "__main__":
    unittest.main()
