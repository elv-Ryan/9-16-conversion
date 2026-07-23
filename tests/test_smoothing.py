import unittest

from vertical_focus.smoothing import SmoothCamera


class SmoothCameraTests(unittest.TestCase):
    def make_camera(self):
        return SmoothCamera(
            crop_width=0.31640625,
            response_time_s=0.46,
            deadband=0.009,
            max_speed_per_s=0.18,
            max_accel_per_s2=0.55,
            fast_response_time_s=0.10,
            fast_max_speed_per_s=0.68,
            fast_max_accel_per_s2=3.0,
            fast_error_threshold=0.08,
            fast_boost_s=0.28,
            target_filter_s=0.16,
            fast_on_large_error=False,
        )

    def test_normal_large_error_does_not_activate_fast_mode(self):
        camera = self.make_camera()
        values = [camera.reset(0.25)]
        dt = 1.0 / 30.0
        for _ in range(90):
            values.append(camera.update(0.75, dt, urgent=False))
        speeds = [abs(b - a) / dt for a, b in zip(values, values[1:])]
        self.assertLessEqual(max(speeds), camera.max_speed + 1e-6)

    def test_urgent_ball_switch_uses_fast_mode(self):
        normal = self.make_camera()
        urgent = self.make_camera()
        normal.reset(0.30)
        urgent.reset(0.30)
        for _ in range(12):
            normal.update(0.70, 1 / 30, urgent=False)
            urgent.update(0.70, 1 / 30, urgent=True)
        self.assertGreater(urgent.x, normal.x + 0.04)

    def test_deadband_stops_residual_drift_immediately(self):
        camera = self.make_camera()
        camera.reset(0.30)
        for _ in range(12):
            camera.update(0.70, 1 / 30, urgent=True)
        fixed_target = camera.x
        camera.filtered_target = fixed_target
        outputs = [camera.update(fixed_target + 0.002, 1 / 30) for _ in range(30)]
        self.assertTrue(all(abs(value - fixed_target) < 1e-12 for value in outputs))
        self.assertEqual(0.0, camera.v)


    def test_named_ball_follow_profile_reduces_target_lag_with_bounded_speed(self):
        profiles = {
            "normal_follow": {
                "response_time_seconds": 0.32,
                "deadband": 0.012,
                "target_filter_seconds": 0.10,
                "max_speed_normalized_per_second": 0.26,
                "max_acceleration_normalized_per_second2": 0.8,
            },
            "ball_follow": {
                "response_time_seconds": 0.10,
                "deadband": 0.005,
                "target_filter_seconds": 0.025,
                "max_speed_normalized_per_second": 0.70,
                "max_acceleration_normalized_per_second2": 3.0,
            },
        }
        normal = self.make_camera()
        ball = SmoothCamera(
            crop_width=0.31640625,
            response_time_s=0.46,
            deadband=0.009,
            max_speed_per_s=0.18,
            max_accel_per_s2=0.55,
            profiles=profiles,
        )
        normal.profiles = {
            name: normal._normalize_profile(values) for name, values in profiles.items()
        }
        normal.reset(0.22)
        ball.reset(0.22)
        dt = 1.0 / 60.0
        ball_values = [ball.x]
        for _ in range(24):
            normal.update(0.72, dt, profile_name="normal_follow")
            ball_values.append(ball.update(0.72, dt, profile_name="ball_follow"))
        self.assertGreater(ball.x, normal.x + 0.04)
        speeds = [
            abs(right - left) / dt
            for left, right in zip(ball_values, ball_values[1:])
        ]
        self.assertLessEqual(max(speeds), 0.70 + 1e-9)

    def test_hold_zeros_velocity_and_target(self):
        camera = self.make_camera()
        camera.reset(0.30)
        for _ in range(6):
            camera.update(0.70, 1 / 30, urgent=True)
        held = camera.hold()
        self.assertEqual(0.0, camera.v)
        self.assertEqual(held, camera.filtered_target)

    def test_follow_profile_has_no_residual_pan_after_target_stops(self):
        camera = SmoothCamera(
            crop_width=0.31640625,
            response_time_s=0.4,
            deadband=0.01,
            max_speed_per_s=0.2,
            max_accel_per_s2=0.5,
            profiles={
                "ball_follow": {
                    "mode": "follow",
                    "response_time_seconds": 0.055,
                    "deadband": 0.003,
                    "tracking_zone": 0.006,
                    "target_filter_seconds": 0.0,
                    "max_speed_normalized_per_second": 1.5,
                    "snap_epsilon": 0.002,
                }
            },
        )
        camera.reset(0.25)
        dt = 1.0 / 60.0
        for _ in range(18):
            camera.update(0.70, dt, profile_name="ball_follow")
        stopped_target = camera.x + 0.004
        held = camera.update(stopped_target, dt, profile_name="ball_follow")
        outputs = [
            camera.update(stopped_target, dt, profile_name="ball_follow")
            for _ in range(30)
        ]
        self.assertTrue(all(abs(value - held) < 1e-12 for value in outputs))
        self.assertEqual(0.0, camera.v)

    def test_reframe_profile_finishes_quickly_and_holds_exactly(self):
        camera = SmoothCamera(
            crop_width=0.31640625,
            response_time_s=0.4,
            deadband=0.01,
            max_speed_per_s=0.2,
            max_accel_per_s2=0.5,
            profiles={
                "confirmed_reframe": {
                    "mode": "reframe",
                    "deadband": 0.004,
                    "tracking_zone": 0.0,
                    "reframe_min_seconds": 0.16,
                    "reframe_max_seconds": 0.30,
                    "reframe_speed_normalized_per_second": 1.5,
                    "retarget_threshold": 0.08,
                    "snap_epsilon": 0.002,
                }
            },
        )
        camera.reset(0.22)
        dt = 1.0 / 60.0
        outputs = [camera.x]
        for _ in range(30):
            outputs.append(
                camera.update(0.72, dt, profile_name="confirmed_reframe")
            )
            if not camera.reframing:
                break
        self.assertLessEqual((len(outputs) - 1) * dt, 0.31)
        self.assertTrue(
            all(right >= left - 1e-12 for left, right in zip(outputs, outputs[1:]))
        )
        self.assertAlmostEqual(0.72, camera.x, places=9)
        self.assertEqual(0.0, camera.v)
        held = camera.hold()
        self.assertTrue(
            all(
                abs(camera.hold() - held) < 1e-12
                for _ in range(20)
            )
        )

    def test_one_frame_reacquire_still_completes_finite_reframe(self):
        camera = SmoothCamera(
            crop_width=0.31640625,
            response_time_s=0.4,
            deadband=0.01,
            max_speed_per_s=0.2,
            max_accel_per_s2=0.5,
            profiles={
                "fast_reacquire": {
                    "mode": "reframe",
                    "deadband": 0.003,
                    "tracking_zone": 0.005,
                    "reframe_min_seconds": 0.16,
                    "reframe_max_seconds": 0.30,
                    "reframe_speed_normalized_per_second": 1.8,
                    "retarget_threshold": 0.10,
                },
                "ball_follow": {
                    "mode": "follow",
                    "response_time_seconds": 0.055,
                    "deadband": 0.003,
                    "tracking_zone": 0.006,
                    "max_speed_normalized_per_second": 1.5,
                    "retarget_threshold": 0.06,
                },
            },
        )
        camera.reset(0.22)
        dt = 1.0 / 60.0
        camera.update(0.72, dt, profile_name="fast_reacquire")
        self.assertTrue(camera.reframing)
        for _ in range(30):
            camera.update(0.72, dt, profile_name="ball_follow")
            if not camera.reframing:
                break
        self.assertFalse(camera.reframing)
        self.assertGreater(camera.x, 0.69)


if __name__ == "__main__":
    unittest.main()
