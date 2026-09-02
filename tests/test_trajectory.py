import unittest

import numpy as np

from nba_yolo_shot_tagger.trajectory import (
    expand_to_source_frames,
    legal_crop_geometry,
    smooth_samples,
)


class TrajectoryTests(unittest.TestCase):
    def test_16_by_9_to_9_by_16_geometry(self):
        width, lower, upper = legal_crop_geometry(1920, 1080, 9 / 16)
        self.assertAlmostEqual(width, 0.31640625)
        self.assertAlmostEqual(lower, 0.158203125)
        self.assertAlmostEqual(upper, 0.841796875)

    def test_static_family_locks(self):
        smoothed = smooth_samples(
            family="static_composition",
            raw_x=[0.3, 0.31, 0.8, 0.3],
            confidences=[0.9, 0.9, 0.01, 0.9],
            inference_fps=10,
            legal_min=0.15,
            legal_max=0.85,
        )
        self.assertTrue(np.allclose(smoothed, smoothed[0]))
        self.assertLess(abs(smoothed[0] - 0.3), 0.03)

    def test_gameplay_fills_missing_and_clamps(self):
        smoothed = smooth_samples(
            family="gameplay_follow",
            raw_x=[None, 0.0, 0.2, None, 1.0],
            confidences=[0, 0.2, 0.4, 0, 0.8],
            inference_fps=10,
            legal_min=0.158,
            legal_max=0.842,
        )
        self.assertEqual(len(smoothed), 5)
        self.assertTrue(np.isfinite(smoothed).all())
        self.assertTrue((smoothed >= 0.158).all())
        self.assertTrue((smoothed <= 0.842).all())

    def test_expansion_accepts_numpy_sample_x(self):
        values = expand_to_source_frames(
            sample_frame_indices=[0, 6, 12],
            sample_x=np.asarray([0.3, 0.5, 0.7], dtype=np.float64),
            start_frame=0,
            end_frame=13,
            legal_min=0.15,
            legal_max=0.85,
        )
        self.assertEqual(len(values), 13)
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(values[0], 0.3)
        self.assertAlmostEqual(values[-1], 0.7)

    def test_expansion_is_one_value_per_source_frame(self):
        values = expand_to_source_frames(
            sample_frame_indices=[10, 20, 29],
            sample_x=[0.2, 0.5, 0.8],
            start_frame=10,
            end_frame=30,
            legal_min=0.15,
            legal_max=0.85,
        )
        self.assertEqual(len(values), 20)
        self.assertAlmostEqual(values[0], 0.2)
        self.assertAlmostEqual(values[-1], 0.8)


if __name__ == "__main__":
    unittest.main()
