import unittest

import cv2
import numpy as np

from vertical_focus.shots import LocalShotDetector


def textured_mosaic(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cells = np.array([0] * 72 + [255] * 72, dtype=np.uint8)
    rng.shuffle(cells)
    grid = cells.reshape(9, 16)
    gray = cv2.resize(grid, (320, 180), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


class LocalShotDetectorTests(unittest.TestCase):
    def make_detector(self):
        return LocalShotDetector(
            threshold=1.1,
            gray_mad_threshold=0.18,
            edge_change_threshold=0.16,
            minimum_seconds=0.1,
            fps=30.0,
        )

    def test_same_palette_structural_cut_uses_secondary_cue(self):
        detector = self.make_detector()
        first = textured_mosaic(1)
        second = textured_mosaic(2)
        self.assertFalse(detector.update(first, 0))
        self.assertTrue(detector.update(second, 5))

    def test_brightness_flash_does_not_trigger_without_edge_change(self):
        detector = self.make_detector()
        first = textured_mosaic(3)
        flashed = np.clip(first.astype(np.int16) + 90, 0, 255).astype(np.uint8)
        self.assertFalse(detector.update(first, 0))
        self.assertFalse(detector.update(flashed, 5))

    def test_small_pan_does_not_trigger_secondary_cut(self):
        detector = self.make_detector()
        first = textured_mosaic(4)
        panned = np.roll(first, 3, axis=1)
        self.assertFalse(detector.update(first, 0))
        self.assertFalse(detector.update(panned, 5))

    def test_small_zoom_does_not_trigger_secondary_cut(self):
        detector = self.make_detector()
        first = textured_mosaic(5)
        height, width = first.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 0.0, 1.03)
        zoomed = cv2.warpAffine(
            first, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        self.assertFalse(detector.update(first, 0))
        self.assertFalse(detector.update(zoomed, 5))

    def test_moderate_multi_cue_cut_catches_below_primary_threshold(self):
        detector = LocalShotDetector(
            threshold=1.1,
            gray_mad_threshold=0.95,
            edge_change_threshold=0.95,
            alignment_response_threshold=0.05,
            moderate_histogram_threshold=0.0,
            moderate_gray_mad_threshold=0.05,
            moderate_edge_change_threshold=0.10,
            moderate_alignment_response_threshold=1.0,
            minimum_seconds=0.1,
            fps=30.0,
        )
        first = textured_mosaic(11)
        second = textured_mosaic(12)
        self.assertFalse(detector.update(first, 0))
        self.assertTrue(detector.update(second, 5))

    def test_moderate_cut_rule_still_rejects_aligned_pan(self):
        detector = LocalShotDetector(
            threshold=1.1,
            gray_mad_threshold=0.95,
            edge_change_threshold=0.95,
            alignment_response_threshold=0.05,
            moderate_histogram_threshold=0.02,
            moderate_gray_mad_threshold=0.01,
            moderate_edge_change_threshold=0.05,
            moderate_alignment_response_threshold=0.35,
            minimum_seconds=0.1,
            fps=30.0,
        )
        first = textured_mosaic(13)
        panned = np.roll(first, 5, axis=1)
        self.assertFalse(detector.update(first, 0))
        self.assertFalse(detector.update(panned, 5))


if __name__ == "__main__":
    unittest.main()
