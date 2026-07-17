import unittest

from vertical_focus.tracker import TrackManager
from vertical_focus.types import Detection


class TrackerTests(unittest.TestCase):
    def test_identity_is_stable_across_small_motion(self):
        tracker = TrackManager()
        first = tracker.update([Detection("person", 0.9, (0.2, 0.2, 0.4, 0.9))], 0)
        second = tracker.update([Detection("person", 0.88, (0.21, 0.2, 0.41, 0.9))], 3)
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(2, second[0].hits)

    def test_fast_ball_keeps_identity_and_predicts_forward(self):
        tracker = TrackManager(ball_match_center_distance=0.30, ball_box_alpha=0.90)
        first = tracker.update([Detection("ball", 0.8, (0.20, 0.45, 0.22, 0.48))], 0)
        second = tracker.update([Detection("ball", 0.82, (0.32, 0.45, 0.34, 0.48))], 6)
        predicted = tracker.snapshot(9)[0]
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertGreater(predicted.box[0], second[0].box[0])


if __name__ == "__main__":
    unittest.main()
