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


if __name__ == "__main__":
    unittest.main()
