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

    def test_ball_association_uses_predicted_position(self):
        tracker = TrackManager(
            ball_match_center_distance=0.18,
            ball_box_alpha=0.90,
            prediction_max_distance=0.20,
        )
        first = tracker.update([Detection("ball", 0.8, (0.20, 0.45, 0.22, 0.48))], 0)
        second = tracker.update([Detection("ball", 0.82, (0.32, 0.45, 0.34, 0.48))], 6)
        third = tracker.update([Detection("ball", 0.84, (0.53, 0.45, 0.55, 0.48))], 12)
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(first[0].track_id, third[0].track_id)
        self.assertEqual(3, third[0].hits)

    def test_low_confidence_far_ball_after_miss_creates_new_track(self):
        tracker = TrackManager(
            ball_match_center_distance=0.18,
            ball_reacquire_center_distance=0.28,
            ball_reacquire_min_confidence=0.55,
        )
        first = tracker.update([Detection("ball", 0.8, (0.20, 0.45, 0.22, 0.48))], 0)
        tracker.update([Detection("ball", 0.82, (0.24, 0.45, 0.26, 0.48))], 3)
        tracker.update([], 6)
        result = tracker.update([Detection("ball", 0.30, (0.60, 0.45, 0.62, 0.48))], 9)
        ids = {item.track_id for item in result}
        self.assertIn(first[0].track_id, ids)
        self.assertEqual(2, len(ids))

    def test_high_confidence_reacquisition_can_use_bounded_wider_gate(self):
        tracker = TrackManager(
            ball_match_center_distance=0.18,
            ball_reacquire_center_distance=0.28,
            ball_reacquire_min_confidence=0.55,
            prediction_max_distance=0.20,
        )
        first = tracker.update([Detection("ball", 0.80, (0.20, 0.45, 0.22, 0.48))], 0)
        tracker.update([Detection("ball", 0.82, (0.24, 0.45, 0.26, 0.48))], 3)
        tracker.update([], 6)
        result = tracker.update([Detection("ball", 0.80, (0.48, 0.45, 0.50, 0.48))], 9)
        self.assertEqual(1, len(result))
        self.assertEqual(first[0].track_id, result[0].track_id)


    def test_class_specific_box_alpha_reduces_observed_person_box_lag(self):
        default = TrackManager(box_alpha=0.50, match_center_distance=0.30)
        responsive = TrackManager(
            box_alpha=0.50,
            match_center_distance=0.30,
            box_alpha_by_label={"person": 0.92},
        )
        initial = Detection("person", 0.9, (0.10, 0.10, 0.30, 0.90))
        moved = Detection("person", 0.9, (0.30, 0.10, 0.50, 0.90))
        default.update([initial], 0)
        responsive.update([initial], 0)
        default_box = default.update([moved], 1)[0].box
        responsive_box = responsive.update([moved], 1)[0].box
        default_center = 0.5 * (default_box[0] + default_box[2])
        responsive_center = 0.5 * (responsive_box[0] + responsive_box[2])
        self.assertGreater(responsive_center, default_center + 0.05)
        self.assertLess(abs(responsive_center - 0.40), abs(default_center - 0.40))

    def test_snapshot_marks_prediction_as_not_observed(self):
        tracker = TrackManager()
        tracker.update([Detection("person", 0.9, (0.2, 0.2, 0.4, 0.9))], 0)
        predicted = tracker.snapshot(1)[0]
        self.assertFalse(predicted.observed)
        observed = tracker.update([Detection("person", 0.9, (0.21, 0.2, 0.41, 0.9))], 3)[0]
        self.assertTrue(observed.observed)

    def test_class_specific_miss_limits_remove_face_before_person(self):
        tracker = TrackManager(
            max_missed_updates=5,
            max_missed_updates_by_label={"face": 1, "person": 4},
        )
        tracker.update(
            [
                Detection("person", 0.9, (0.2, 0.2, 0.4, 0.9)),
                Detection("face", 0.9, (0.25, 0.22, 0.35, 0.38)),
            ],
            0,
        )
        one_miss = tracker.update([], 1)
        self.assertEqual({"face", "person"}, {item.label for item in one_miss})
        two_misses = tracker.update([], 2)
        self.assertEqual({"person"}, {item.label for item in two_misses})


if __name__ == "__main__":
    unittest.main()
