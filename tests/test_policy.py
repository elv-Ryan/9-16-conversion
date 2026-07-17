import unittest

from vertical_focus.policy import FocusPolicy
from vertical_focus.types import TrackView


SPORTS_CONFIG = {
    "selection": {
        "min_track_hits": 2,
        "min_person_area": 0.0025,
        "ball_min_track_hits": 1,
        "ball_max_misses": 2,
        "ball_min_area": 0.00001,
        "ball_max_area": 0.03,
        "ball_context_radius": 0.34,
        "ball_stable_hits_without_context": 3,
        "ball_focus_weight": 0.82,
        "ball_priority_score": 0.82,
        "ball_player_radius": 0.28,
        "min_ball_player_score": 0.35,
        "group_radius": 0.24,
        "max_group_size": 3,
        "announcer_face_min_area": 0.025,
        "min_motion_score": 0.14,
        "motion_only_without_subjects": True,
        "crowd_min_people": 4,
        "crowd_max_median_area": 0.015,
        "crowd_max_motion_score": 0.18,
        "crowd_group_score": 0.36,
    },
    "temporal": {
        "min_hold_seconds": 0.42,
        "lost_hold_seconds": 0.32,
        "switch_margin": 0.10,
        "hold_last_on_empty": True,
        "fast_switch_labels": ["ball_focus"],
        "fast_switch_min_hold_seconds": 0.05,
        "fast_switch_margin": -0.03,
    },
}

MOVIE_CONFIG = {
    "selection": {
        "min_track_hits": 2,
        "min_face_area": 0.0012,
        "group_score_ratio": 0.70,
        "group_crop_width_factor": 1.08,
        "min_motion_score": 0.14,
        "motion_only_without_subjects": True,
    },
    "temporal": {
        "min_hold_seconds": 0.68,
        "lost_hold_seconds": 0.42,
        "switch_margin": 0.14,
        "hold_last_on_empty": True,
        "fast_switch_labels": ["action_region"],
        "fast_switch_min_hold_seconds": 0.18,
        "fast_switch_margin": 0.04,
    },
}


class FocusPolicyTests(unittest.TestCase):
    def test_sports_ball_is_primary_and_player_is_context(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        player_center = (0.38 + 0.53) / 2
        ball_center = (0.50 + 0.515) / 2
        tracks = [
            TrackView("person:1", "person", 0.90, (0.38, 0.25, 0.53, 0.92), hits=4),
            TrackView("person:2", "person", 0.84, (0.70, 0.30, 0.82, 0.90), hits=4),
            TrackView("ball:3", "ball", 0.72, (0.50, 0.55, 0.515, 0.575), hits=3),
        ]
        selected = policy.select(tracks, None, crop_width=0.316, time_s=0.0)
        self.assertEqual("ball_focus", selected.label)
        self.assertIn("ball:3", selected.key)
        self.assertEqual("ball", selected.boxes[0].label)
        self.assertLess(abs(selected.center_x - ball_center), abs(selected.center_x - player_center))

    def test_ball_priority_can_interrupt_player_hold(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        player = TrackView("person:1", "person", 0.92, (0.15, 0.2, 0.32, 0.92), hits=4)
        first = policy.select([player], None, crop_width=0.316, time_s=0.0)
        self.assertEqual("active_player", first.label)
        ball = TrackView("ball:2", "ball", 0.80, (0.74, 0.48, 0.755, 0.505), hits=1)
        near_player = TrackView("person:3", "person", 0.85, (0.66, 0.25, 0.80, 0.93), hits=4)
        second = policy.select([player, near_player, ball], None, crop_width=0.316, time_s=0.10)
        self.assertEqual("ball_focus", second.label)

    def test_sports_crowd_uses_group_not_random_small_person(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        crowd = [
            TrackView(f"person:{index}", "person", 0.75, (0.10 + index * 0.14, 0.18, 0.15 + index * 0.14, 0.38), hits=4)
            for index in range(5)
        ]
        selected = policy.select(crowd, None, crop_width=0.316, time_s=0.0)
        self.assertEqual("crowd_group", selected.label)
        self.assertGreaterEqual(len(selected.boxes), 2)

    def test_no_evidence_holds_last_position(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        face = TrackView("face:1", "face", 0.88, (0.65, 0.20, 0.78, 0.48), hits=4)
        first = policy.select([face], None, crop_width=0.316, time_s=0.0)
        second = policy.select([], None, crop_width=0.316, time_s=2.0)
        self.assertEqual(first.key, second.key)
        self.assertAlmostEqual(first.center_x, second.center_x)

    def test_movie_rejects_transient_background_face(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        primary = TrackView("face:1", "face", 0.88, (0.35, 0.20, 0.48, 0.48), hits=4)
        first = policy.select([primary], None, crop_width=0.316, time_s=0.0)
        transient = TrackView("face:2", "face", 0.99, (0.72, 0.10, 0.78, 0.20), hits=1)
        second = policy.select([primary, transient], None, crop_width=0.316, time_s=1.0)
        self.assertEqual(first.key, second.key)
        self.assertIn("face:1", second.key)

    def test_movie_can_select_interaction_group(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        faces = [
            TrackView("face:1", "face", 0.92, (0.37, 0.20, 0.45, 0.39), hits=4),
            TrackView("face:2", "face", 0.89, (0.52, 0.22, 0.60, 0.41), hits=4),
        ]
        selected = policy.select(faces, None, crop_width=0.316, time_s=0.0)
        self.assertEqual("interaction_group", selected.label)
        self.assertEqual(2, len(selected.boxes))


if __name__ == "__main__":
    unittest.main()
