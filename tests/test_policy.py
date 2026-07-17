import unittest

from vertical_focus.policy import FocusPolicy
from vertical_focus.types import Detection, TrackView


SPORTS_CONFIG = {
    "selection": {
        "min_track_hits": 2,
        "min_person_area": 0.0025,
        "direct_ball_min_area": 0.004,
        "ball_player_radius": 0.22,
        "min_ball_player_score": 0.45,
        "group_radius": 0.24,
        "max_group_size": 3,
        "announcer_face_min_area": 0.025,
        "min_motion_score": 0.08,
        "crowd_min_people": 4,
        "crowd_max_median_area": 0.015,
        "crowd_max_motion_score": 0.20,
        "crowd_group_score": 0.42,
    },
    "temporal": {"min_hold_seconds": 0.5, "lost_hold_seconds": 0.4, "switch_margin": 0.13},
}

MOVIE_CONFIG = {
    "selection": {
        "min_track_hits": 2,
        "min_face_area": 0.0015,
        "group_score_ratio": 0.72,
        "group_crop_width_factor": 1.05,
        "min_motion_score": 0.1,
    },
    "temporal": {"min_hold_seconds": 0.8, "lost_hold_seconds": 0.5, "switch_margin": 0.17},
}


class FocusPolicyTests(unittest.TestCase):
    def test_sports_ball_selects_near_player_not_tiny_ball(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        tracks = [
            TrackView("person:1", "person", 0.90, (0.38, 0.25, 0.53, 0.92), hits=4),
            TrackView("person:2", "person", 0.84, (0.70, 0.30, 0.82, 0.90), hits=4),
            TrackView("ball:3", "ball", 0.72, (0.50, 0.55, 0.515, 0.575), hits=3),
        ]
        selected = policy.select(tracks, None, crop_width=0.316, time_s=0.0)
        self.assertEqual("player_near_ball", selected.label)
        self.assertIn("person:1", selected.key)
        self.assertEqual(2, len(selected.boxes))


    def test_sports_crowd_uses_group_not_random_small_person(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        crowd = [
            TrackView(f"person:{index}", "person", 0.75, (0.10 + index * 0.14, 0.18, 0.15 + index * 0.14, 0.38), hits=4)
            for index in range(5)
        ]
        selected = policy.select(crowd, None, crop_width=0.316, time_s=0.0)
        self.assertEqual("crowd_group", selected.label)
        self.assertGreaterEqual(len(selected.boxes), 2)

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
