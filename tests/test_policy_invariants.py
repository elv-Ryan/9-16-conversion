import math
import random
import unittest

from vertical_focus.config import RuntimeParams, load_configuration
from vertical_focus.policy import FocusPolicy
from vertical_focus.types import Detection, TrackView


SPORTS_CONFIG, _, _ = load_configuration(RuntimeParams(mode="sports"))
MOVIE_CONFIG, _, _ = load_configuration(RuntimeParams(mode="movie"))
ALLOWED_REGIMES = {
    "locked",
    "normal_follow",
    "ball_follow",
    "fast_reacquire",
    "compose",
    "confirmed_reframe",
    "settle_then_lock",
    "cut_reset",
}


def bounded_box(rng: random.Random):
    width = rng.uniform(0.01, 0.24)
    height = rng.uniform(0.02, 0.70)
    x1 = rng.uniform(0.0, 1.0 - width)
    y1 = rng.uniform(0.0, 1.0 - height)
    return (x1, y1, x1 + width, y1 + height)


class PolicyInvariantTests(unittest.TestCase):
    def test_sports_ball_dominance_is_translation_invariant(self):
        for center in (0.06, 0.18, 0.34, 0.50, 0.66, 0.82, 0.94):
            with self.subTest(center=center):
                policy = FocusPolicy("sports", SPORTS_CONFIG)
                ball = TrackView(
                    "ball:1",
                    "ball",
                    0.85,
                    (center - 0.01, 0.45, center + 0.01, 0.48),
                    hits=4,
                )
                player_center = min(0.94, max(0.06, center + (0.14 if center < 0.5 else -0.14)))
                player = TrackView(
                    "person:1",
                    "person",
                    0.90,
                    (player_center - 0.07, 0.20, player_center + 0.07, 0.92),
                    hits=5,
                )
                selected = policy.select([player, ball], None, crop_width=0.316, time_s=0.0)
                self.assertEqual("ball_focus", selected.label)
                self.assertLessEqual(abs(selected.center_x - center), 0.04)
                self.assertLess(
                    abs(selected.center_x - center),
                    abs(selected.center_x - player_center),
                )

    def test_movie_character_identity_is_translation_invariant(self):
        for center in (0.16, 0.32, 0.50, 0.68, 0.84):
            with self.subTest(center=center):
                policy = FocusPolicy("movie", MOVIE_CONFIG)
                person = TrackView(
                    "person:1",
                    "person",
                    0.90,
                    (center - 0.10, 0.16, center + 0.10, 0.92),
                    hits=5,
                )
                face = TrackView(
                    "face:1",
                    "face",
                    0.92,
                    (center - 0.045, 0.18, center + 0.045, 0.38),
                    hits=5,
                )
                selected = policy.select([person, face], None, crop_width=0.316, time_s=0.0)
                self.assertEqual("character:person:1", selected.key)
                self.assertAlmostEqual(center, selected.center_x, places=6)

    def test_randomized_sequences_preserve_output_invariants(self):
        rng = random.Random(20260720)
        for mode, config in (("sports", SPORTS_CONFIG), ("movie", MOVIE_CONFIG)):
            policy = FocusPolicy(mode, config)
            for step in range(400):
                if step and step % 80 == 0:
                    policy.reset()
                tracks = []
                for index in range(rng.randint(0, 10)):
                    label = rng.choice(["person", "person", "face", "head", "ball"])
                    tracks.append(
                        TrackView(
                            f"{label}:{index}",
                            label,
                            rng.uniform(0.03, 1.0),
                            bounded_box(rng),
                            hits=rng.randint(1, 9),
                            misses=rng.randint(0, 4),
                            vx_per_frame=rng.uniform(-0.02, 0.02),
                            vy_per_frame=rng.uniform(-0.02, 0.02),
                            observed_this_frame=rng.choice([True, False]),
                        )
                    )
                motion = None
                if rng.random() < 0.65:
                    motion = Detection("motion", rng.random(), bounded_box(rng), source="test")
                selected = policy.select(
                    tracks,
                    motion,
                    crop_width=rng.uniform(0.20, 0.55),
                    time_s=step * 0.1,
                )
                self.assertTrue(math.isfinite(selected.center_x))
                self.assertTrue(math.isfinite(selected.score))
                self.assertTrue(0.0 <= selected.center_x <= 1.0)
                self.assertTrue(0.0 <= selected.score <= 1.0)
                self.assertIn(selected.smoothing_regime, ALLOWED_REGIMES)
                for box in selected.boxes:
                    x1, y1, x2, y2 = box.box
                    self.assertTrue(0.0 <= x1 < x2 <= 1.0)
                    self.assertTrue(0.0 <= y1 < y2 <= 1.0)


if __name__ == "__main__":
    unittest.main()
