import unittest

from nba_yolo_shot_tagger.service import ShotFocusService
from nba_yolo_shot_tagger.types import Candidate, FrameEvidence


class FamilyRoutingTests(unittest.TestCase):
    def test_safe_center_does_not_outvote_real_family(self):
        evidence = [
            FrameEvidence(
                frame_index=0,
                timestamp_ms=0,
                candidates=(
                    Candidate("safe_center", 0.99, (0.4, 0.0, 0.6, 1.0)),
                    Candidate("gameplay_follow", 0.25, (0.2, 0.2, 0.4, 0.8)),
                ),
            ),
            FrameEvidence(
                frame_index=6,
                timestamp_ms=100,
                candidates=(
                    Candidate("gameplay_follow", 0.30, (0.5, 0.2, 0.7, 0.8)),
                ),
            ),
        ]
        family, confidence = ShotFocusService._family_vote(evidence)
        self.assertEqual(family, "gameplay_follow")
        self.assertGreater(confidence, 0.99)

    def test_matching_family_is_preferred_for_focus(self):
        frame = FrameEvidence(
            frame_index=0,
            timestamp_ms=0,
            candidates=(
                Candidate("person_subject", 0.9, (0.1, 0.1, 0.3, 0.9)),
                Candidate("active_speaker", 0.6, (0.6, 0.1, 0.8, 0.9)),
            ),
        )
        selected = ShotFocusService._select_focus(frame, "active_speaker")
        self.assertIsNotNone(selected)
        self.assertEqual(selected.family, "active_speaker")


if __name__ == "__main__":
    unittest.main()
