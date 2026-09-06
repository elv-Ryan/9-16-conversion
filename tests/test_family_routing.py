import unittest

from nba_yolo_shot_tagger.service import FOCUS_LINK_MAX_GAP_SAMPLES, ShotFocusService
from nba_yolo_shot_tagger.trajectory import _fill_missing
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

    def test_absent_family_yields_no_focus_rather_than_a_foreign_box(self):
        # Substituting another family's box moves the crop onto a subject the
        # shot is not about, and does it silently: the sample still looks like
        # evidence downstream. A gap is the honest answer.
        frame = FrameEvidence(
            frame_index=0,
            timestamp_ms=0,
            candidates=(
                Candidate("graphic_text_lock", 0.95, (0.7, 0.1, 0.9, 0.9)),
                Candidate("person_subject", 0.80, (0.1, 0.1, 0.3, 0.9)),
            ),
        )
        self.assertIsNone(ShotFocusService._select_focus(frame, "gameplay_follow"))

    def test_a_gap_is_interpolated_from_the_right_familys_neighbours(self):
        service = ShotFocusService.__new__(ShotFocusService)
        service._cycle_focus_anchor = None
        service._cycle_focus_gap = 0
        evidence = [
            FrameEvidence(0, 0, (Candidate("gameplay_follow", 0.7, (0.1, 0.2, 0.3, 0.8)),)),
            # Nothing of the voted family here, only a loud distractor at the
            # far edge of the frame.
            FrameEvidence(6, 100, (Candidate("graphic_text_lock", 0.99, (0.8, 0.2, 1.0, 0.8)),)),
            FrameEvidence(12, 200, (Candidate("gameplay_follow", 0.7, (0.3, 0.2, 0.5, 0.8)),)),
        ]
        _, raw_x, confidences, samples = service._select_focus_series(
            evidence, "gameplay_follow"
        )
        self.assertEqual(raw_x, [0.2, None, 0.4])
        self.assertEqual(confidences, [0.7, 0.0, 0.7])
        self.assertEqual(samples[1].family, "safe_center")
        self.assertIsNone(samples[1].bbox)
        # The gap interpolates between the two real observations instead of
        # snapping to the distractor's centre at 0.9.
        filled = _fill_missing(raw_x)
        self.assertAlmostEqual(float(filled[1]), 0.3)

    def test_near_tied_boxes_are_decided_by_continuity_not_noise(self):
        # Two person_subject boxes a third of the frame apart, separated by a
        # confidence difference of 1e-4 -- the exact shape that flipped between
        # fp16 and fp32 in practice. Without an anchor the argmax picks the
        # right-hand one; with the crop already on the left, the left one wins.
        frame = FrameEvidence(
            frame_index=0,
            timestamp_ms=0,
            candidates=(
                Candidate("person_subject", 0.1366, (0.60, 0.1, 0.80, 0.9)),
                Candidate("person_subject", 0.1365, (0.25, 0.1, 0.45, 0.9)),
            ),
        )
        self.assertAlmostEqual(
            ShotFocusService._select_focus(frame, "person_subject").x_center, 0.70
        )
        self.assertAlmostEqual(
            ShotFocusService._select_focus(
                frame, "person_subject", anchor_x=0.34
            ).x_center,
            0.35,
        )

    def test_a_clearly_better_box_still_wins_over_continuity(self):
        # Continuity must not become a lock: a subject the detector genuinely
        # prefers has to be able to take the focus.
        frame = FrameEvidence(
            frame_index=0,
            timestamp_ms=0,
            candidates=(
                Candidate("person_subject", 0.80, (0.60, 0.1, 0.80, 0.9)),
                Candidate("person_subject", 0.10, (0.25, 0.1, 0.45, 0.9)),
            ),
        )
        selected = ShotFocusService._select_focus(
            frame, "person_subject", anchor_x=0.35
        )
        self.assertAlmostEqual(selected.x_center, 0.70)

    def test_continuity_is_not_invented_across_the_frame(self):
        # The nearest tied box is still nowhere near the anchor, so linking to
        # it would be fiction. Fall back to the detector's own preference.
        frame = FrameEvidence(
            frame_index=0,
            timestamp_ms=0,
            candidates=(
                Candidate("person_subject", 0.50, (0.80, 0.1, 0.90, 0.9)),
                Candidate("person_subject", 0.45, (0.60, 0.1, 0.70, 0.9)),
            ),
        )
        selected = ShotFocusService._select_focus(
            frame, "person_subject", anchor_x=0.05
        )
        self.assertAlmostEqual(selected.x_center, 0.85)

    def test_the_anchor_goes_stale_after_a_long_absence(self):
        service = ShotFocusService.__new__(ShotFocusService)
        service._cycle_focus_anchor = None
        service._cycle_focus_gap = 0
        tied = (
            Candidate("person_subject", 0.30, (0.60, 0.1, 0.80, 0.9)),
            Candidate("person_subject", 0.29, (0.10, 0.1, 0.30, 0.9)),
        )
        # Anchor the focus on the left box, then starve it past the gap limit.
        evidence = [FrameEvidence(0, 0, (tied[1],))]
        evidence += [
            FrameEvidence(i, i * 100, ()) for i in range(1, FOCUS_LINK_MAX_GAP_SAMPLES + 3)
        ]
        evidence.append(FrameEvidence(99, 9900, tied))
        _, raw_x, _, _ = service._select_focus_series(evidence, "person_subject")
        # Anchor expired, so the reappearance is decided by confidence alone.
        self.assertAlmostEqual(raw_x[0], 0.20)
        self.assertAlmostEqual(raw_x[-1], 0.70)

    def test_the_anchor_survives_a_short_absence(self):
        service = ShotFocusService.__new__(ShotFocusService)
        service._cycle_focus_anchor = None
        service._cycle_focus_gap = 0
        tied = (
            Candidate("person_subject", 0.30, (0.60, 0.1, 0.80, 0.9)),
            Candidate("person_subject", 0.29, (0.10, 0.1, 0.30, 0.9)),
        )
        evidence = [FrameEvidence(0, 0, (tied[1],))]
        evidence += [FrameEvidence(i, i * 100, ()) for i in range(1, 3)]
        evidence.append(FrameEvidence(9, 900, tied))
        _, raw_x, _, _ = service._select_focus_series(evidence, "person_subject")
        self.assertAlmostEqual(raw_x[-1], 0.20)

    def test_association_carries_across_batches(self):
        service = ShotFocusService.__new__(ShotFocusService)
        service._cycle_focus_anchor = None
        service._cycle_focus_gap = 0
        tied = (
            Candidate("person_subject", 0.30, (0.60, 0.1, 0.80, 0.9)),
            Candidate("person_subject", 0.29, (0.10, 0.1, 0.30, 0.9)),
        )
        service._select_focus_series([FrameEvidence(0, 0, (tied[1],))], "person_subject")
        self.assertAlmostEqual(service._cycle_focus_anchor, 0.20)
        # A second call stands in for the next segment of the same open shot.
        _, raw_x, _, _ = service._select_focus_series(
            [FrameEvidence(6, 100, tied)], "person_subject"
        )
        self.assertAlmostEqual(raw_x[0], 0.20)

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
