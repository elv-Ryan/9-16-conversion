import unittest

from nba_yolo_shot_tagger.config import RuntimeConfig
from nba_yolo_shot_tagger.service import ShotFocusService
from nba_yolo_shot_tagger.types import Candidate, FrameEvidence, ModelIdentity, VideoInfo


FPS = 59.94
FRAME_COUNT = 120  # ~2.002s at FPS, matching the real 2s test segments
DURATION_MS = round(1000.0 * FRAME_COUNT / FPS)
SPLIT_FRAME = FRAME_COUNT // 2


def _video_info() -> VideoInfo:
    return VideoInfo(fps=FPS, frame_count=FRAME_COUNT, width=1920, height=1080, duration_ms=DURATION_MS)


def _evidence(family: str = "gameplay_follow", confidence: float = 0.8):
    # Sample every 6th frame, matching ~10fps sampling of a 59.94fps source.
    frame_indices = list(range(0, FRAME_COUNT, 6))
    return [
        FrameEvidence(
            frame_index=index,
            timestamp_ms=round(1000.0 * index / FPS),
            candidates=(Candidate(family=family, confidence=confidence, bbox=(0.2, 0.2, 0.4, 0.8)),),
        )
        for index in frame_indices
    ]


class _FakeModel:
    identity = ModelIdentity(path="fake.pt", sha256="a" * 64, artifact_version="test", class_names=())


def _service(family_determination_max_seconds: float) -> ShotFocusService:
    service = ShotFocusService.__new__(ShotFocusService)
    service.config = RuntimeConfig(
        verify_model_sha256=False,
        family_determination_max_seconds=family_determination_max_seconds,
    )
    service.model = _FakeModel()
    service.segment_state = "determining_family"
    service.segment_family = None
    service.segment_family_confidence = None
    service.segment_raw_x = []
    service.segment_confidences = []
    service.segment_sample_frame_indices = []
    service.segment_focus_samples = []
    service._segment_evidence_buffer = []
    service._segment_buffered_seconds = 0.0
    service._segment_files_since_output_began = 0
    service._segment_shot_index = 0
    service._segment_cumulative_frames = 0
    service._segment_cumulative_ms = 0
    return service


class SegmentBufferingTests(unittest.TestCase):
    def test_family_determination_waits_for_enough_buffered_seconds(self):
        service = _service(family_determination_max_seconds=4.0)
        completed = service._advance_segment_state("file0.mp4", _video_info(), _evidence())
        self.assertEqual(completed, [])
        self.assertEqual(service.segment_state, "determining_family")
        self.assertIsNone(service.segment_family)
        self.assertEqual(service.segment_raw_x, [])

    def test_family_is_determined_once_threshold_reached_across_files(self):
        service = _service(family_determination_max_seconds=4.0)
        service._advance_segment_state("file0.mp4", _video_info(), _evidence())
        completed = service._advance_segment_state("file1.mp4", _video_info(), _evidence())
        self.assertEqual(completed, [])
        self.assertEqual(service.segment_state, "outputting_offset")
        self.assertEqual(service.segment_family, "gameplay_follow")
        # Both buffered files' evidence should have been backfilled into raw_x et al.
        expected_samples = len(_evidence()) * 2
        self.assertEqual(len(service.segment_raw_x), expected_samples)
        self.assertEqual(len(service.segment_confidences), expected_samples)
        self.assertEqual(len(service.segment_sample_frame_indices), expected_samples)
        self.assertEqual(len(service.segment_focus_samples), expected_samples)

    def test_subsequent_files_extend_buffers_without_revoting(self):
        service = _service(family_determination_max_seconds=4.0)
        service._advance_segment_state("file0.mp4", _video_info(), _evidence())
        service._advance_segment_state("file1.mp4", _video_info(), _evidence())
        after_family_count = len(service.segment_raw_x)
        completed = service._advance_segment_state(
            "file2.mp4", _video_info(), _evidence(family="active_speaker", confidence=0.99)
        )
        self.assertEqual(completed, [])
        self.assertEqual(service.segment_state, "outputting_offset")
        self.assertEqual(service.segment_family, "gameplay_follow")  # unchanged: no re-vote
        self.assertEqual(len(service.segment_raw_x), after_family_count + len(_evidence()))

    def test_arbitrary_boundary_emits_shot_with_negative_offsets(self):
        # NOTE: this exercises the real ShotAnalysis-building tail
        # (legal_crop_geometry/smooth_samples/expand_to_source_frames), which
        # needs numpy to actually run.
        service = _service(family_determination_max_seconds=4.0)
        completed = []
        for index in range(8):
            completed.extend(
                service._advance_segment_state(f"file{index}.mp4", _video_info(), _evidence())
            )
        # file7 is the 6th file since outputting_offset began (file2..file7) and triggers the boundary.
        self.assertEqual(len(completed), 1)
        shot = completed[0]
        self.assertEqual(service.segment_state, "determining_family")
        self.assertEqual(shot.family, "gameplay_follow")
        self.assertEqual(shot.source_media, "file7.mp4")

        # 2 files worth of family-determination backfill (file0, file1) + 5 full
        # files (file2..file6) + first half of the 6th outputting_offset file
        # (file7) precede the boundary -> that many frames/ms of negative offset.
        expected_shift_frames = FRAME_COUNT * 7
        expected_shift_ms = DURATION_MS * 7
        self.assertEqual(shot.start_frame, -expected_shift_frames)
        self.assertEqual(shot.start_ms, -expected_shift_ms)
        self.assertEqual(shot.start_frame + shot.frame_count, SPLIT_FRAME)
        self.assertEqual(shot.end_ms, round(1000.0 * SPLIT_FRAME / FPS))

        # The tail of the shot (file7's own first-half frames) keeps its
        # natural, small, non-negative frame_index; the head (file0's frames)
        # is shifted all the way back to -expected_shift_frames.
        self.assertTrue(0 <= shot.focus_samples[-1].frame_index < SPLIT_FRAME)
        self.assertEqual(shot.focus_samples[0].frame_index, -expected_shift_frames)

        # The leftover second half of the boundary file re-seeds the next cycle.
        self.assertEqual(service.segment_raw_x, [])
        self.assertIsNone(service.segment_family)

    def test_second_shot_offsets_are_relative_to_its_own_boundary_file(self):
        service = _service(family_determination_max_seconds=4.0)
        completed = []
        for index in range(20):
            completed.extend(
                service._advance_segment_state(f"file{index}.mp4", _video_info(), _evidence())
            )
        self.assertEqual(len(completed), 2)
        _, shot2 = completed
        self.assertEqual(shot2.source_media, "file15.mp4")
        # leftover(60) + file8, file9 (family det, 120 each) + file10..file14 (5*120) = 900
        expected_shift_frames = 60 + 120 + 120 + 5 * 120
        self.assertEqual(shot2.start_frame, -expected_shift_frames)
        self.assertEqual(shot2.start_frame + shot2.frame_count, SPLIT_FRAME)

    def test_empty_evidence_raises(self):
        service = _service(family_determination_max_seconds=4.0)
        with self.assertRaises(ValueError):
            service._advance_segment_state("empty.mp4", _video_info(), [])


if __name__ == "__main__":
    unittest.main()
