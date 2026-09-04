import unittest

from nba_yolo_shot_tagger.config import RuntimeConfig
from nba_yolo_shot_tagger.service import ShotFocusService
from nba_yolo_shot_tagger.trajectory import (
    expand_to_source_frames,
    legal_crop_geometry,
    smooth_samples,
)
from nba_yolo_shot_tagger.types import Candidate, FrameEvidence, ModelIdentity, VideoInfo


FPS = 59.94
FRAME_COUNT = 120  # ~2.002s at FPS
DURATION_MS = round(1000.0 * FRAME_COUNT / FPS)


def _video_info() -> VideoInfo:
    return VideoInfo(fps=FPS, frame_count=FRAME_COUNT, width=1920, height=1080, duration_ms=DURATION_MS)


def _evidence(family: str = "gameplay_follow", confidence: float = 0.8):
    # Sample every 6th frame, matching ~10fps sampling of a 59.94fps source.
    return [
        FrameEvidence(
            frame_index=index,
            timestamp_ms=round(1000.0 * index / FPS),
            candidates=(Candidate(family=family, confidence=confidence, bbox=(0.2, 0.2, 0.4, 0.8)),),
        )
        for index in range(0, FRAME_COUNT, 6)
    ]


class _FakeVideo:
    def __init__(self, path: str) -> None:
        self.path = path
        self.info = _video_info()


class _FakeDetector:
    """Stands in for StreamShotDetector: cuts relative to the pushed file.

    Negative entries model a cut the real detector deferred until it had
    frames after it, and so reports during the following file.
    """

    def __init__(self, cuts_by_file, pending=()) -> None:
        self.cuts_by_file = cuts_by_file
        self.pending = list(pending)
        self.pushed = []

    def push(self, path):
        self.pushed.append(path)
        return sorted(self.cuts_by_file.get(path, []))

    def flush(self):
        return sorted(self.pending)


class _FakeModel:
    identity = ModelIdentity(path="fake.pt", sha256="a" * 64, artifact_version="test", class_names=())


def _service(input_mode: str, detector=None, **overrides) -> ShotFocusService:
    service = ShotFocusService.__new__(ShotFocusService)
    service.config = RuntimeConfig(verify_model_sha256=False, input_mode=input_mode, **overrides)
    service.model = _FakeModel()
    service.shot_detector = detector
    service._commit_lag = (
        service.config.trajectory_commit_lag_frames if input_mode == "segment_file" else 0
    )
    service._shot_index = 0
    service._abs_frames = 0
    service._abs_ms = 0
    service._last_source_media = None
    service._last_video_info = None
    service._last_abs_start = 0
    service._last_abs_start_ms = 0
    service._reset_cycle(0, 0)
    return service


def _feed(service: ShotFocusService, *paths: str):
    completed = []
    for path in paths:
        completed.extend(service._consume_file(_FakeVideo(path), _evidence()))
    return completed


class ShotFileModeTests(unittest.TestCase):
    def test_each_file_is_exactly_one_shot(self):
        service = _service("shot_file")
        completed = _feed(service, "a.mp4", "b.mp4")
        self.assertEqual(len(completed), 2)
        for shot, path in zip(completed, ("a.mp4", "b.mp4")):
            self.assertEqual(shot.source_media, path)
            self.assertEqual(shot.start_frame, 0)
            self.assertEqual(shot.start_ms, 0)
            self.assertEqual(shot.frame_count, FRAME_COUNT)
            self.assertEqual(shot.end_ms, DURATION_MS)
            self.assertEqual(shot.family, "gameplay_follow")
        self.assertEqual(service.finalize(), [])

    def test_output_still_matches_whole_shot_smoothing(self):
        # shot_file cuts at every file end, so a shot is never committed in
        # pieces and must come out exactly as a single whole-shot pass.
        service = _service("shot_file")
        evidence = _evidence()
        shot = service._consume_file(_FakeVideo("a.mp4"), evidence)[0]

        indices, raw_x, confidences, _ = service._select_focus_series(evidence, shot.family)
        _, legal_min, legal_max = legal_crop_geometry(
            1920, 1080, service.config.target_aspect_width_over_height
        )
        sample_x = smooth_samples(
            family=shot.family,
            raw_x=raw_x,
            confidences=confidences,
            inference_fps=service.config.inference_fps,
            legal_min=legal_min,
            legal_max=legal_max,
        )
        expected = expand_to_source_frames(
            sample_frame_indices=indices,
            sample_x=sample_x,
            start_frame=0,
            end_frame=FRAME_COUNT,
            legal_min=legal_min,
            legal_max=legal_max,
        )
        self.assertEqual(len(shot.x_coordinates), len(expected))
        for produced, reference in zip(shot.x_coordinates, expected):
            self.assertAlmostEqual(produced, float(reference), places=12)

    def test_family_is_voted_per_file(self):
        service = _service("shot_file")
        _feed(service, "a.mp4")
        second = service._consume_file(
            _FakeVideo("b.mp4"), _evidence(family="active_speaker", confidence=0.9)
        )
        self.assertEqual(second[0].family, "active_speaker")


class SegmentFileModeTests(unittest.TestCase):
    def test_shot_spanning_files_closes_with_negative_offsets(self):
        detector = _FakeDetector({"f3.mp4": [60]})
        service = _service("segment_file", detector, family_determination_max_seconds=3.0)
        completed = _feed(service, "f0.mp4", "f1.mp4", "f2.mp4", "f3.mp4")

        self.assertEqual(len(completed), 1)
        shot = completed[0]
        self.assertEqual(shot.source_media, "f3.mp4")
        self.assertEqual(shot.start_frame, -FRAME_COUNT * 3)
        self.assertLess(shot.start_ms, 0)
        self.assertEqual(shot.start_frame + shot.frame_count, 60)
        self.assertEqual(shot.focus_samples[-1].frame_index, 54)
        self.assertEqual(shot.focus_samples[0].frame_index, -FRAME_COUNT * 3)

    def test_leftover_after_a_cut_starts_the_next_shot(self):
        detector = _FakeDetector({"f0.mp4": [60]})
        service = _service("segment_file", detector, family_determination_max_seconds=3.0)
        _feed(service, "f0.mp4")

        completed = service.finalize()
        self.assertEqual(len(completed), 1)
        shot = completed[0]
        self.assertEqual(shot.source_media, "f0.mp4")
        self.assertEqual(shot.start_frame, 60)
        self.assertEqual(shot.frame_count, 60)
        self.assertEqual(shot.focus_samples[0].frame_index, 60)

    def test_several_shots_inside_one_file(self):
        detector = _FakeDetector({"f0.mp4": [40, 80]})
        service = _service("segment_file", detector, family_determination_max_seconds=3.0)
        completed = _feed(service, "f0.mp4")

        self.assertEqual(len(completed), 2)
        first, second = completed
        self.assertEqual((first.start_frame, first.frame_count), (0, 40))
        self.assertEqual((second.start_frame, second.frame_count), (40, 40))
        tail = service.finalize()
        self.assertEqual(len(tail), 1)
        self.assertEqual((tail[0].start_frame, tail[0].frame_count), (80, 40))

    def test_deferred_cut_lands_in_the_previous_file(self):
        # The detector could not see past f0's tail until f1 arrived, so it
        # reports the cut 10 frames before f1 starts.
        detector = _FakeDetector({"f1.mp4": [-10]})
        service = _service("segment_file", detector, family_determination_max_seconds=3.0)
        completed = _feed(service, "f0.mp4", "f1.mp4")

        self.assertEqual(len(completed), 1)
        shot = completed[0]
        self.assertEqual(shot.source_media, "f1.mp4")
        # The shot ran from f0's frame 0 to 10 frames before f1 began.
        self.assertEqual(shot.start_frame, -FRAME_COUNT)
        self.assertEqual(shot.end_ms, round(1000.0 * -10 / FPS))
        self.assertEqual(shot.frame_count, FRAME_COUNT - 10)
        # The 10 deferred frames opened the next shot, which f1 continues.
        remaining = service.finalize()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].frame_count, 10 + FRAME_COUNT)

    def test_flush_emits_a_cut_the_detector_was_holding(self):
        detector = _FakeDetector({}, pending=[60])
        service = _service("segment_file", detector, family_determination_max_seconds=3.0)
        _feed(service, "f0.mp4")
        completed = service.finalize()
        self.assertEqual(len(completed), 2)
        self.assertEqual(completed[0].frame_count, 60)
        self.assertEqual(completed[1].frame_count, 60)

    def test_short_shot_votes_on_what_it_has(self):
        detector = _FakeDetector({"f0.mp4": [40]})
        service = _service("segment_file", detector, family_determination_max_seconds=3.0)
        completed = _feed(service, "f0.mp4")
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].family, "gameplay_follow")
        self.assertEqual(completed[0].frame_count, 40)

    def test_family_is_locked_in_and_not_revoted_mid_shot(self):
        detector = _FakeDetector({"f3.mp4": [60]})
        service = _service("segment_file", detector, family_determination_max_seconds=3.0)
        _feed(service, "f0.mp4", "f1.mp4", "f2.mp4")
        self.assertEqual(service.shot_state, "outputting_offset")
        self.assertEqual(service.current_family, "gameplay_follow")

        completed = service._consume_file(
            _FakeVideo("f3.mp4"), _evidence(family="active_speaker", confidence=0.99)
        )
        self.assertEqual(completed[0].family, "gameplay_follow")

    def test_finalize_is_a_noop_without_input(self):
        service = _service("segment_file", _FakeDetector({}))
        self.assertEqual(service.finalize(), [])

    def test_shots_tile_the_stream_without_gaps(self):
        detector = _FakeDetector({"f0.mp4": [40], "f2.mp4": [-5, 90]})
        service = _service("segment_file", detector, family_determination_max_seconds=3.0)
        completed = _feed(service, "f0.mp4", "f1.mp4", "f2.mp4") + service.finalize()
        self.assertEqual(sum(shot.frame_count for shot in completed), 3 * FRAME_COUNT)

    def test_trajectory_is_committed_as_segments_arrive(self):
        service = _service("segment_file", _FakeDetector({}), family_determination_max_seconds=3.0)
        _feed(service, "f0.mp4", "f1.mp4")
        # Still inside the vote threshold plus the commit lag, so undecided.
        self.assertIsNone(service.current_family)
        self.assertEqual(service._cycle_smoothed, [])

        _feed(service, "f2.mp4")
        # The family is now settled and the backlog has been caught up,
        # without waiting for the shot to end.
        self.assertEqual(service.current_family, "gameplay_follow")
        self.assertEqual(len(service._cycle_smoothed), len(service._cycle_indices))
        self.assertGreater(len(service._cycle_smoothed), 0)
        # ... but it stays the commit lag behind, so a late cut cannot reach it.
        horizon = service._abs_frames - service._commit_lag - service._cycle_start_abs
        self.assertLess(max(service._cycle_indices), horizon)
        self.assertGreater(len(service._cycle_pending), 0)

    def test_committed_trajectory_survives_a_late_cut(self):
        detector = _FakeDetector({"f3.mp4": [-10]})
        service = _service("segment_file", detector, family_determination_max_seconds=3.0)
        _feed(service, "f0.mp4", "f1.mp4", "f2.mp4")
        committed = list(zip(service._cycle_indices, service._cycle_smoothed))
        self.assertGreater(len(committed), 0)

        shot = _feed(service, "f3.mp4")[0]
        # np.interp passes exactly through its sample points, so each committed
        # sample must still be readable, unchanged, in the emitted trajectory.
        for cycle_index, value in committed:
            self.assertAlmostEqual(shot.x_coordinates[cycle_index], value, places=9)

    def test_static_family_holds_one_crop_across_segments(self):
        service = _service(
            "segment_file", _FakeDetector({}), family_determination_max_seconds=3.0
        )
        for path in ("f0.mp4", "f1.mp4", "f2.mp4", "f3.mp4"):
            service._consume_file(_FakeVideo(path), _evidence(family="static_composition"))
        shot = service.finalize()[0]
        self.assertEqual(shot.family, "static_composition")
        # Committed in several batches, but still a single locked crop.
        self.assertEqual(len(set(shot.x_coordinates)), 1)

    def test_shot_ids_increment_across_shots(self):
        detector = _FakeDetector({"f0.mp4": [40, 80]})
        service = _service("segment_file", detector, family_determination_max_seconds=3.0)
        completed = _feed(service, "f0.mp4")
        self.assertEqual([shot.shot_id for shot in completed], ["shot_000000", "shot_000001"])


if __name__ == "__main__":
    unittest.main()
