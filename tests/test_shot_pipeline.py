import unittest

from nba_yolo_shot_tagger.config import RuntimeConfig
from nba_yolo_shot_tagger.live import NoopSink
from nba_yolo_shot_tagger.service import ShotFocusService
from nba_yolo_shot_tagger.trajectory import (
    expand_to_source_frames,
    legal_crop_geometry,
    smooth_samples,
)
from nba_yolo_shot_tagger.types import Candidate, FrameEvidence, ModelIdentity, VideoInfo
from nba_yolo_shot_tagger.model import EXPECTED_FAMILIES
from nba_yolo_shot_tagger.reviewed_runtime import ReviewedRound01Runtime, reconstruct_source_frames


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


# These exercise the cut/commit machinery itself, so they pin the tuning they
# were written against instead of tracking whatever the deployed defaults are.
# The deployed values are a separate, and separately reviewable, decision --
# when they moved (commit lag 120 -> 300, min shot 0.25 -> 0.75) these tests
# started asserting against a pipeline that could not reach their conditions.
MECHANISM = dict(
    family_determination_max_seconds=3.0,
    trajectory_commit_lag_frames=120,
    min_shot_seconds=0.25,
)


def _service(input_mode: str, detector=None, **overrides) -> ShotFocusService:
    service = ShotFocusService.__new__(ShotFocusService)
    service.config = RuntimeConfig(verify_model_sha256=False, input_mode=input_mode, **overrides)
    service.model = _FakeModel()
    service.x_sink = NoopSink()
    service.shot_detector = detector
    service._apply_config()
    service._reset_stream()
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
        # The contract is now the frozen reviewed runtime, not the superseded
        # bidirectional botics smoother. Keep the test ID to track the change.
        service = _service("shot_file")
        evidence = _evidence()
        shot = service._consume_file(_FakeVideo("a.mp4"), evidence)[0]
        runtime = ReviewedRound01Runtime(service.config.runtime_manifest_path, 1920, 1080)
        xs, times, events = [], [], []
        for i, frame in enumerate(evidence):
            cs = [{"class_id": EXPECTED_FAMILIES.index(c.family), "family": c.family,
                   "confidence": c.confidence, "bbox_xyxy_norm": list(c.bbox)}
                  for c in frame.candidates]
            t = frame.frame_index / FPS
            x, _, event = runtime.update(cs, t, shot_start=i == 0)
            xs.append(x); times.append(t); events.append(event)
        expected = reconstruct_source_frames(times, xs, events, 0, FRAME_COUNT, 5994, 100)
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
    def test_a_shot_spanning_files_is_emitted_one_tag_per_file(self):
        # This used to be one tag anchored to whichever file was in hand, with
        # a large negative offset. Framing is on the input files now, so the
        # same shot arrives as four tags, each addressed to its own file.
        detector = _FakeDetector({"f3.mp4": [60]})
        service = _service("segment_file", detector, **MECHANISM)
        completed = _feed(service, "f0.mp4", "f1.mp4", "f2.mp4", "f3.mp4")

        self.assertEqual(
            [shot.source_media for shot in completed],
            ["f0.mp4", "f1.mp4", "f2.mp4", "f3.mp4"],
        )
        self.assertEqual(len({shot.shot_id for shot in completed}), 1)
        for shot in completed:
            self.assertEqual(shot.start_frame, 0)
            self.assertEqual(shot.start_ms, 0)
        self.assertEqual(
            [shot.frame_count for shot in completed],
            [FRAME_COUNT, FRAME_COUNT, FRAME_COUNT, 60],
        )
        self.assertEqual(completed[0].focus_samples[0].frame_index, 0)
        self.assertEqual(completed[-1].focus_samples[-1].frame_index, 54)

    def test_leftover_after_a_cut_starts_the_next_shot(self):
        detector = _FakeDetector({"f0.mp4": [60]})
        service = _service("segment_file", detector, **MECHANISM)
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
        service = _service("segment_file", detector, **MECHANISM)
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
        service = _service("segment_file", detector, **MECHANISM)
        completed = _feed(service, "f0.mp4", "f1.mp4")

        # The shot ran from f0's frame 0 to 10 frames before f1 began, so it
        # is f0 that gets the tag -- reported while f1 was being processed.
        self.assertEqual(len(completed), 1)
        shot = completed[0]
        self.assertEqual(shot.source_media, "f0.mp4")
        self.assertEqual(shot.start_frame, 0)
        self.assertEqual(shot.frame_count, FRAME_COUNT - 10)
        # The 10 deferred frames opened the next shot, which f1 continues; that
        # shot owes a tag to each of the two files it touches.
        remaining = service.finalize()
        self.assertEqual([r.source_media for r in remaining], ["f0.mp4", "f1.mp4"])
        self.assertEqual([r.start_frame for r in remaining], [FRAME_COUNT - 10, 0])
        self.assertEqual(sum(r.frame_count for r in remaining), 10 + FRAME_COUNT)

    def test_flush_emits_a_cut_the_detector_was_holding(self):
        detector = _FakeDetector({}, pending=[60])
        service = _service("segment_file", detector, **MECHANISM)
        _feed(service, "f0.mp4")
        completed = service.finalize()
        self.assertEqual(len(completed), 2)
        self.assertEqual(completed[0].frame_count, 60)
        self.assertEqual(completed[1].frame_count, 60)

    def test_short_shot_votes_on_what_it_has(self):
        detector = _FakeDetector({"f0.mp4": [40]})
        service = _service("segment_file", detector, **MECHANISM)
        completed = _feed(service, "f0.mp4")
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].family, "gameplay_follow")
        self.assertEqual(completed[0].frame_count, 40)

    def test_family_is_locked_in_and_not_revoted_mid_shot(self):
        detector = _FakeDetector({"f3.mp4": [60]})
        service = _service("segment_file", detector, **MECHANISM)
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
        service = _service("segment_file", detector, **MECHANISM)
        completed = _feed(service, "f0.mp4", "f1.mp4", "f2.mp4") + service.finalize()
        self.assertEqual(sum(shot.frame_count for shot in completed), 3 * FRAME_COUNT)

    def test_trajectory_is_committed_as_segments_arrive(self):
        service = _service("segment_file", _FakeDetector({}), **MECHANISM)
        _feed(service, "f0.mp4")
        # One file is 2.0s of a 3.0s vote window, so still undecided. The gate
        # is measured on evidence folded, not on the commit horizon, so the
        # commit lag does not push this out any further.
        self.assertIsNone(service.current_family)
        self.assertEqual(service._cycle_smoothed, [])

        _feed(service, "f1.mp4", "f2.mp4")
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
        service = _service("segment_file", detector, **MECHANISM)
        # Files now get their tags as soon as they are complete, so collect
        # every emission, not just the batch the cut lands in.
        emitted = _feed(service, "f0.mp4", "f1.mp4", "f2.mp4")
        committed = list(zip(service._cycle_indices, service._cycle_smoothed))
        self.assertGreater(len(committed), 0)

        emitted += _feed(service, "f3.mp4")
        # np.interp passes exactly through its sample points, so each committed
        # sample must still be readable, unchanged, in the emitted trajectory --
        # now reassembled from the per-file tags it was split across.
        stream = {}
        for shot in emitted:
            base = int(shot.source_media[1]) * FRAME_COUNT + shot.start_frame
            for offset, value in enumerate(shot.x_coordinates):
                stream[base + offset] = value
        for cycle_index, value in committed:
            self.assertAlmostEqual(stream[cycle_index], value, places=9)

    def test_static_family_holds_one_crop_across_segments(self):
        # Reviewed static-family settling is causal and need not be constant.
        # What must survive the joins is one continuous, reference-matching state.
        service = _service("segment_file", _FakeDetector({}),
                           **MECHANISM)
        frames = []
        emitted = []
        for i, path in enumerate(("f0.mp4", "f1.mp4", "f2.mp4", "f3.mp4")):
            evidence = _evidence(family="static_composition")
            frames.extend((i * FRAME_COUNT + f.frame_index, f.candidates) for f in evidence)
            emitted.extend(service._consume_file(_FakeVideo(path), evidence))
        emitted.extend(service.finalize())
        shot = emitted[0]
        self.assertEqual(shot.family, "static_composition")
        # Reassemble the shot from its per-file tags before comparing.
        produced_x = [x for tag in emitted for x in tag.x_coordinates]
        runtime = ReviewedRound01Runtime(service.config.runtime_manifest_path, 1920, 1080)
        xs, times, events = [], [], []
        for i, (fi, candidates) in enumerate(frames):
            cs = [{"class_id": EXPECTED_FAMILIES.index(c.family), "family": c.family,
                   "confidence": c.confidence, "bbox_xyxy_norm": list(c.bbox)}
                  for c in candidates]
            t = fi / FPS
            x, _, event = runtime.update(cs, t, shot_start=i == 0)
            times.append(t); xs.append(x); events.append(event)
        expected = reconstruct_source_frames(times, xs, events, 0, 4 * FRAME_COUNT, 5994, 100)
        self.assertEqual(len(produced_x), len(expected))
        for produced, reference in zip(produced_x, expected):
            self.assertAlmostEqual(produced, float(reference), places=12)

    def test_shot_ids_increment_across_shots(self):
        detector = _FakeDetector({"f0.mp4": [40, 80]})
        service = _service("segment_file", detector, **MECHANISM)
        completed = _feed(service, "f0.mp4")
        self.assertEqual([shot.shot_id for shot in completed], ["shot_000000", "shot_000001"])



class MinimumShotLengthTests(unittest.TestCase):
    def test_a_flickering_transition_does_not_emit_micro_shots(self):
        # TransNetV2 reporting three boundaries a few frames apart across one
        # dissolve. Only the first is a shot boundary; the rest are the same
        # transition, and their frames belong to the shot that follows.
        detector = _FakeDetector({"a.mp4": [], "b.mp4": [10, 12, 16]})
        service = _service("segment_file", detector=detector)
        completed = _feed(service, "a.mp4", "b.mp4")
        # One shot ending at the first cut, not three ending 2 and 4 frames on.
        # It covers both files, so it is written as one tag per file.
        self.assertEqual(len({shot.shot_id for shot in completed}), 1)
        self.assertEqual([c.source_media for c in completed], ["a.mp4", "b.mp4"])
        self.assertEqual(sum(c.frame_count for c in completed), FRAME_COUNT + 10)
        # The absorbed frames are not lost: they open the next shot, which the
        # final flush closes.
        remaining = service.finalize()
        self.assertEqual(
            sum(r.frame_count for r in remaining),
            2 * FRAME_COUNT - (FRAME_COUNT + 10),
        )

    def test_shot_file_boundaries_are_honoured_however_short(self):
        # The caller declared these boundaries; the guard is only a defence
        # against a detector inventing its own.
        service = _service("shot_file")
        self.assertEqual(service._min_shot_seconds, 0.0)

    def test_a_short_final_shot_is_still_emitted(self):
        # Nothing follows the last shot, so there is nothing for it to be
        # absorbed into. It has to come out however short it is.
        detector = _FakeDetector({"a.mp4": [FRAME_COUNT - 4]})
        service = _service("segment_file", detector=detector)
        _feed(service, "a.mp4")
        remaining = service.finalize()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].frame_count, 4)

if __name__ == "__main__":
    unittest.main()


class FileFramedTagTests(unittest.TestCase):
    """Tags are framed on the input files, not on the shots."""

    EARLY = dict(
        family_determination_max_seconds=0.1,
        trajectory_commit_lag_frames=25,
        min_shot_seconds=0.0,
    )

    def _run(self, files, cuts=None):
        service = _service(
            "segment_file", detector=_FakeDetector(cuts or {}), **self.EARLY
        )
        completed = _feed(service, *files)
        completed.extend(service.finalize())
        return completed

    @staticmethod
    def _abs(shot):
        """Absolute [start, end) of a tag, from its own file's position."""
        base = int(shot.source_media.split(".")[0]) * FRAME_COUNT
        return base + shot.start_frame, base + shot.start_frame + shot.frame_count

    def test_one_tag_per_input_file_for_a_shot_that_spans_files(self):
        files = [f"{i}.mp4" for i in range(6)]
        tags = self._run(files)
        # One shot, six segments -> six tags, one addressed to each segment.
        self.assertEqual(len(tags), len(files))
        self.assertEqual([t.source_media for t in tags], files)
        self.assertEqual(len({t.shot_id for t in tags}), 1)
        for tag in tags:
            self.assertEqual(tag.frame_count, FRAME_COUNT)

    def test_offsets_are_relative_to_the_tags_own_file(self):
        # The old framing anchored a late tag to whatever file was being
        # processed, which made its offsets negative. A tag now describes its
        # own file, so they start at zero.
        for tag in self._run([f"{i}.mp4" for i in range(6)]):
            self.assertEqual(tag.start_frame, 0)
            self.assertEqual(tag.start_ms, 0)
            self.assertGreaterEqual(tag.start_frame, 0)

    def test_a_cut_inside_a_file_gives_that_file_two_tags(self):
        tags = self._run(["0.mp4", "1.mp4"], cuts={"1.mp4": [40]})
        by_file = {}
        for tag in tags:
            by_file.setdefault(tag.source_media, []).append(tag)
        self.assertEqual(len(by_file["1.mp4"]), 2)
        first, second = by_file["1.mp4"]
        self.assertNotEqual(first.shot_id, second.shot_id)
        self.assertEqual((first.start_frame, first.frame_count), (0, 40))
        self.assertEqual((second.start_frame, second.frame_count), (40, FRAME_COUNT - 40))

    def test_shot_file_mode_is_one_tag_per_file(self):
        service = _service("shot_file")
        tags = _feed(service, "0.mp4", "1.mp4", "2.mp4")
        self.assertEqual([t.source_media for t in tags], ["0.mp4", "1.mp4", "2.mp4"])
        for tag in tags:
            self.assertEqual((tag.start_frame, tag.frame_count), (0, FRAME_COUNT))
            self.assertTrue(tag.is_final)

    def test_the_tags_tile_the_stream_exactly(self):
        tags = self._run([f"{i}.mp4" for i in range(6)], cuts={"3.mp4": [40]})
        spans = sorted(self._abs(t) for t in tags)
        self.assertEqual(spans[0][0], 0)
        for (_, end), (start, _) in zip(spans, spans[1:]):
            self.assertEqual(end, start)  # no gap, no overlap
        self.assertEqual(spans[-1][1], 6 * FRAME_COUNT)

    def test_a_delayed_tag_still_names_its_own_segment(self):
        # Nothing is final for several files, so the first round of tags is
        # written long after those segments arrived -- still addressed to them.
        service = _service(
            "segment_file",
            detector=_FakeDetector({}),
            family_determination_max_seconds=3.0,
            trajectory_commit_lag_frames=120,
            min_shot_seconds=0.0,
        )
        self.assertEqual(_feed(service, "0.mp4", "1.mp4"), [])
        later = _feed(service, "2.mp4", "3.mp4")
        self.assertTrue(later)
        self.assertEqual(later[0].source_media, "0.mp4")
        self.assertEqual(later[0].start_frame, 0)

    def test_only_the_last_tag_of_a_shot_is_final(self):
        tags = self._run([f"{i}.mp4" for i in range(4)])
        self.assertEqual([t.is_final for t in tags], [False, False, False, True])
        self.assertEqual([t.part_index for t in tags], [0, 1, 2, 3])

    def test_no_frame_is_emitted_before_its_value_is_final(self):
        service = _service("segment_file", detector=_FakeDetector({}), **self.EARLY)
        _feed(service, "0.mp4", "1.mp4", "2.mp4")
        last_committed = service._cycle_start_abs + service._cycle_indices[-1] + 1
        self.assertLessEqual(service._emitted_abs, last_committed)

    def test_values_match_what_a_whole_shot_pass_would_produce(self):
        # File framing changes which tag a frame is written into, never its X.
        files = [f"{i}.mp4" for i in range(5)]
        framed = [x for t in self._run(files) for x in t.x_coordinates]
        service = _service("segment_file", detector=_FakeDetector({}), **self.EARLY)
        for path in files:
            service._consume_file(_FakeVideo(path), _evidence())
        service._advance(service._abs_frames - service._cycle_start_abs, force_family=True)
        reference = list(service._cycle_smoothed)
        self.assertEqual(len(framed), 5 * FRAME_COUNT)
        # Every committed sample value must appear in the framed output.
        self.assertTrue(set(round(v, 9) for v in reference) <= set(round(v, 9) for v in framed))


class LiveSinkTests(unittest.TestCase):
    """The sink and the tags must be the same data, in the same order."""

    class _Recorder:
        def __init__(self): self.values = []
        def publish(self, x_vals): self.values.extend(x_vals)

    def _run(self, mode, files, cuts=None, **overrides):
        service = _service(
            mode, detector=_FakeDetector(cuts or {}) if mode == "segment_file" else None,
            **overrides,
        )
        sink = self._Recorder()
        service.x_sink = sink
        tags = _feed(service, *files)
        tags.extend(service.finalize())
        # What a consumer of out.jsonl actually reads, rounding included.
        from nba_yolo_shot_tagger.contract import tags_for_analysis
        serialized = [
            x
            for analysis in tags
            for x in tags_for_analysis(analysis, service.config)[0].additional_info[
                "x-coordinates"
            ]
        ]
        return tags, sink.values, serialized

    def test_the_sink_gets_one_value_per_source_frame(self):
        files = [f"{i}.mp4" for i in range(4)]
        tags, published, serialized = self._run(
            "segment_file", files,
            family_determination_max_seconds=0.1,
            trajectory_commit_lag_frames=25,
            min_shot_seconds=0.0,
        )
        self.assertEqual(len(published), len(files) * FRAME_COUNT)
        # Exactly what the tags carry on the wire, not merely close to it.
        self.assertEqual(published, serialized)

    def test_shot_file_mode_too(self):
        tags, published, serialized = self._run("shot_file", ["0.mp4", "1.mp4"])
        self.assertEqual(len(published), 2 * FRAME_COUNT)
        self.assertEqual(published, serialized)

    def test_a_cut_does_not_duplicate_or_drop_a_frame(self):
        files = [f"{i}.mp4" for i in range(4)]
        tags, published, serialized = self._run(
            "segment_file", files, cuts={"2.mp4": [40]},
            family_determination_max_seconds=0.1,
            trajectory_commit_lag_frames=25,
            min_shot_seconds=0.0,
        )
        self.assertEqual(len(published), len(files) * FRAME_COUNT)
        self.assertEqual(published, serialized)


class EncodeXTests(unittest.TestCase):
    def test_it_clamps_instead_of_losing_the_segment(self):
        # encode_x runs inside the per-file commit, so raising would abort the
        # whole segment rather than drop one sample.
        from nba_yolo_shot_tagger.live import encode_x
        self.assertEqual(encode_x(0.613), encode_x(0.613))
        self.assertEqual(encode_x(-0.001), encode_x(0.0))
        self.assertEqual(encode_x(1.001), encode_x(1.0))
        self.assertEqual(encode_x(float("nan")), encode_x(0.5))
        self.assertEqual(encode_x(float("inf")), encode_x(0.5))

