from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import RuntimeConfig, parse_hold_and_cut_families, resolve_runtime_path
from .model import EXPECTED_FAMILIES, YoloStudentModel
from .reviewed_runtime import ReviewedRound01Runtime
from .trajectory import (
    STATIC_FAMILIES,
    expand_to_source_frames,
    legal_crop_geometry,
    smooth_samples,
)
from .types import Candidate, FocusSample, FrameEvidence, ShotAnalysis, VideoInfo
from .video import VideoSource
from .live import VerticalSink


CATEGORY_BY_FAMILY = {
    "active_speaker": "talking",
    "gameplay_follow": "gameplay",
    "graphic_text_lock": "graphic",
    "person_subject": "person",
    "safe_center": "other",
    "split_screen": "split_screen",
    "static_composition": "static",
}

# Already-committed samples fed back in as left context when smoothing the
# next batch, so a batch boundary does not show up as a kink. The filters are
# Python loops, so this also keeps the per-segment cost constant instead of
# growing with the length of the open shot.
SMOOTHING_CONTEXT_SAMPLES = 64

# Focus selection is otherwise a memoryless argmax, which makes it decided by
# whichever near-tied box the detector happened to score highest. That is not a
# hypothetical: running the same frames in fp16 and fp32 flips the winner on a
# small number of them, once moving the crop by 0.34 of the frame off a
# confidence difference of 1e-4 (0.1366 against 0.1365). Between consecutive
# samples of a real shot the same tie-break fires far more often.
#
# So among the boxes that are statistically tied with the best one, prefer the
# one nearest to where the crop already is. A candidate must reach this
# fraction of the best confidence to count as tied -- below it the detector has
# a real preference and continuity must not override it.
FOCUS_TIE_FRACTION = 0.75
# Beyond this normalized distance the "continuity" would be fictional: the
# nearest tied box is nowhere near the previous one, so take the best instead.
FOCUS_LINK_MAX_DISTANCE = 0.25
# Samples of absence after which the anchor is stale and linking stops. At the
# default 10 fps inference rate this is one second.
FOCUS_LINK_MAX_GAP_SAMPLES = 10


class ShotFocusService:
    """One pipeline for both input modes.

    Every input file is decoded and inferred exactly once and folded onto a
    single absolute stream axis. Shots are cut out of that axis; the only
    difference between the modes is where the cuts come from:

    ``segment_file``
        TransNetV2 over a rolling buffer that carries frames across the joins
        between files (see :mod:`shot_boundaries`). A shot may span several
        files, and one file may contain several shots.
    ``shot_file``
        Each file already is one shot, so a cut is placed at the end of it.

    A shot's family is voted on once ``family_determination_max_seconds`` of it
    has been seen (or on whatever it has, if it is cut before that). From then
    on its X trajectory is worked out as the segments arrive rather than in one
    pass at the end: each segment catches up whatever the family vote was
    waiting on and then keeps pace. Committed X values are final, because the
    pipeline stays ``trajectory_commit_lag_frames`` behind the frames it has
    folded -- clearing the margin shot detection needs before it will commit a
    cut -- so a cut reported late never lands in already-committed
    trajectory.

    Because a shot is emitted against the file being processed when it is cut,
    everything it saw earlier is expressed as a negative offset from that
    file's frame 0.
    """

    def __init__(self, config: RuntimeConfig, vertical_sink: VerticalSink) -> None:
        self.config = config
        self.model = YoloStudentModel(
            model_path=config.model_path,
            manifest_path=config.model_manifest_path,
            verify_sha256=config.verify_model_sha256,
            device=config.device,
            imgsz=config.imgsz,
            min_confidence=config.min_detection_confidence,
            iou=config.iou,
            max_detections=config.max_detections,
            top_k=config.top_k,
            use_fp16=config.use_fp16,
        )

        self.x_sink = vertical_sink

        if config.input_mode == "segment_file":
            # Imported here so shot_file mode never has to load torch/TransNet.
            from .shot_boundaries import StreamShotDetector

            self.shot_detector = StreamShotDetector(
                config.shot_model_path,
                device="cpu" if config.device == "cpu" else None,
            )
        else:
            self.shot_detector = None
        self._apply_config()

        self._shot_index = 0
        self._abs_frames = 0  # source frames folded so far, across all files
        self._abs_ms = 0
        self._last_source_media: Optional[str] = None
        self._last_video_info: Optional[VideoInfo] = None
        self._last_abs_start = 0
        self._last_abs_start_ms = 0
        self._reset_cycle(0, 0)

    def _apply_config(self) -> None:
        """Everything derived from RuntimeConfig, in one place.

        Kept separate from __init__ because __init__ also loads a checkpoint
        and, in segment_file mode, TransNetV2. Tests build the service through
        __new__ to skip that, and they should not have to know which derived
        fields exist -- adding one here must not break them.
        """
        config = self.config
        segmenting = config.input_mode == "segment_file"
        # Only segment_file defers cuts, so only it has to hold back...
        self._commit_lag = config.trajectory_commit_lag_frames if segmenting else 0
        # ...and only it invents its own boundaries, so only it has to defend
        # against a flickering one.
        self._min_shot_seconds = config.min_shot_seconds if segmenting else 0.0
        self._hold_and_cut_families = parse_hold_and_cut_families(
            config.hold_and_cut_families
        )

    @property
    def shot_state(self) -> str:
        return "determining_family" if self.current_family is None else "outputting_offset"

    # ------------------------------------------------------------------
    # public entry points
    # ------------------------------------------------------------------

    def analyze_file(self, source_media: str) -> List[ShotAnalysis]:
        """Consume one input file, returning every shot cut while doing so."""
        video = VideoSource(source_media)
        evidence = self._infer_file(video)
        return self._consume_file(video, evidence)

    def finalize(self) -> List[ShotAnalysis]:
        """Close out the stream once there are no more input files.

        The detector is holding back any cut it could not yet see past, and
        the last shot has no closing cut at all, so both are settled here.
        """
        if self._last_video_info is None:
            return []
        completed: List[ShotAnalysis] = []
        if self.shot_detector is not None:
            for relative_cut in self.shot_detector.flush():
                analysis = self._close(self._last_abs_start + relative_cut)
                if analysis is not None:
                    completed.append(analysis)
        analysis = self._close(self._abs_frames, force=True)
        if analysis is not None:
            completed.append(analysis)
        return completed

    # ------------------------------------------------------------------
    # per-file processing
    # ------------------------------------------------------------------

    def _infer_file(self, video: VideoSource) -> List[FrameEvidence]:
        evidence: List[FrameEvidence] = []
        for frame_indices, frames in video.iter_sample_batches(
            start_frame=0,
            end_frame=video.info.frame_count,
            inference_fps=self.config.inference_fps,
            batch_size=self.config.batch_size,
        ):
            evidence.extend(
                self.model.infer_batch(
                    frame_indices=frame_indices,
                    frames=frames,
                    source_fps=video.info.fps,
                )
            )
        if not evidence:
            raise ValueError(f"{video.path} produced no decodable frames")
        return evidence

    def _cuts_for(self, video: VideoSource, abs_start: int) -> List[int]:
        """Absolute stream frames at which a new shot starts."""
        if self.shot_detector is None:
            # shot_file: the file is the shot, so it is cut at the file end.
            return [abs_start + video.info.frame_count]
        # Cuts come back relative to this file's start, and are negative when
        # they fall in the tail the detector had deferred until now.
        return [abs_start + cut for cut in self.shot_detector.push(video.path)]

    def _consume_file(
        self, video: VideoSource, evidence: Sequence[FrameEvidence]
    ) -> List[ShotAnalysis]:
        info = video.info
        abs_start = self._abs_frames
        abs_start_ms = self._abs_ms
        self._last_source_media = video.path
        self._last_video_info = info
        self._last_abs_start = abs_start
        self._last_abs_start_ms = abs_start_ms

        cuts = self._cuts_for(video, abs_start)
        self._fold(evidence, info, abs_start, abs_start_ms)

        # Cuts first: everything committed so far sits behind the oldest cut
        # this round can report, so none of it has to be revisited.
        completed: List[ShotAnalysis] = []
        for cut in cuts:
            analysis = self._close(cut)
            if analysis is not None:
                completed.append(analysis)

        # Then catch up / keep up on the shot that is still open.
        self._advance(self._abs_frames - self._commit_lag - self._cycle_start_abs)

        # ...and write out what that settled, if the caller asked for tags
        # before the shot ends.
        partial = self._flush_partial()
        if partial is not None:
            completed.append(partial)

        if (self._abs_ms - self._cycle_start_abs_ms) / 1000.0 >= self.config.max_shot_seconds:
            # Safety valve: an open shot is held in memory, and a stream can
            # run a long way without a detected cut.
            analysis = self._close(self._abs_frames, force=True)
            if analysis is not None:
                completed.append(analysis)
        return completed

    def _fold(
        self,
        evidence: Sequence[FrameEvidence],
        info: VideoInfo,
        abs_start: int,
        abs_start_ms: int,
    ) -> None:
        """Add a file's evidence to the open shot, on that shot's own axis."""
        frame_base = abs_start - self._cycle_start_abs
        ms_base = abs_start_ms - self._cycle_start_abs_ms
        self._cycle_pending.extend(
            replace(
                frame,
                frame_index=frame.frame_index + frame_base,
                timestamp_ms=frame.timestamp_ms + ms_base,
            )
            for frame in evidence
        )
        self._abs_frames = abs_start + info.frame_count
        self._abs_ms = abs_start_ms + info.duration_ms

    def _emit_span(self, span_end: int, *, is_final: bool) -> Optional[ShotAnalysis]:
        """Emit committed trajectory for [already emitted, ``span_end``).

        ``span_end`` is on the open shot's own frame axis. Everything is
        anchored to the file currently being processed, exactly as a whole-shot
        emission is, so a span that began before that file simply carries a
        more negative offset.
        """
        span_start = self._cycle_emitted_frames
        if span_end <= span_start or self.current_family is None:
            return None
        info = self._last_video_info
        anchor_abs = self._last_abs_start
        anchor_abs_ms = self._last_abs_start_ms

        shot_start_frame = self._cycle_start_abs - anchor_abs
        shot_start_ms = self._cycle_start_abs_ms - anchor_abs_ms
        start_frame = shot_start_frame + span_start
        end_frame = shot_start_frame + span_end
        start_ms = shot_start_ms + _frame_to_ms(span_start, info.fps)
        # A sub-millisecond span must still be a non-empty interval.
        end_ms = max(_frame_to_ms(end_frame, info.fps), start_ms + 1)

        # The whole committed sample series is handed over, not just this
        # span's slice: interpolation at the left edge needs the sample before
        # the span, and expand_to_source_frames takes only the frames asked for.
        analysis = self._build_shot_analysis(
            source_media=self._last_source_media,
            shot_id=f"shot_{self._shot_index:06d}",
            start_ms=start_ms,
            end_ms=end_ms,
            start_frame=start_frame,
            end_frame=end_frame,
            info=info,
            sample_frame_indices=[
                index + shot_start_frame for index in self._cycle_indices
            ],
            sample_x=self._cycle_smoothed,
            sample_events=self._cycle_events,
            focus_samples=[
                replace(
                    sample,
                    frame_index=sample.frame_index + shot_start_frame,
                    timestamp_ms=sample.timestamp_ms + shot_start_ms,
                )
                for sample in self._cycle_focus
                if span_start <= sample.frame_index < span_end
            ],
            part_index=self._cycle_parts,
            is_final=is_final,
        )
        self._cycle_emitted_frames = span_end
        self._cycle_parts += 1
        return analysis

    def _flush_partial(self) -> Optional[ShotAnalysis]:
        """Write out the open shot's committed prefix, if it has grown enough.

        Stops at the last committed sample rather than at the commit horizon:
        the frames after it interpolate toward a sample that has not arrived,
        so their values are not yet final and must not be published.
        """
        if self.config.max_tag_latency_seconds <= 0.0:
            return None
        if self.current_family is None or not self._cycle_indices:
            return None
        info = self._last_video_info
        span_end = self._cycle_indices[-1] + 1
        pending_ms = _frame_to_ms(span_end - self._cycle_emitted_frames, info.fps)
        if pending_ms < 1000.0 * self.config.max_tag_latency_seconds:
            return None
        return self._emit_span(span_end, is_final=False)

    def _close(self, cut_abs: int, *, force: bool = False) -> Optional[ShotAnalysis]:
        """Cut the open shot at an absolute stream frame and emit it.

        The shot is emitted against the file currently being processed, so a
        cut that lands before that file starts (one the detector deferred)
        simply produces a more negative offset.

        A cut that would carve off less than ``min_shot_seconds`` is dropped
        instead: a two-frame shot is a flickering transition, not a shot, and
        emitting one costs a family voted on two frames of evidence and a
        30 ms crop instruction for the consumer to honour. The frames are not
        lost -- the cycle stays open and they join the following shot. ``force``
        is for closes that are not cuts at all (the final flush, the
        max_shot_seconds valve), where there is no following shot to join.
        """
        length = cut_abs - self._cycle_start_abs
        if length <= 0:
            return None
        info = self._last_video_info
        if (
            not force
            and _frame_to_ms(length, info.fps) < 1000.0 * self._min_shot_seconds
        ):
            return None
        anchor_abs = self._last_abs_start
        anchor_abs_ms = self._last_abs_start_ms

        # Settle the rest of this shot: a family if it was cut before the vote
        # was due, and the trajectory for whatever was still being held back.
        self._advance(length, force_family=True)

        # Whatever of this shot has not already been written out. With
        # max_tag_latency_seconds off that is the entire shot, which is the
        # historical single-tag emission.
        analysis = self._emit_span(length, is_final=True)
        self._shot_index += 1

        end_frame = cut_abs - anchor_abs
        start_ms = self._cycle_start_abs_ms - anchor_abs_ms
        end_ms = max(_frame_to_ms(end_frame, info.fps), start_ms + 1)

        # Whatever came after the cut opens the next shot. It was never
        # committed, so it carries over as raw evidence and is re-decided.
        length_ms = _frame_to_ms(length, info.fps)
        remainder = [
            replace(
                frame,
                frame_index=frame.frame_index - length,
                timestamp_ms=frame.timestamp_ms - length_ms,
            )
            for frame in self._cycle_pending
        ]
        self._reset_cycle(cut_abs, anchor_abs_ms + end_ms, remainder)
        return analysis

    # ------------------------------------------------------------------
    # open-shot state
    # ------------------------------------------------------------------

    def _reset_cycle(
        self,
        start_abs: int,
        start_abs_ms: int,
        pending: Optional[List[FrameEvidence]] = None,
    ) -> None:
        self._cycle_start_abs = start_abs
        self._cycle_start_abs_ms = start_abs_ms
        # Evidence past the commit horizon, still waiting to be decided.
        self._cycle_pending: List[FrameEvidence] = pending or []
        # Committed, parallel, on this shot's own frame axis.
        self._cycle_indices: List[int] = []
        self._cycle_raw_x: List[Optional[float]] = []
        self._cycle_confidences: List[float] = []
        self._cycle_focus: List[FocusSample] = []
        self._cycle_smoothed: List[float] = []
        # One reviewed runtime per shot. A segment/batch boundary is not a reset.
        self._cycle_reviewed_runtime: Optional[ReviewedRound01Runtime] = None
        self._cycle_reviewed_geometry: Optional[Tuple[int, int, float]] = None
        # Container metadata can report tiny FPS differences between adjacent
        # chunks of the same continuous stream. Freeze the reviewed controller
        # clock to the FPS of the first segment in the shot rather than letting
        # those metadata fluctuations alter controller time.
        self._cycle_reviewed_fps: Optional[float] = None
        self._cycle_reviewed_last_frame: Optional[int] = None
        self._cycle_events: List[int] = []
        # Shot-axis frames already written out as tags. Non-zero only when
        # max_tag_latency_seconds is flushing an open shot in pieces.
        self._cycle_emitted_frames = 0
        self._cycle_parts = 0
        self._cycle_static_lock: Optional[float] = None
        # Where the focus was last actually observed, and how many samples ago,
        # so association survives a batch boundary the way smoothing context
        # does. Cleared here because a new shot has no continuity with the old.
        self._cycle_focus_anchor: Optional[float] = None
        self._cycle_focus_gap = 0
        self.current_family: Optional[str] = None
        self.current_family_confidence: Optional[float] = None

    def _advance(self, limit: int, force_family: bool = False) -> None:
        """Commit the open shot's trajectory for everything before ``limit``.

        ``limit`` is a frame position on the shot's own axis. Evidence beyond
        it stays pending: it is inside the margin where a cut can still be
        reported, so deciding it now could mean redoing it.
        """
        ready = [frame for frame in self._cycle_pending if frame.frame_index < limit]
        if self.current_family is None:
            seen_seconds = _frame_to_ms(limit, self._last_video_info.fps) / 1000.0
            if not force_family and seen_seconds < self.config.family_determination_max_seconds:
                return
            self.current_family, self.current_family_confidence = self._family_vote(ready)
        if not ready:
            return
        self._cycle_pending = [
            frame for frame in self._cycle_pending if frame.frame_index >= limit
        ]

        # The locked family remains botics metadata. It must not filter the
        # evidence sent to the reviewed per-observation controller.
        new_smoothed_samples = self._reviewed_samples(ready)
        self._cycle_smoothed.extend(new_smoothed_samples)

        self.x_sink.publish(new_smoothed_samples)

        ## write out new_smoothed_samples here
        ## it will be some amount of data, equivalent to a segment in the steady state, but there may be more or less if the shot is just starting or ending. 
        
    def _reviewed_samples(self, evidence: Sequence[FrameEvidence]) -> List[float]:
        """Commit each new observation exactly once through the frozen v4 runtime.

        Use the shot-relative source-frame clock, as v4 does, rather than rounded
        metadata milliseconds. Botics' serialized timestamps/offsets are unchanged.
        No new look-ahead buffer and no legacy smoother or target selector here.
        """
        info = self._last_video_info
        if info is None or not math.isfinite(info.fps) or info.fps <= 0:
            raise ValueError("Reviewed runtime requires valid video geometry/FPS")
        _, legal_min, legal_max = legal_crop_geometry(
            info.width, info.height, self.config.target_aspect_width_over_height
        )
        geometry = (
            info.width,
            info.height,
            self.config.target_aspect_width_over_height,
        )

        if (
            self._cycle_reviewed_geometry is not None
            and self._cycle_reviewed_geometry != geometry
        ):
            raise ValueError(
                "Video geometry/FPS changed inside an open reviewed shot"
            )

        self._cycle_reviewed_geometry = geometry

        if self._cycle_reviewed_fps is None:
            self._cycle_reviewed_fps = float(info.fps)
        elif not math.isclose(
            float(info.fps),
            self._cycle_reviewed_fps,
            rel_tol=1e-4,
            abs_tol=1e-3,
        ):
            # Reject a real rate switch while allowing the very small
            # container/OpenCV metadata jitter seen between consecutive
            # ~59.94-fps stream chunks.
            raise ValueError(
                "Video geometry/FPS changed inside an open reviewed shot"
            )

        if self._cycle_reviewed_runtime is None:
            root = Path(__file__).resolve().parents[2]
            manifest = resolve_runtime_path(self.config.runtime_manifest_path, root)
            # Core uses geometry only for half-width. On normal 16:9 -> 9:16
            # input this is the exact unmodified v4 construction. A narrow
            # source with a custom target aspect needs a harmless bootstrap
            # width, followed by botics' existing legal crop interval.
            bootstrap_width = max(info.width, info.height * 9.0 / 16.0 + 1.0)
            runtime = ReviewedRound01Runtime(str(manifest), bootstrap_width, info.height)
            runtime.features.half = legal_min
            runtime.features.ctrl.half = legal_min
            self._cycle_reviewed_runtime = runtime

        # Validate the batch before advancing any controller state.
        last = self._cycle_reviewed_last_frame
        for frame in evidence:
            fi = frame.frame_index
            if isinstance(fi, bool) or int(fi) != fi or fi < 0:
                raise ValueError("Reviewed sample frame index must be a nonnegative integer")
            if last is not None and fi <= last:
                raise ValueError("Reviewed sample frames must strictly increase; do not replay context")
            last = fi
            for candidate in frame.candidates:
                if candidate.family not in EXPECTED_FAMILIES:
                    raise ValueError("Reviewed candidate violates seven-family contract")
                if not math.isfinite(candidate.confidence) or not 0 <= candidate.confidence <= 1:
                    raise ValueError("Reviewed candidate confidence must be finite and in [0,1]")
                if len(candidate.bbox) != 4 or not all(math.isfinite(v) for v in candidate.bbox):
                    raise ValueError("Reviewed candidate box must contain four finite values")
                x1, y1, x2, y2 = candidate.bbox
                if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                    raise ValueError("Reviewed candidate box must be ordered and normalized")

        assert self._cycle_reviewed_fps is not None
        fps_rate = Fraction(
            self._cycle_reviewed_fps
        ).limit_denominator(1_000_000)

        values: List[float] = []
        for frame in evidence:
            candidates = [
                {"class_id": EXPECTED_FAMILIES.index(c.family), "family": c.family,
                 "confidence": float(c.confidence), "bbox_xyxy_norm": list(c.bbox)}
                for c in frame.candidates
            ]
            x, _, event = self._cycle_reviewed_runtime.update(
                candidates, frame.frame_index * fps_rate.denominator / fps_rate.numerator,
                shot_start=self._cycle_reviewed_last_frame is None,
            )
            if not math.isfinite(x) or not legal_min - 1e-12 <= x <= legal_max + 1e-12:
                raise ValueError("Reviewed runtime produced an invalid crop center")
            top = max(frame.candidates, key=lambda c: c.confidence, default=None)
            self._cycle_indices.append(int(frame.frame_index))
            self._cycle_raw_x.append(top.x_center if top is not None else None)
            self._cycle_confidences.append(top.confidence if top is not None else 0.0)
            self._cycle_focus.append(FocusSample(
                frame_index=frame.frame_index, timestamp_ms=frame.timestamp_ms,
                family=top.family if top is not None else "safe_center",
                confidence=top.confidence if top is not None else 0.0,
                bbox=top.bbox if top is not None else None,
                raw_x_center_norm=top.x_center if top is not None else None,
            ))
            self._cycle_events.append(int(event))
            self._cycle_reviewed_last_frame = int(frame.frame_index)
            values.append(float(x))
        return values

    def _smooth_from(self, first_new: int) -> List[float]:
        """Legacy botics helper, retained for reference; not called by active steering."""
        info = self._last_video_info
        _, legal_min, legal_max = legal_crop_geometry(
            info.width, info.height, self.config.target_aspect_width_over_height
        )
        if self._cycle_static_lock is not None:
            # A static family holds one crop for the whole shot; re-deriving it
            # per batch would make a shot that is meant to be still drift around randomly.
            return [self._cycle_static_lock] * (len(self._cycle_raw_x) - first_new)

        # Feed already-committed samples back in as left context so the batch
        # boundary does not show up as a kink.
        window_start = max(0, first_new - SMOOTHING_CONTEXT_SAMPLES)
        smoothed = smooth_samples(
            family=self.current_family,
            raw_x=self._cycle_raw_x[window_start:],
            confidences=self._cycle_confidences[window_start:],
            inference_fps=self.config.inference_fps,
            legal_min=legal_min,
            legal_max=legal_max,
            hold_and_cut=self.current_family in self._hold_and_cut_families,
            min_hold_seconds=self.config.min_hold_seconds,
        )
        values = [float(value) for value in smoothed[first_new - window_start :]]
        if self.current_family in STATIC_FAMILIES and values:
            self._cycle_static_lock = values[0]
        return values

    # ------------------------------------------------------------------
    # family / focus selection
    # ------------------------------------------------------------------

    @staticmethod
    def _family_vote(evidence: Sequence[FrameEvidence]) -> tuple[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        for frame in evidence:
            best_per_family: Dict[str, float] = {}
            for candidate in frame.candidates:
                best_per_family[candidate.family] = max(
                    best_per_family.get(candidate.family, 0.0),
                    candidate.confidence,
                )
            for family, confidence in best_per_family.items():
                if family == "safe_center":
                    continue
                scores[family] += max(0.001, confidence) ** 1.5
        if not scores:
            return "safe_center", 1.0
        family = max(scores, key=scores.get)
        total = sum(scores.values())
        return family, scores[family] / total if total > 0.0 else 0.0

    @staticmethod
    def _select_focus(
        frame: FrameEvidence,
        family: str,
        *,
        anchor_x: Optional[float] = None,
    ) -> Optional[Candidate]:
        """The best box of the shot's own family, or nothing at all.

        A frame the voted family did not fire on is a gap, not an opportunity
        to use some other family's box: substituting one moves the crop onto a
        subject the shot is not about -- a graphic_text_lock box steering a
        gameplay_follow shot, say -- and does it silently, because the sample
        still looks like evidence downstream. Returning None instead lets
        _fill_missing interpolate across the gap from the neighbouring real
        observations of the right family.

        ``anchor_x`` is where the focus was last observed. When several boxes
        of the family are within FOCUS_TIE_FRACTION of the best confidence,
        the detector has expressed no real preference between them, and the
        raw argmax would hand the choice to numerical noise -- so the one
        nearest the anchor wins instead. A box the detector genuinely prefers
        still wins outright, which is what lets the focus move to a new
        subject rather than sticking to the first one it ever saw.
        """
        matching = [candidate for candidate in frame.candidates if candidate.family == family]
        if not matching:
            return None
        best = max(matching, key=lambda item: item.confidence)
        if anchor_x is None or len(matching) == 1:
            return best
        floor = best.confidence * FOCUS_TIE_FRACTION
        tied = [candidate for candidate in matching if candidate.confidence >= floor]
        if len(tied) == 1:
            return best
        nearest = min(tied, key=lambda item: abs(item.x_center - anchor_x))
        if abs(nearest.x_center - anchor_x) > FOCUS_LINK_MAX_DISTANCE:
            return best
        return nearest

    def _select_focus_series(
        self, evidence: Sequence[FrameEvidence], family: str
    ) -> Tuple[List[int], List[Optional[float]], List[float], List[FocusSample]]:
        sample_frame_indices: List[int] = []
        raw_x: List[Optional[float]] = []
        confidences: List[float] = []
        focus_samples: List[FocusSample] = []
        # Carried across batches, so a segment boundary does not reset the
        # association the way it must not reset the smoothing context.
        anchor_x = self._cycle_focus_anchor
        gap = self._cycle_focus_gap
        for frame in evidence:
            selected = self._select_focus(frame, family, anchor_x=anchor_x)
            sample_frame_indices.append(frame.frame_index)
            if selected is None:
                gap += 1
                if gap > FOCUS_LINK_MAX_GAP_SAMPLES:
                    # The subject has been gone long enough that where it used
                    # to be says nothing about where it is now.
                    anchor_x = None
                raw_x.append(None)
                confidences.append(0.0)
                focus_samples.append(
                    FocusSample(
                        frame_index=frame.frame_index,
                        timestamp_ms=frame.timestamp_ms,
                        family="safe_center",
                        confidence=0.0,
                        bbox=None,
                        raw_x_center_norm=None,
                    )
                )
            else:
                anchor_x = selected.x_center
                gap = 0
                raw_x.append(selected.x_center)
                confidences.append(selected.confidence)
                focus_samples.append(
                    FocusSample(
                        frame_index=frame.frame_index,
                        timestamp_ms=frame.timestamp_ms,
                        family=selected.family,
                        confidence=selected.confidence,
                        bbox=selected.bbox,
                        raw_x_center_norm=selected.x_center,
                    )
                )
        self._cycle_focus_anchor = anchor_x
        self._cycle_focus_gap = gap
        return sample_frame_indices, raw_x, confidences, focus_samples

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------

    def _build_shot_analysis(
        self,
        *,
        source_media: str,
        shot_id: str,
        start_ms: int,
        end_ms: int,
        start_frame: int,
        end_frame: int,
        info: VideoInfo,
        sample_frame_indices: Sequence[int],
        sample_x: Sequence[float],
        focus_samples: Sequence[FocusSample],
        sample_events: Optional[Sequence[int]] = None,
        part_index: int = 0,
        is_final: bool = True,
    ) -> ShotAnalysis:
        crop_width, legal_min, legal_max = legal_crop_geometry(
            info.width, info.height, self.config.target_aspect_width_over_height
        )
        expanded = expand_to_source_frames(
            sample_frame_indices=sample_frame_indices,
            sample_x=sample_x,
            start_frame=start_frame,
            end_frame=end_frame,
            legal_min=legal_min,
            legal_max=legal_max,
            sample_events=sample_events,
        )
        return ShotAnalysis(
            source_media=source_media,
            shot_id=shot_id,
            start_ms=start_ms,
            end_ms=end_ms,
            start_frame=start_frame,
            frame_count=end_frame - start_frame,
            source_fps=info.fps,
            source_width=info.width,
            source_height=info.height,
            family=self.current_family,
            category=CATEGORY_BY_FAMILY.get(self.current_family, "other"),
            family_confidence=self.current_family_confidence,
            crop_width_norm=crop_width,
            x_coordinates=tuple(float(value) for value in expanded),
            focus_samples=tuple(focus_samples),
            model=self.model.identity,
            part_index=part_index,
            is_final=is_final,
        )


def _frame_to_ms(frame: int, fps: float) -> int:
    return int(round(1000.0 * frame / fps))
