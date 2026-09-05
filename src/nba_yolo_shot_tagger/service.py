from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from .config import RuntimeConfig
from .model import YoloStudentModel
from .trajectory import (
    STATIC_FAMILIES,
    expand_to_source_frames,
    legal_crop_geometry,
    smooth_samples,
)
from .types import Candidate, FocusSample, FrameEvidence, ShotAnalysis, VideoInfo
from .video import VideoSource


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

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.model = YoloStudentModel(
            model_path=config.model_path,
            manifest_path=config.model_manifest_path,
            verify_sha256=config.verify_model_sha256,
            device=config.device,
            imgsz=config.imgsz,
            min_confidence=config.min_detection_confidence,
            use_fp16=config.use_fp16,
        )

        if config.input_mode == "segment_file":
            # Imported here so shot_file mode never has to load torch/TransNet.
            from .shot_boundaries import StreamShotDetector

            self.shot_detector = StreamShotDetector(
                config.shot_model_path,
                device="cpu" if config.device == "cpu" else None,
            )
            # Only segment_file defers cuts, so only it has to hold back.
            self._commit_lag = config.trajectory_commit_lag_frames
        else:
            self.shot_detector = None
            self._commit_lag = 0

        self._shot_index = 0
        self._abs_frames = 0  # source frames folded so far, across all files
        self._abs_ms = 0
        self._last_source_media: Optional[str] = None
        self._last_video_info: Optional[VideoInfo] = None
        self._last_abs_start = 0
        self._last_abs_start_ms = 0
        self._reset_cycle(0, 0)

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
        analysis = self._close(self._abs_frames)
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

        if (self._abs_ms - self._cycle_start_abs_ms) / 1000.0 >= self.config.max_shot_seconds:
            # Safety valve: an open shot is held in memory, and a stream can
            # run a long way without a detected cut.
            analysis = self._close(self._abs_frames)
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

    def _close(self, cut_abs: int) -> Optional[ShotAnalysis]:
        """Cut the open shot at an absolute stream frame and emit it.

        The shot is emitted against the file currently being processed, so a
        cut that lands before that file starts (one the detector deferred)
        simply produces a more negative offset.
        """
        length = cut_abs - self._cycle_start_abs
        if length <= 0:
            return None
        info = self._last_video_info
        anchor_abs = self._last_abs_start
        anchor_abs_ms = self._last_abs_start_ms

        # Settle the rest of this shot: a family if it was cut before the vote
        # was due, and the trajectory for whatever was still being held back.
        self._advance(length, force_family=True)

        start_frame = self._cycle_start_abs - anchor_abs
        end_frame = cut_abs - anchor_abs
        start_ms = self._cycle_start_abs_ms - anchor_abs_ms
        # A sub-millisecond shot must still be a non-empty interval.
        end_ms = max(_frame_to_ms(end_frame, info.fps), start_ms + 1)

        analysis = self._build_shot_analysis(
            source_media=self._last_source_media,
            shot_id=f"shot_{self._shot_index:06d}",
            start_ms=start_ms,
            end_ms=end_ms,
            start_frame=start_frame,
            end_frame=end_frame,
            info=info,
            sample_frame_indices=[index + start_frame for index in self._cycle_indices],
            sample_x=self._cycle_smoothed,
            focus_samples=[
                replace(
                    sample,
                    frame_index=sample.frame_index + start_frame,
                    timestamp_ms=sample.timestamp_ms + start_ms,
                )
                for sample in self._cycle_focus
            ],
        )
        self._shot_index += 1

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
        self._cycle_static_lock: Optional[float] = None
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

        first_new = len(self._cycle_indices)
        indices, raw_x, confidences, focus_samples = self._select_focus_series(
            ready, self.current_family
        )
        self._cycle_indices.extend(indices)
        self._cycle_raw_x.extend(raw_x)
        self._cycle_confidences.extend(confidences)
        self._cycle_focus.extend(focus_samples)
        new_smoothed_samples = self._smooth_from(first_new)
        self._cycle_smoothed.extend(new_smoothed_samples)

        ## write out new_smoothed_samples here
        ## it will be some amount of data, equivalent to a segment in the steady state, but there may be more or less if the shot is just starting or ending. 
        
    def _smooth_from(self, first_new: int) -> List[float]:
        """Smoothed X for samples ``first_new`` onwards, once and for all."""
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
    def _select_focus(frame: FrameEvidence, family: str) -> Optional[Candidate]:
        matching = [candidate for candidate in frame.candidates if candidate.family == family]
        if matching:
            return max(matching, key=lambda item: item.confidence)
        if frame.candidates:
            return max(frame.candidates, key=lambda item: item.confidence)
        return None

    def _select_focus_series(
        self, evidence: Sequence[FrameEvidence], family: str
    ) -> Tuple[List[int], List[Optional[float]], List[float], List[FocusSample]]:
        sample_frame_indices: List[int] = []
        raw_x: List[Optional[float]] = []
        confidences: List[float] = []
        focus_samples: List[FocusSample] = []
        for frame in evidence:
            selected = self._select_focus(frame, family)
            sample_frame_indices.append(frame.frame_index)
            if selected is None:
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
        )


def _frame_to_ms(frame: int, fps: float) -> int:
    return int(round(1000.0 * frame / fps))
