from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from shot.model import ShotDetector

from .config import RuntimeConfig
from .model import EXPECTED_FAMILIES, YoloStudentModel
from .trajectory import expand_to_source_frames, legal_crop_geometry, smooth_samples
from .types import (
    Candidate,
    FocusSample,
    FrameEvidence,
    ShotAnalysis,
    ShotInterval,
    VideoInfo,
)
from .video import VideoSource


# Placeholder until real shot-boundary detection (src/shot) is wired into
# segment_file mode: pretend a shot boundary falls halfway through this many
# files after family becomes known.
_ARBITRARY_SHOT_BOUNDARY_FILE_COUNT = 6


CATEGORY_BY_FAMILY = {
    "active_speaker": "talking",
    "gameplay_follow": "gameplay",
    "graphic_text_lock": "graphic",
    "person_subject": "person",
    "safe_center": "other",
    "split_screen": "split_screen",
    "static_composition": "static",
}


class ShotManifest:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"shot manifest missing: {self.path}")
        self.payload = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def _parse_list(items: object, source_media: str) -> List[ShotInterval]:
        if not isinstance(items, list):
            raise ValueError(f"shot manifest entry for {source_media!r} must be a list")
        intervals: List[ShotInterval] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"shot manifest item {index} must be an object")
            start = item.get("start_ms")
            end = item.get("end_ms")
            if isinstance(start, bool) or not isinstance(start, int):
                raise ValueError(f"shot manifest item {index}.start_ms must be an integer")
            if isinstance(end, bool) or not isinstance(end, int):
                raise ValueError(f"shot manifest item {index}.end_ms must be an integer")
            shot_id = str(item.get("shot_id", f"shot_{index:06d}"))
            intervals.append(ShotInterval(shot_id=shot_id, start_ms=start, end_ms=end))
        return intervals

    def intervals_for(self, source_media: str) -> List[ShotInterval]:
        payload = self.payload
        if isinstance(payload, list):
            return self._parse_list(payload, source_media)
        if not isinstance(payload, dict):
            raise ValueError("shot manifest root must be a list or object")
        if "shots" in payload:
            return self._parse_list(payload["shots"], source_media)
        candidates = [source_media, str(Path(source_media).resolve()), Path(source_media).name]
        for key in candidates:
            if key in payload:
                return self._parse_list(payload[key], source_media)
        raise ValueError(f"shot manifest has no entry for source media: {source_media}")


class ShotFocusService:
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
        self.manifest = ShotManifest(config.shot_manifest_path) if config.input_mode == "shot_manifest" else None

        if config.input_mode == "segment_file":
            self.shot_detector = ShotDetector(
                transnet_path=config.shot_model_path,
                contiguous=False,  # True,
            )
        else:
            self.shot_detector = None

        # segment_file mode: stateful buffering across incoming ~2s segment
        # files, until real shot-boundary detection replaces the arbitrary
        # placeholder boundary below.
        self.segment_state = "determining_family"
        self.segment_family: Optional[str] = None
        self.segment_family_confidence: Optional[float] = None
        self.segment_raw_x: List[Optional[float]] = []
        self.segment_confidences: List[float] = []
        self.segment_sample_frame_indices: List[int] = []
        self.segment_focus_samples: List[FocusSample] = []
        self._segment_evidence_buffer: List[FrameEvidence] = []
        self._segment_buffered_seconds = 0.0
        self._segment_files_since_output_began = 0
        self._segment_shot_index = 0
        # Frames/ms fully consumed so far in the current shot cycle, counted
        # from the cycle's first file. At output time this becomes the shift
        # that re-bases everything negative, relative to the file being
        # processed when the shot concludes (see _finalize_segment_shot).
        self._segment_cumulative_frames = 0
        self._segment_cumulative_ms = 0

    def intervals_for(self, video: VideoSource) -> List[ShotInterval]:
        if self.manifest is not None:
            return self.manifest.intervals_for(video.path)

        if self.shot_detector is not None:
            shot_boundaries = self.shot_detector.tag_file_given_info(video.path, video.info.fps, round(video.info.duration_ms))
            intervals: List[ShotInterval] = []

            for index, interval in enumerate(shot_boundaries):
                intervals.append(
                    ShotInterval(
                        shot_id=f"shot_{index:06d}",
                        start_ms=interval.start_time,
                        end_ms=interval.end_time,
                    )
                )
            return intervals

        return [
            ShotInterval(
                shot_id="shot_000000",
                start_ms=0,
                end_ms=video.info.duration_ms,
            )
        ]

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

    def _build_shot_analysis(
        self,
        *,
        source_media: str,
        shot_id: str,
        start_ms: int,
        end_ms: int,
        start_frame: int,
        end_frame: int,
        source_fps: float,
        source_width: int,
        source_height: int,
        family: str,
        family_confidence: float,
        sample_frame_indices: Sequence[int],
        raw_x: Sequence[Optional[float]],
        confidences: Sequence[float],
        focus_samples: Sequence[FocusSample],
    ) -> ShotAnalysis:
        crop_width, legal_min, legal_max = legal_crop_geometry(
            source_width, source_height, self.config.target_aspect_width_over_height
        )
        sample_x = smooth_samples(
            family=family,
            raw_x=raw_x,
            confidences=confidences,
            inference_fps=self.config.inference_fps,
            legal_min=legal_min,
            legal_max=legal_max,
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
            source_fps=source_fps,
            source_width=source_width,
            source_height=source_height,
            family=family,
            category=CATEGORY_BY_FAMILY.get(family, "other"),
            family_confidence=family_confidence,
            crop_width_norm=crop_width,
            x_coordinates=tuple(float(value) for value in expanded),
            focus_samples=tuple(focus_samples),
            model=self.model.identity,
        )

    def _extend_offset_buffers(self, evidence: Sequence[FrameEvidence]) -> None:
        sample_frame_indices, raw_x, confidences, focus_samples = self._select_focus_series(
            evidence, self.segment_family
        )
        self.segment_sample_frame_indices.extend(sample_frame_indices)
        self.segment_raw_x.extend(raw_x)
        self.segment_confidences.extend(confidences)
        self.segment_focus_samples.extend(focus_samples)

    def _reset_current_segment_shot(self) -> None:
        self.segment_state = "determining_family"
        self.segment_family = None
        self.segment_family_confidence = None
        self.segment_raw_x = []
        self.segment_confidences = []
        self.segment_sample_frame_indices = []
        self.segment_focus_samples = []
        self._segment_evidence_buffer = []
        self._segment_buffered_seconds = 0.0
        self._segment_files_since_output_began = 0
        self._segment_cumulative_frames = 0
        self._segment_cumulative_ms = 0

    def _finalize_segment_shot(
        self, *, source_media: str, video_info: VideoInfo, end_frame: int
    ) -> ShotAnalysis:
        # Everything buffered so far is relative to the start of the current
        # shot cycle. Re-base it negative, relative to frame 0 of the file
        # being processed right now (the one concluding the shot), by
        # shifting back by however many frames/ms of *earlier* files fed
        # this cycle before it.
        shift_frames = self._segment_cumulative_frames
        shift_ms = self._segment_cumulative_ms
        shot_id = f"shot_{self._segment_shot_index:06d}"
        self._segment_shot_index += 1
        sample_frame_indices = [index - shift_frames for index in self.segment_sample_frame_indices]
        focus_samples = [
            replace(
                sample,
                frame_index=sample.frame_index - shift_frames,
                timestamp_ms=sample.timestamp_ms - shift_ms,
            )
            for sample in self.segment_focus_samples
        ]
        return self._build_shot_analysis(
            source_media=source_media,
            shot_id=shot_id,
            start_ms=-shift_ms,
            end_ms=int(round(1000.0 * end_frame / video_info.fps)),
            start_frame=-shift_frames,
            end_frame=end_frame,
            source_fps=video_info.fps,
            source_width=video_info.width,
            source_height=video_info.height,
            family=self.segment_family,
            family_confidence=self.segment_family_confidence,
            sample_frame_indices=sample_frame_indices,
            raw_x=self.segment_raw_x,
            confidences=self.segment_confidences,
            focus_samples=focus_samples,
        )

    def _try_determine_family(self) -> None:
        if self._segment_buffered_seconds < self.config.family_determination_max_seconds:
            return
        family, family_confidence = self._family_vote(self._segment_evidence_buffer)
        self.segment_family = family
        self.segment_family_confidence = family_confidence
        self._extend_offset_buffers(self._segment_evidence_buffer)
        self._segment_evidence_buffer = []
        self.segment_state = "outputting_offset"
        self._segment_files_since_output_began = 0

    def _advance_segment_state(
        self, source_media: str, video_info: VideoInfo, evidence: Sequence[FrameEvidence]
    ) -> List[ShotAnalysis]:
        if not evidence:
            raise ValueError(f"segment {source_media} produced no decodable frames")

        # Re-base this file's frame indices/timestamps onto the running,
        # monotonically increasing axis for the current shot cycle (frame 0
        # == the cycle's first file's frame 0), so buffered evidence from
        # different physical files can be interpolated/finalized together.
        frame_base = self._segment_cumulative_frames
        ms_base = self._segment_cumulative_ms
        remapped = [
            FrameEvidence(
                frame_index=frame.frame_index + frame_base,
                timestamp_ms=frame.timestamp_ms + ms_base,
                candidates=frame.candidates,
            )
            for frame in evidence
        ]

        if self.segment_state == "determining_family":
            self._segment_evidence_buffer.extend(remapped)
            self._segment_buffered_seconds += video_info.duration_ms / 1000.0
            self._segment_cumulative_frames += video_info.frame_count
            self._segment_cumulative_ms += video_info.duration_ms
            self._try_determine_family()
            return []

        # outputting_offset
        self._segment_files_since_output_began += 1
        if self._segment_files_since_output_began < _ARBITRARY_SHOT_BOUNDARY_FILE_COUNT:
            self._extend_offset_buffers(remapped)
            self._segment_cumulative_frames += video_info.frame_count
            self._segment_cumulative_ms += video_info.duration_ms
            return []

        # Arbitrary placeholder shot boundary: split this file in half.
        split_frame = video_info.frame_count // 2
        first_half = [frame for frame, raw in zip(remapped, evidence) if raw.frame_index < split_frame]
        second_half = [raw for raw in evidence if raw.frame_index >= split_frame]
        self._extend_offset_buffers(first_half)

        completed = self._finalize_segment_shot(
            source_media=source_media, video_info=video_info, end_frame=split_frame
        )
        self._reset_current_segment_shot()

        # Hand the leftover half straight to the next determining_family
        # cycle, so the buffer never has to rewind to a file's beginning.
        # The new cycle's reference frame 0 is the leftover's own first
        # frame, so its indices/timestamps are rebased back by split_frame.
        if second_half:
            split_frame_ms = int(round(1000.0 * split_frame / video_info.fps))
            rebased_second_half = [
                FrameEvidence(
                    frame_index=raw.frame_index - split_frame,
                    timestamp_ms=raw.timestamp_ms - split_frame_ms,
                    candidates=raw.candidates,
                )
                for raw in second_half
            ]
            self._segment_evidence_buffer.extend(rebased_second_half)
            remaining_frames = video_info.frame_count - split_frame
            self._segment_buffered_seconds += remaining_frames / video_info.fps
            self._segment_cumulative_frames = remaining_frames
            self._segment_cumulative_ms = video_info.duration_ms - split_frame_ms
            self._try_determine_family()

        return [completed]

    def ingest_segment_file(self, source_media: str) -> List[ShotAnalysis]:
        video = VideoSource(source_media)
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
        return self._advance_segment_state(source_media, video.info, evidence)

    def analyze_file(self, source_media: str) -> List[ShotAnalysis]:
        if self.config.input_mode == "segment_file":
            return self.ingest_segment_file(source_media)

        video = VideoSource(source_media)
        analyses: List[ShotAnalysis] = []
        for interval in self.intervals_for(video):
            analyses.append(self._analyze_interval(video, interval))
        return analyses

    def _analyze_interval(self, video: VideoSource, interval: ShotInterval) -> ShotAnalysis:
        start_frame, end_frame = video.validate_interval(interval, self.config.max_shot_seconds)
        evidence: List[FrameEvidence] = []
        for frame_indices, frames in video.iter_sample_batches(
            start_frame=start_frame,
            end_frame=end_frame,
            inference_fps=self.config.inference_fps,
            batch_size=self.config.batch_size,
            max_seconds=self.config.family_determination_max_seconds,
        ):
            evidence.extend(
                self.model.infer_batch(
                    frame_indices=frame_indices,
                    frames=frames,
                    source_fps=video.info.fps,
                )
            )
        if not evidence:
            raise ValueError(f"shot {interval.shot_id} produced no decodable frames")

        family, family_confidence = self._family_vote(evidence)
        sample_frame_indices, raw_x, confidences, focus_samples = self._select_focus_series(evidence, family)
        return self._build_shot_analysis(
            source_media=video.path,
            shot_id=interval.shot_id,
            start_ms=interval.start_ms,
            end_ms=interval.end_ms,
            start_frame=start_frame,
            end_frame=end_frame,
            source_fps=video.info.fps,
            source_width=video.info.width,
            source_height=video.info.height,
            family=family,
            family_confidence=family_confidence,
            sample_frame_indices=sample_frame_indices,
            raw_x=raw_x,
            confidences=confidences,
            focus_samples=focus_samples,
        )
