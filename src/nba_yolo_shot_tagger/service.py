from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from shot.model import ShotDetector

from .config import RuntimeConfig
from .model import EXPECTED_FAMILIES, YoloStudentModel
from .trajectory import expand_to_source_frames, legal_crop_geometry, smooth_samples
from .types import Candidate, FocusSample, FrameEvidence, ShotAnalysis, ShotInterval
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

    def analyze_file(self, source_media: str) -> List[ShotAnalysis]:



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
        ):
            print("infer batch size", len(frame_indices))
            if len(frame_indices) > 0: print("frame[0]", frame_indices[0])

            evidence.extend(
                self.model.infer_batch(
                    frame_indices=frame_indices,
                    frames=frames,
                    source_fps=video.info.fps,
                )
            )
        if not evidence:
            raise ValueError(f"shot {interval.shot_id} produced no decodable frames")

        family_determination_cutoff_frame = start_frame + int(
            round(self.config.family_determination_max_seconds * video.info.fps)
        )
        family_vote_evidence = [
            frame for frame in evidence if frame.frame_index < family_determination_cutoff_frame
        ] or evidence
        family, family_confidence = self._family_vote(family_vote_evidence)
        print("family, confidence", family, family_confidence)
        focus_samples: List[FocusSample] = []
        raw_x: List[Optional[float]] = []
        confidences: List[float] = []
        sample_frame_indices: List[int] = []
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

        crop_width, legal_min, legal_max = legal_crop_geometry(
            video.info.width,
            video.info.height,
            self.config.target_aspect_width_over_height,
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
            source_media=video.path,
            shot_id=interval.shot_id,
            start_ms=interval.start_ms,
            end_ms=interval.end_ms,
            start_frame=start_frame,
            frame_count=end_frame - start_frame,
            source_fps=video.info.fps,
            source_width=video.info.width,
            source_height=video.info.height,
            family=family,
            category=CATEGORY_BY_FAMILY.get(family, "other"),
            family_confidence=family_confidence,
            crop_width_norm=crop_width,
            x_coordinates=tuple(float(value) for value in expanded),
            focus_samples=tuple(focus_samples),
            model=self.model.identity,
        )
