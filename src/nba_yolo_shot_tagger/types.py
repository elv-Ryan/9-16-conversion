from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_ms: int


@dataclass(frozen=True)
class Candidate:
    family: str
    confidence: float
    bbox: BBox

    @property
    def x_center(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])


@dataclass(frozen=True)
class FrameEvidence:
    frame_index: int
    timestamp_ms: int
    candidates: Sequence[Candidate]


@dataclass(frozen=True)
class FocusSample:
    frame_index: int
    timestamp_ms: int
    family: str
    confidence: float
    bbox: Optional[BBox]
    raw_x_center_norm: Optional[float]


@dataclass(frozen=True)
class ModelIdentity:
    path: str
    sha256: str
    artifact_version: str
    class_names: Sequence[str]


@dataclass(frozen=True)
class ShotAnalysis:
    source_media: str
    shot_id: str
    start_ms: int
    end_ms: int
    start_frame: int
    frame_count: int
    source_fps: float
    source_width: int
    source_height: int
    family: str
    category: str
    family_confidence: float
    crop_width_norm: float
    x_coordinates: Sequence[float]
    focus_samples: Sequence[FocusSample]
    model: ModelIdentity
