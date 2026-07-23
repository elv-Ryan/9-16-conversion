from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """Normalized detector output."""

    label: str
    score: float
    box: BBox
    source: str = "detector"


@dataclass(frozen=True)
class TrackView:
    """Read-only tracked detection presented to a policy.

    Motion and observation fields are internal diagnostics only. They are not
    serialized into the common-ml output contract.
    """

    track_id: str
    label: str
    score: float
    box: BBox
    hits: int
    misses: int = 0
    last_detection_frame: int = -1
    vx_per_frame: float = 0.0
    vy_per_frame: float = 0.0
    observed_this_frame: bool = True

    @property
    def observed(self) -> bool:
        return self.observed_this_frame


@dataclass(frozen=True)
class FocusBox:
    focus_id: str
    label: str
    score: float
    box: BBox


@dataclass(frozen=True)
class FocusCandidate:
    key: str
    label: str
    score: float
    center_x: float
    boxes: Tuple[FocusBox, ...] = ()
    scene_state: str = "unknown"
    evidence_kind: str = "unknown"
    smoothing_regime: str = "normal"


@dataclass(frozen=True)
class SelectedFocus:
    key: str
    label: str
    score: float
    center_x: float
    boxes: Tuple[FocusBox, ...] = ()
    scene_state: str = "unknown"
    evidence_kind: str = "unknown"
    smoothing_regime: str = "normal"


@dataclass(frozen=True)
class FrameDecision:
    frame_idx: int
    time_ms: int
    x_center: float
    label: str
    confidence: float
    boxes: Tuple[FocusBox, ...] = ()
    selected_key: str = ""
    scene_state: str = "unknown"
    evidence_kind: str = "unknown"
    smoothing_regime: str = "normal"


@dataclass
class ShotOutput:
    shot_id: str
    start_ms: int
    end_ms: int
    start_frame_idx: int
    crop_width: float
    decisions: List[FrameDecision] = field(default_factory=list)


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration_ms: int
    frame_count: int = 0


@dataclass(frozen=True)
class FileProcessResult:
    source_media: str
    duration_ms: int
    shots: Sequence[ShotOutput]
