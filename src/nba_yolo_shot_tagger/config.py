from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


# Kept in step with model.EXPECTED_FAMILIES, but named here too so config
# validation does not have to import the model module (and with it numpy).
EXPECTED_FAMILIES = (
    "active_speaker",
    "gameplay_follow",
    "graphic_text_lock",
    "person_subject",
    "safe_center",
    "split_screen",
    "static_composition",
)

DEFAULT_MODEL_PATH = "models/nba_yolo_student/best.pt"
DEFAULT_MODEL_MANIFEST_PATH = "models/nba_yolo_student/model_manifest.json"
DEFAULT_SHOT_MODEL_PATH = "models/shot/transnetv2/torch_transnetv2.pth"

# A shot_file input is already one whole shot, so its family is voted on the
# whole thing; a segment_file input is an arbitrary slice of a continuous
# stream, so the vote is locked in after a few seconds and reused for the
# rest of the shot.
DEFAULT_FAMILY_DETERMINATION_MAX_SECONDS = {
    "shot_file": 999999.0,
    "segment_file": 4.0,
}

# TransNetV2 keeps the middle 50 predictions of each 100-frame window, so it
# needs about this much real video after a frame before it will commit to a
# cut there. Nothing may be decided inside this margin, because a cut reported
# late would land in it.
SHOT_DETECTION_LOOKAHEAD_FRAMES = 25


@dataclass(frozen=True)
class RuntimeConfig:
    """Strict runtime configuration supplied through ``--params``.

    ``shot_file`` (the default) treats every stdin media path as one already
    segmented shot. ``segment_file`` treats them as consecutive slices of one
    continuous stream and detects shot boundaries itself.
    """

    model_path: str = DEFAULT_MODEL_PATH
    shot_model_path: str = DEFAULT_SHOT_MODEL_PATH
    model_manifest_path: str = DEFAULT_MODEL_MANIFEST_PATH
    verify_model_sha256: bool = True
    device: str = "0"
    imgsz: int = 1280
    inference_fps: float = 10.0
    batch_size: int = 8
    # The checkpoint's head is end-to-end (NMS-free) and returns its top
    # ``max_detections`` boxes regardless of quality, so this floor is the only
    # thing standing between the family vote and a hundred noise boxes per
    # frame. Both are passed straight through to predict().
    #
    # 0.05 is measured, not guessed: over 3,150 sampled frames of test-files
    # this checkpoint returns 96 boxes per frame at the old 0.001 floor, whose
    # x-centres are spread over 0.73 of the frame -- they agree on nothing. At
    # 0.05 that falls to 2.3 boxes per frame spread over 0.011, 78% of frames
    # still carry a box, and the family vote changes on 1 file in 150. Raising
    # it further starts costing real evidence: 0.15 leaves only 43% of frames
    # with any box and flips 19% of the votes. The checkpoint's confidences are
    # compressed (p50 = 0.002, p99 = 0.10), so this is a low number by design.
    min_detection_confidence: float = 0.05
    # Applied as top-k by score before the confidence filter, so it can only
    # ever discard boxes the floor would have dropped anyway: at 0.05 no frame
    # in the sample kept more than 15 boxes.
    max_detections: int = 20
    use_fp16: bool = True
    live_data_stream: str = ""

    input_mode: str = "segment_file"
    max_shot_seconds: float = 900.0
    # TransNetV2's per-frame prediction flickers across a dissolve, and
    # predict_frames_shots() collapses only strictly consecutive positives, so
    # one soft transition can report several boundaries a few frames apart.
    # Ninety consecutive segments produced two shots of 34 ms and 66 ms this
    # way, each carrying a full family vote taken on two frames of evidence. A
    # cut closer than this to the start of the open shot is dropped and its
    # frames are absorbed into the following shot.
    #
    # Only segment_file detects its own cuts. shot_file boundaries are declared
    # by the caller, so they are always honoured however short the file is.
    min_shot_seconds: float = 0.75
    # Resolved per input_mode in __post_init__ when not supplied explicitly.
    family_determination_max_seconds: Optional[float] = None
    # How far behind the decoded frames the X trajectory is committed, in
    # source frames. Must clear SHOT_DETECTION_LOOKAHEAD_FRAMES so a late cut
    # can never land in committed trajectory; beyond that it buys smoothing
    # right-context, which the bidirectional passes need, at the cost of
    # leaving more of a shot to be finished when it closes.
    trajectory_commit_lag_frames: int = 240

    # Families whose trajectory holds one framing and cuts, instead of being
    # continuously smoothed. Comma-separated so it can be retargeted through
    # --params without a rebuild: interviews are meant to land on
    # ``active_speaker``, but if this checkpoint votes them ``person_subject``
    # the same treatment is one deploy-time edit away.
    hold_and_cut_families: str = "active_speaker"
    # How long a subject must hold the framing before a change of subject is
    # accepted as a cut. Shorter reads as restless; longer misses real
    # exchanges. Interview cutting rarely goes faster than about a second.
    min_hold_seconds: float = 1.0

    output_track: str = "vertical_video"
    focus_track: str = "focus"
    emit_focus_track: bool = False
    include_focus_samples: bool = False
    coordinate_decimals: int = 6
    target_aspect_width_over_height: float = 9.0 / 16.0

    emit_progress_ratio: bool = False

    def __post_init__(self) -> None:
        if self.family_determination_max_seconds is None:
            object.__setattr__(
                self,
                "family_determination_max_seconds",
                DEFAULT_FAMILY_DETERMINATION_MAX_SECONDS.get(self.input_mode, 999999.0),
            )


_COMMON_ML_KEYS = {
    "continue_on_error",
    "allow_single_frame",
    "fps",
    "emit_progress",
}


def _strict_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be a JSON boolean")


def _strict_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a JSON integer")
    return value


def _strict_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    return float(value)


def config_from_params(params: Mapping[str, Any]) -> RuntimeConfig:
    if not isinstance(params, Mapping):
        raise ValueError("--params must decode to a JSON object")

    raw: Dict[str, Any] = dict(params)
    for key in _COMMON_ML_KEYS:
        raw.pop(key, None)

    allowed = {field.name for field in fields(RuntimeConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unsupported NBA shot-tagger params: {unknown}")

    bool_fields = {
        "verify_model_sha256",
        "use_fp16",
        "emit_focus_track",
        "include_focus_samples",
        "emit_progress_ratio",
    }
    int_fields = {
        "imgsz",
        "batch_size",
        "coordinate_decimals",
        "trajectory_commit_lag_frames",
        "max_detections",
    }
    number_fields = {
        "inference_fps",
        "min_detection_confidence",
        "max_shot_seconds",
        "min_shot_seconds",
        "min_hold_seconds",
        "family_determination_max_seconds",
        "target_aspect_width_over_height",
    }

    normalized: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in bool_fields:
            normalized[key] = _strict_bool(key, value)
        elif key in int_fields:
            normalized[key] = _strict_int(key, value)
        elif key in number_fields:
            normalized[key] = _strict_number(key, value)
        elif not isinstance(value, str):
            raise ValueError(f"{key} must be a JSON string")
        else:
            normalized[key] = value

    config = RuntimeConfig(**normalized)
    _validate(config)
    return config


def _validate(config: RuntimeConfig) -> None:
    if config.imgsz < 320 or config.imgsz > 4096:
        raise ValueError("imgsz must be between 320 and 4096")
    if config.batch_size < 1 or config.batch_size > 128:
        raise ValueError("batch_size must be between 1 and 128")
    if not (0.0 < config.inference_fps <= 120.0):
        raise ValueError("inference_fps must be in (0, 120]")
    if not (0.0 <= config.min_detection_confidence < 1.0):
        raise ValueError("min_detection_confidence must be in [0, 1)")
    if config.max_detections < 1 or config.max_detections > 300:
        # 300 is the checkpoint's own Detect.max_det; asking for more is a
        # silent no-op rather than an error, so reject it here.
        raise ValueError("max_detections must be between 1 and 300")
    if not (0.1 <= config.max_shot_seconds <= 7200.0):
        raise ValueError("max_shot_seconds must be between 0.1 and 7200")
    if not (0.0 <= config.min_shot_seconds <= 10.0):
        raise ValueError("min_shot_seconds must be between 0 and 10")
    if config.min_shot_seconds >= config.max_shot_seconds:
        raise ValueError("min_shot_seconds must be below max_shot_seconds")
    if not (0.1 <= config.family_determination_max_seconds <= 1_000_000.0):
        raise ValueError("family_determination_max_seconds must be between 0.1 and 1000000")
    if config.trajectory_commit_lag_frames < SHOT_DETECTION_LOOKAHEAD_FRAMES:
        raise ValueError(
            "trajectory_commit_lag_frames must be at least "
            f"{SHOT_DETECTION_LOOKAHEAD_FRAMES} (shot detection's lookahead)"
        )
    if not (0.05 <= config.target_aspect_width_over_height <= 2.0):
        raise ValueError("target_aspect_width_over_height is invalid")
    if not (0 <= config.coordinate_decimals <= 9):
        raise ValueError("coordinate_decimals must be between 0 and 9")
    if not (0.0 <= config.min_hold_seconds <= 30.0):
        raise ValueError("min_hold_seconds must be between 0 and 30")
    unknown_families = sorted(parse_hold_and_cut_families(config.hold_and_cut_families) - set(EXPECTED_FAMILIES))
    if unknown_families:
        raise ValueError(f"hold_and_cut_families contains unknown families: {unknown_families}")
    if config.input_mode not in {"shot_file", "segment_file"}:
        raise ValueError("input_mode must be 'shot_file' or 'segment_file'")
    if not config.output_track.strip():
        raise ValueError("output_track must be non-empty")
    if config.emit_focus_track and not config.focus_track.strip():
        raise ValueError("focus_track must be non-empty when emit_focus_track=true")


def parse_hold_and_cut_families(raw: str) -> set:
    """Comma-separated family names, tolerant of spaces and a trailing comma."""
    return {piece.strip() for piece in str(raw).split(",") if piece.strip()}


def resolve_runtime_path(path: str, repo_root: Optional[Path] = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    if repo_root is not None:
        alternate = repo_root / candidate
        if alternate.exists():
            return alternate
    return candidate
