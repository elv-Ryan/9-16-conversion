from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


DEFAULT_MODEL_PATH = "models/nba_yolo_student/best.pt"
DEFAULT_MODEL_MANIFEST_PATH = "models/nba_yolo_student/model_manifest.json"
DEFAULT_SHOT_MODEL_PATH = "models/shot/transnetv2/torch_transnetv2.pth"

# A shot_file input is already one whole shot, so its family is voted on the
# whole thing; a segment_file input is an arbitrary slice of a continuous
# stream, so the vote is locked in after a few seconds and reused for the
# rest of the shot.
DEFAULT_FAMILY_DETERMINATION_MAX_SECONDS = {
    "shot_file": 999999.0,
    "segment_file": 3.0,
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
    min_detection_confidence: float = 0.001
    use_fp16: bool = True

    input_mode: str = "shot_file"
    max_shot_seconds: float = 900.0
    # Resolved per input_mode in __post_init__ when not supplied explicitly.
    family_determination_max_seconds: Optional[float] = None
    # How far behind the decoded frames the X trajectory is committed, in
    # source frames. Must clear SHOT_DETECTION_LOOKAHEAD_FRAMES so a late cut
    # can never land in committed trajectory; beyond that it buys smoothing
    # right-context, which the bidirectional passes need, at the cost of
    # leaving more of a shot to be finished when it closes.
    trajectory_commit_lag_frames: int = 120

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
    int_fields = {"imgsz", "batch_size", "coordinate_decimals", "trajectory_commit_lag_frames"}
    number_fields = {
        "inference_fps",
        "min_detection_confidence",
        "max_shot_seconds",
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
    if not (0.1 <= config.max_shot_seconds <= 7200.0):
        raise ValueError("max_shot_seconds must be between 0.1 and 7200")
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
    if config.input_mode not in {"shot_file", "segment_file"}:
        raise ValueError("input_mode must be 'shot_file' or 'segment_file'")
    if not config.output_track.strip():
        raise ValueError("output_track must be non-empty")
    if config.emit_focus_track and not config.focus_track.strip():
        raise ValueError("focus_track must be non-empty when emit_focus_track=true")


def resolve_runtime_path(path: str, repo_root: Optional[Path] = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    if repo_root is not None:
        alternate = repo_root / candidate
        if alternate.exists():
            return alternate
    return candidate
