from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping

DEFAULT_MODEL_PATH = "/elv/model/models/nba_yolo_student/best.pt"
DEFAULT_MODEL_MANIFEST_PATH = "/elv/model/models/nba_yolo_student/model_manifest.json"
DEFAULT_RUNTIME_MANIFEST_PATH = "/elv/model/models/nba_yolo_student/runtime_manifest.json"


@dataclass(frozen=True)
class RuntimeConfig:
    # Frozen reviewed artifact contract.
    model_path: str = DEFAULT_MODEL_PATH
    model_manifest_path: str = DEFAULT_MODEL_MANIFEST_PATH
    runtime_manifest_path: str = DEFAULT_RUNTIME_MANIFEST_PATH
    verify_model_sha256: bool = True
    device: str = "0"
    imgsz: int = 1280
    inference_fps: float = 10.0
    batch_size: int = 8
    min_detection_confidence: float = 0.001
    iou: float = 0.7
    max_det: int = 100
    top_k: int = 20
    use_fp16: bool = False

    # Input is already segmented in VOD deployment. No shot detection here.
    input_mode: str = "shot_file"
    shot_manifest_path: str = ""
    max_shot_seconds: float = 900.0

    # Wire output.
    output_track: str = "vertical_video"
    tag_name: str = "nba_yolo_round01_oneeuro"
    provenance: str = "nba_reviewed_round01_oneeuro_v1"
    source_iq: str = ""
    coordinate_decimals: int = 9
    chunk_frames: int = 240
    target_aspect_width_over_height: float = 9.0 / 16.0


_COMMON_ML_KEYS = {"continue_on_error", "allow_single_frame", "fps", "emit_progress"}


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
    bool_fields = {"verify_model_sha256", "use_fp16"}
    int_fields = {"imgsz", "batch_size", "max_det", "top_k", "coordinate_decimals", "chunk_frames"}
    number_fields = {"inference_fps", "min_detection_confidence", "iou", "max_shot_seconds", "target_aspect_width_over_height"}
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


def _validate(c: RuntimeConfig) -> None:
    if not 320 <= c.imgsz <= 4096:
        raise ValueError("imgsz must be between 320 and 4096")
    if not 1 <= c.batch_size <= 128:
        raise ValueError("batch_size must be in [1, 128]")
    if not 0 < c.inference_fps <= 120:
        raise ValueError("inference_fps must be in (0, 120]")
    if not 0 <= c.min_detection_confidence < 1:
        raise ValueError("min_detection_confidence must be in [0,1)")
    if not 0 < c.iou <= 1:
        raise ValueError("iou must be in (0,1]")
    if not 1 <= c.max_det <= 1000:
        raise ValueError("max_det must be in [1,1000]")
    if not 1 <= c.top_k <= c.max_det:
        raise ValueError("top_k must be in [1,max_det]")
    if not 1 <= c.chunk_frames <= 10000:
        raise ValueError("chunk_frames must be positive")
    if c.input_mode not in {"shot_file", "shot_manifest"}:
        raise ValueError("input_mode must be shot_file or shot_manifest")
    if c.input_mode == "shot_manifest" and not c.shot_manifest_path:
        raise ValueError("shot_manifest_path is required")
    if not 0.05 <= c.target_aspect_width_over_height <= 2:
        raise ValueError("invalid target aspect")
    if not 0 <= c.coordinate_decimals <= 9:
        raise ValueError("coordinate_decimals must be in [0,9]")
