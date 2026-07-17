from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import os

import yaml

DEFAULT_CONFIG_PATH = Path(os.getenv("VERTICAL_FOCUS_CONFIG", "configs/policies.yml"))
DEFAULT_MODEL_PATH = "/elv/model/models/mp_tasks/object_detector/efficientdet_lite0.tflite"


@dataclass(frozen=True)
class RuntimeParams:
    mode: str = "movie"
    config_path: str = str(DEFAULT_CONFIG_PATH)
    object_model: str = DEFAULT_MODEL_PATH
    delegate: str = "cpu"
    detection_fps: Optional[float] = None
    shot_track: str = "shot_detection"
    tagstore_url: str = "https://ai.contentfabric.io"
    request_timeout_seconds: float = 30.0
    initial_content_offset_ms: int = 0
    progress_log_interval_seconds: float = 2.0
    vertical_track: Optional[str] = None
    focus_track: Optional[str] = None
    bbox_track: Optional[str] = None
    max_boxes_per_shot: Optional[int] = None
    policy_overrides: Optional[Dict[str, Any]] = None


def _nested_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _nested_update(result[key], value)
        else:
            result[key] = value
    return result


def runtime_params_from_dict(params: Dict[str, Any]) -> RuntimeParams:
    values = dict(params or {})
    if "policy" in values:
        values["mode"] = values.pop("policy")
    if "profile" in values:
        values["mode"] = values.pop("profile")

    # Consumed by common-ml rather than the focus engine.
    for key in ("continue_on_error", "allow_single_frame", "fps", "emit_progress"):
        values.pop(key, None)

    allowed = set(RuntimeParams.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unsupported vertical focus params: {unknown}")

    mode = str(values.get("mode", "movie")).strip().lower()
    if mode not in {"sports", "movie"}:
        raise ValueError("mode must be exactly 'sports' or 'movie'")
    values["mode"] = mode

    delegate = str(values.get("delegate", "cpu")).strip().lower()
    if delegate not in {"cpu", "gpu"}:
        raise ValueError("delegate must be 'cpu' or 'gpu'")
    values["delegate"] = delegate

    if values.get("detection_fps") is not None and float(values["detection_fps"]) <= 0:
        raise ValueError("detection_fps must be > 0")
    return RuntimeParams(**values)


def load_configuration(runtime: RuntimeParams) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    path = Path(runtime.config_path)
    if not path.exists():
        alternate = Path(__file__).resolve().parents[2] / "configs" / "policies.yml"
        if alternate.exists():
            path = alternate
    with path.open("r", encoding="utf-8") as handle:
        root = yaml.safe_load(handle) or {}

    policies = root.get("policies", {})
    if runtime.mode not in policies:
        raise ValueError(f"Policy {runtime.mode!r} is not present in {path}")
    policy = deepcopy(policies[runtime.mode])
    if runtime.policy_overrides:
        policy = _nested_update(policy, runtime.policy_overrides)
    if runtime.detection_fps is not None:
        policy.setdefault("detection", {})["detection_fps"] = float(runtime.detection_fps)

    output = deepcopy(root.get("output", {}))
    if runtime.vertical_track:
        output["vertical_track"] = runtime.vertical_track
    if runtime.focus_track:
        output["focus_track"] = runtime.focus_track
    if runtime.bbox_track:
        output["bbox_track"] = runtime.bbox_track
    if runtime.max_boxes_per_shot is not None:
        output["max_boxes_per_shot"] = int(runtime.max_boxes_per_shot)

    metadata = {
        "policy": runtime.mode,
        "policy_version": root.get("version", "unknown"),
        "policy_schema": root.get("schema_version", "unknown"),
    }
    return policy, output, metadata
