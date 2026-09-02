from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterable, List

from common_ml.tagging.messages import FrameInfo, Tag

from .config import RuntimeConfig
from .types import FocusSample, ShotAnalysis

SCHEMA_VERSION = "eluvio.nba-yolo-shot-x.v1"


def _round(value: float, decimals: int) -> float:
    return round(float(value), decimals)


def _focus_sample(sample: FocusSample, decimals: int) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "frame_index": int(sample.frame_index),
        "timestamp_ms": int(sample.timestamp_ms),
        "family": sample.family,
        "confidence": _round(sample.confidence, decimals),
        "raw_x_center_norm": (
            None if sample.raw_x_center_norm is None else _round(sample.raw_x_center_norm, decimals)
        ),
    }
    if sample.bbox is not None:
        x1, y1, x2, y2 = sample.bbox
        payload["bbox"] = {
            "x1": _round(x1, decimals),
            "y1": _round(y1, decimals),
            "x2": _round(x2, decimals),
            "y2": _round(y2, decimals),
        }
    return payload


def validate_analysis(analysis: ShotAnalysis) -> None:
    if analysis.end_ms <= analysis.start_ms:
        raise ValueError("shot end_time must be greater than start_time")
    if analysis.frame_count <= 0:
        raise ValueError("shot frame_count must be positive")
    if len(analysis.x_coordinates) != analysis.frame_count:
        raise ValueError(
            "x-coordinates must contain exactly one value per source frame: "
            f"values={len(analysis.x_coordinates)} frames={analysis.frame_count}"
        )
    legal_min = 0.5 * analysis.crop_width_norm
    legal_max = 1.0 - legal_min
    for value in analysis.x_coordinates:
        if not isinstance(value, (int, float)):
            raise ValueError("x-coordinates contains a non-numeric value")
        if value < legal_min - 1e-9 or value > legal_max + 1e-9:
            raise ValueError(
                f"x-coordinate {value} violates legal crop-center range "
                f"[{legal_min}, {legal_max}]"
            )


def tags_for_analysis(analysis: ShotAnalysis, config: RuntimeConfig) -> List[Tag]:
    validate_analysis(analysis)
    decimals = config.coordinate_decimals
    legal_min = 0.5 * analysis.crop_width_norm
    legal_max = 1.0 - legal_min
    additional_info: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "shot_id": analysis.shot_id,
        "input_mode": config.input_mode,
        "family": analysis.family,
        "category": analysis.category,
        "category_level": "runtime_policy_family",
        "family_confidence": _round(analysis.family_confidence, decimals),
        "source_fps": _round(analysis.source_fps, decimals),
        "source_width": int(analysis.source_width),
        "source_height": int(analysis.source_height),
        "source_frame_start": int(analysis.start_frame),
        "frame_count": int(analysis.frame_count),
        "x_coordinate_alignment": "source_frame",
        "x_coordinate_units": "normalized_source_width",
        "crop_width_norm": _round(analysis.crop_width_norm, decimals),
        "legal_x_center_min": _round(legal_min, decimals),
        "legal_x_center_max": _round(legal_max, decimals),
        "x-coordinates": [_round(value, decimals) for value in analysis.x_coordinates],
        "model": {
            "artifact_version": analysis.model.artifact_version,
            "sha256": analysis.model.sha256,
            "class_names": list(analysis.model.class_names),
        },
    }
    if config.include_focus_samples:
        additional_info["focus_samples"] = [
            _focus_sample(sample, decimals) for sample in analysis.focus_samples
        ]
        additional_info["focus_sample_fps"] = _round(config.inference_fps, decimals)

    tags = [
        Tag(
            tag=analysis.family,
            start_time=int(analysis.start_ms),
            end_time=int(analysis.end_ms),
            source_media=analysis.source_media,
            track=config.output_track,
            additional_info=additional_info,
        )
    ]
    if config.emit_focus_track:
        representative = max(
            (sample for sample in analysis.focus_samples if sample.bbox is not None),
            key=lambda sample: sample.confidence,
            default=None,
        )
        frame_info = None
        if representative is not None and representative.bbox is not None:
            x1, y1, x2, y2 = representative.bbox
            frame_info = FrameInfo(
                frame_idx=int(representative.frame_index),
                box={
                    "x1": _round(x1, decimals),
                    "y1": _round(y1, decimals),
                    "x2": _round(x2, decimals),
                    "y2": _round(y2, decimals),
                },
            )
        tags.append(
            Tag(
                tag=analysis.family,
                start_time=int(analysis.start_ms),
                end_time=int(analysis.end_ms),
                source_media=analysis.source_media,
                track=config.focus_track,
                frame_info=frame_info,
                additional_info={
                    "schema_version": SCHEMA_VERSION,
                    "shot_id": analysis.shot_id,
                    "family": analysis.family,
                    "category": analysis.category,
                    "family_confidence": _round(analysis.family_confidence, decimals),
                    "samples": [
                        _focus_sample(sample, decimals) for sample in analysis.focus_samples
                    ],
                },
            )
        )
    return tags
