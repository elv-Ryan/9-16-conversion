from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .geometry import clamp, clamp_crop_center, sanitize_box
from .types import FocusBox, ShotOutput


@dataclass(frozen=True)
class TagSpec:
    tag: str
    start_time: int
    end_time: int
    track: str
    frame_info: Dict[str, Any]
    additional_info: Dict[str, Any]


def build_shot_tag_specs(
    shot: ShotOutput,
    *,
    output_config: Dict[str, Any],
    policy_metadata: Dict[str, Any],
) -> List[TagSpec]:
    if not shot.decisions:
        return []
    decimals = int(output_config.get("coordinate_decimals", 6))
    vertical_track = str(output_config.get("vertical_track", "vertical_video"))
    focus_track = str(output_config.get("focus_track", "focus"))
    bbox_track = str(output_config.get("bbox_track", "focus_bbox"))
    max_boxes = max(1, int(output_config.get("max_boxes_per_shot", 3)))

    label_counts = Counter(decision.label for decision in shot.decisions)
    dominant_label = label_counts.most_common(1)[0][0]
    x_coordinates = [round(clamp(float(d.x_center), 0.0, 1.0), decimals) for d in shot.decisions]
    average_confidence = sum(float(d.confidence) for d in shot.decisions) / len(shot.decisions)

    focus_occurrences: Dict[str, List[tuple[int, FocusBox]]] = defaultdict(list)
    for decision in shot.decisions:
        for box in decision.boxes:
            focus_occurrences[box.focus_id].append((decision.frame_idx, box))

    ranked_ids = sorted(
        focus_occurrences,
        key=lambda focus_id: (
            -len(focus_occurrences[focus_id]),
            -max(item[1].score for item in focus_occurrences[focus_id]),
            focus_id,
        ),
    )[:max_boxes]

    labels_in_order = [label for label, _ in label_counts.most_common()]
    specs = [
        TagSpec(
            tag=dominant_label,
            start_time=int(shot.start_ms),
            end_time=int(shot.end_ms),
            track=vertical_track,
            frame_info={"frame_idx": int(shot.start_frame_idx)},
            additional_info={"x-coordinates": x_coordinates},
        ),
        TagSpec(
            tag=dominant_label,
            start_time=int(shot.start_ms),
            end_time=int(shot.end_ms),
            track=focus_track,
            frame_info={"frame_idx": int(shot.start_frame_idx)},
            additional_info={
                "focus_ids": ranked_ids,
                "labels": labels_in_order,
                "confidence": round(average_confidence, decimals),
                **policy_metadata,
            },
        ),
    ]

    for focus_id in ranked_ids:
        occurrences = focus_occurrences[focus_id]
        # Representative box is the highest-confidence observation, avoiding
        # a noisy average that might no longer enclose the tracked subject.
        frame_idx, representative = max(occurrences, key=lambda item: (item[1].score, -item[0]))
        x1, y1, x2, y2 = sanitize_box(representative.box)
        specs.append(
            TagSpec(
                tag=representative.label,
                start_time=int(shot.start_ms),
                end_time=int(shot.end_ms),
                track=bbox_track,
                frame_info={
                    "frame_idx": int(frame_idx),
                    "box": {
                        "x1": round(x1, decimals),
                        "y1": round(y1, decimals),
                        "x2": round(x2, decimals),
                        "y2": round(y2, decimals),
                    },
                },
                additional_info={
                    "focus_id": focus_id,
                    "confidence": round(float(representative.score), decimals),
                },
            )
        )

    if not ranked_ids:
        # A safe-center fallback is still a real focus region: the fixed-width,
        # full-height 9:16 crop window. This preserves all three output tracks.
        center = clamp_crop_center(x_coordinates[len(x_coordinates) // 2], shot.crop_width)
        half = shot.crop_width * 0.5
        specs.append(
            TagSpec(
                tag="safe_center",
                start_time=int(shot.start_ms),
                end_time=int(shot.end_ms),
                track=bbox_track,
                frame_info={
                    "frame_idx": int(shot.start_frame_idx),
                    "box": {
                        "x1": round(clamp(center - half, 0.0, 1.0), decimals),
                        "y1": 0.0,
                        "x2": round(clamp(center + half, 0.0, 1.0), decimals),
                        "y2": 1.0,
                    },
                },
                additional_info={"focus_id": "safe_center", "confidence": round(average_confidence, decimals)},
            )
        )
    return specs


def validate_tag_spec(spec: TagSpec) -> None:
    if spec.end_time <= spec.start_time:
        raise ValueError("tag end_time must be greater than start_time")
    if not spec.track or not spec.tag:
        raise ValueError("tag and track must be non-empty")
    frame_idx = spec.frame_info.get("frame_idx")
    if not isinstance(frame_idx, int) or frame_idx < 0:
        raise ValueError("frame_info.frame_idx must be a non-negative integer")
    xs = spec.additional_info.get("x-coordinates")
    if xs is not None:
        if not isinstance(xs, list) or not xs:
            raise ValueError("x-coordinates must be a non-empty list")
        if any(not isinstance(x, (int, float)) or x < 0.0 or x > 1.0 for x in xs):
            raise ValueError("x-coordinates must contain normalized finite numbers")
