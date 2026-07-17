from __future__ import annotations

import math
from typing import Iterable, Sequence

from .types import BBox


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def valid_box(box: BBox) -> bool:
    x1, y1, x2, y2 = box
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def sanitize_box(box: BBox) -> BBox:
    x1, y1, x2, y2 = box
    x1 = clamp(float(x1), 0.0, 1.0)
    y1 = clamp(float(y1), 0.0, 1.0)
    x2 = clamp(float(x2), 0.0, 1.0)
    y2 = clamp(float(y2), 0.0, 1.0)
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-6)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-6)
    return x1, y1, x2, y2


def box_width(box: BBox) -> float:
    return max(0.0, box[2] - box[0])


def box_height(box: BBox) -> float:
    return max(0.0, box[3] - box[1])


def box_area(box: BBox) -> float:
    return box_width(box) * box_height(box)


def box_center(box: BBox) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def center_distance(a: BBox, b: BBox) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return math.hypot(ax - bx, ay - by)


def point_box_distance(x: float, y: float, box: BBox) -> float:
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return math.hypot(dx, dy)


def iou(a: BBox, b: BBox) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def union_box(boxes: Iterable[BBox]) -> BBox:
    values = list(boxes)
    if not values:
        return (0.5, 0.5, 0.5 + 1e-6, 0.5 + 1e-6)
    return sanitize_box(
        (
            min(b[0] for b in values),
            min(b[1] for b in values),
            max(b[2] for b in values),
            max(b[3] for b in values),
        )
    )


def weighted_center_x(boxes: Sequence[tuple[BBox, float]]) -> float:
    if not boxes:
        return 0.5
    total = sum(max(1e-6, weight) for _, weight in boxes)
    return sum(box_center(box)[0] * max(1e-6, weight) for box, weight in boxes) / total


def crop_width_normalized(width: int, height: int) -> float:
    """Width of a full-height 9:16 crop, normalized by source width."""
    if width <= 0 or height <= 0:
        return 1.0
    return clamp((9.0 / 16.0) * (float(height) / float(width)), 0.0, 1.0)


def clamp_crop_center(center_x: float, crop_width: float) -> float:
    half = clamp(crop_width * 0.5, 0.0, 0.5)
    return clamp(float(center_x), half, 1.0 - half)
