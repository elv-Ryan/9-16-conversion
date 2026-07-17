from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .geometry import sanitize_box
from .types import Detection


class MotionEstimator:
    """Produces one coarse normalized action region from frame differencing."""

    def __init__(self, *, scale_width: int = 320, threshold: int = 22, min_area: float = 0.002) -> None:
        self.scale_width = max(96, int(scale_width))
        self.threshold = int(threshold)
        self.min_area = float(min_area)
        self._previous: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._previous = None

    def update(self, rgb: np.ndarray) -> Optional[Detection]:
        height, width = rgb.shape[:2]
        if width <= 0 or height <= 0:
            return None
        scaled_height = max(1, round(height * self.scale_width / width))
        small = cv2.resize(rgb, (self.scale_width, scaled_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._previous is None:
            self._previous = gray
            return None

        diff = cv2.absdiff(gray, self._previous)
        self._previous = gray
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(self.scale_width * scaled_height)
        kept = [c for c in contours if cv2.contourArea(c) / frame_area >= self.min_area]
        if not kept:
            return None

        kept = sorted(kept, key=cv2.contourArea, reverse=True)[:6]
        x1 = min(cv2.boundingRect(c)[0] for c in kept)
        y1 = min(cv2.boundingRect(c)[1] for c in kept)
        x2 = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in kept)
        y2 = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in kept)
        total_area = sum(cv2.contourArea(c) for c in kept) / frame_area
        return Detection(
            label="motion",
            score=min(1.0, total_area * 8.0),
            box=sanitize_box(
                (
                    x1 / self.scale_width,
                    y1 / scaled_height,
                    x2 / self.scale_width,
                    y2 / scaled_height,
                )
            ),
            source="motion",
        )
