from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .types import VideoInfo


@dataclass(frozen=True)
class VideoFrame:
    frame_idx: int
    time_ms: int
    rgb: np.ndarray


class VideoReader:
    """OpenCV video reader with monotonic presentation timestamps."""

    def __init__(self, path: str) -> None:
        if not Path(path).is_file():
            raise FileNotFoundError(f"input file does not exist: {path}")
        self.path = path
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise ValueError(f"could not open video: {path}")
        width = int(round(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        frame_count = int(round(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        duration_ms = int(round(frame_count * 1000.0 / fps)) if frame_count > 0 else 0
        self.info = VideoInfo(width, height, fps, duration_ms, frame_count)
        self._last_time_ms = -1
        self._decoded = 0

    def __iter__(self) -> Iterator[VideoFrame]:
        try:
            frame_idx = 0
            while True:
                ok, bgr = self.capture.read()
                if not ok:
                    break
                reported_ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
                fallback_ms = frame_idx * 1000.0 / self.info.fps
                time_ms = int(round(reported_ms if np.isfinite(reported_ms) and reported_ms >= 0 else fallback_ms))
                if self._last_time_ms >= 0 and time_ms <= self._last_time_ms:
                    time_ms = max(self._last_time_ms + 1, int(round(fallback_ms)))
                self._last_time_ms = time_ms
                self._decoded = frame_idx + 1
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                yield VideoFrame(frame_idx=frame_idx, time_ms=time_ms, rgb=rgb)
                frame_idx += 1
        finally:
            self.close()

    @property
    def effective_duration_ms(self) -> int:
        frame_duration = int(round(1000.0 / self.info.fps))
        decoded_duration = self._last_time_ms + frame_duration if self._last_time_ms >= 0 else 0
        return max(self.info.duration_ms, decoded_duration)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None  # type: ignore[assignment]
