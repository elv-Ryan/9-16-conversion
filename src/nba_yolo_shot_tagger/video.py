from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np

from .types import VideoInfo


class VideoSource:
    def __init__(self, path: str) -> None:
        self.path = str(path)
        if not Path(self.path).is_file():
            raise FileNotFoundError(f"input media does not exist: {self.path}")
        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open input media: {self.path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"invalid source fps for {self.path}: {fps}")
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(
                f"invalid video metadata for {self.path}: "
                f"frames={frame_count} width={width} height={height}"
            )
        self.info = VideoInfo(
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_ms=int(round(1000.0 * frame_count / fps)),
        )

    @staticmethod
    def sample_indices(start_frame: int, end_frame: int, source_fps: float, inference_fps: float) -> List[int]:
        if end_frame <= start_frame:
            return []
        effective = min(source_fps, inference_fps)
        step = source_fps / effective
        values = np.arange(start_frame, end_frame, step, dtype=np.float64)
        indices = sorted({min(end_frame - 1, max(start_frame, int(round(value)))) for value in values})
        if not indices:
            indices = [start_frame]
        if indices[-1] != end_frame - 1:
            indices.append(end_frame - 1)
        return indices

    def iter_sample_batches(
        self,
        *,
        start_frame: int,
        end_frame: int,
        inference_fps: float,
        batch_size: int,
    ) -> Iterator[Tuple[List[int], List[np.ndarray]]]:
        targets = self.sample_indices(start_frame, end_frame, self.info.fps, inference_fps)
        if not targets:
            return
        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not reopen input media: {self.path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        target_position = 0
        batch_indices: List[int] = []
        batch_frames: List[np.ndarray] = []
        try:
            while target_position < len(targets):
                target = targets[target_position]
                while current_frame < target:
                    if not capture.grab():
                        raise EOFError(
                            f"unexpected end of video at frame {current_frame}; target={target}"
                        )
                    current_frame += 1
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise EOFError(f"failed to decode requested frame {target}")
                current_frame += 1
                batch_indices.append(target)
                batch_frames.append(frame)
                target_position += 1
                if len(batch_frames) >= batch_size:
                    yield batch_indices, batch_frames
                    batch_indices, batch_frames = [], []
            if batch_frames:
                yield batch_indices, batch_frames
        finally:
            capture.release()
