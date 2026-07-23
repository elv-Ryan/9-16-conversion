from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import List, Optional
import os

import cv2
from loguru import logger
import numpy as np
import requests


@dataclass(frozen=True)
class ShotRange:
    shot_id: str
    start_ms: int
    end_ms: int


class TagStoreShots:
    """Paginated shot_detection reader with O(log n) frame lookup."""

    def __init__(self, shots: List[ShotRange]) -> None:
        self.shots = sorted(shots, key=lambda shot: (shot.start_ms, shot.end_ms, shot.shot_id))
        self._starts = [shot.start_ms for shot in self.shots]

    @property
    def available(self) -> bool:
        return bool(self.shots)

    def at(self, global_time_ms: int) -> Optional[ShotRange]:
        if not self.shots:
            return None
        index = bisect_right(self._starts, int(global_time_ms)) - 1
        if index < 0:
            return None
        shot = self.shots[index]
        return shot if shot.start_ms <= global_time_ms < shot.end_ms else None

    @classmethod
    def from_environment(
        cls,
        *,
        base_url: str,
        track: str,
        timeout_seconds: float,
        page_size: int = 1000,
    ) -> "TagStoreShots":
        iq = os.getenv("ELV_CONTENT")
        token = os.getenv("ELV_TOKEN")
        if not iq or not token:
            logger.warning("ELV_CONTENT/ELV_TOKEN not set; using local shot detection")
            return cls([])

        endpoint = f"{base_url.rstrip('/')}/tagstore/{iq}/tags"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        start = 0
        rows: List[dict] = []
        with requests.Session() as session:
            while True:
                response = session.get(
                    endpoint,
                    headers=headers,
                    params={"track": track, "start": start, "limit": page_size},
                    timeout=float(timeout_seconds),
                )
                response.raise_for_status()
                payload = response.json()
                page = payload.get("tags") or payload.get("data") or []
                if not isinstance(page, list):
                    raise ValueError("Tagstore response does not contain a tag list")
                rows.extend(page)
                total = (payload.get("meta") or {}).get("total")
                start += len(page)
                if not page or (isinstance(total, int) and start >= total) or len(page) < page_size:
                    break

        shots: List[ShotRange] = []
        for index, row in enumerate(rows):
            try:
                start_ms = int(row["start_time"])
                end_ms = int(row["end_time"])
            except (KeyError, TypeError, ValueError):
                continue
            if end_ms <= start_ms:
                continue
            raw_id = row.get("id") or f"shot_{index:06d}_{start_ms}_{end_ms}"
            shots.append(ShotRange(str(raw_id), start_ms, end_ms))
        logger.info("Loaded {} shot ranges from track {}", len(shots), track)
        return cls(shots)


class LocalShotDetector:
    """Fallback hard-cut detector used when Tagstore shots are unavailable.

    HSV histogram distance remains the primary cue. A conjunctive grayscale
    mean-absolute-difference and edge-change cue catches same-palette cuts while
    rejecting most flashes and continuous camera motion.
    """

    def __init__(
        self,
        *,
        threshold: float,
        minimum_seconds: float,
        fps: float,
        gray_mad_threshold: float = 0.18,
        edge_change_threshold: float = 0.20,
        alignment_response_threshold: float = 0.45,
        moderate_histogram_threshold: Optional[float] = None,
        moderate_gray_mad_threshold: float = 0.14,
        moderate_edge_change_threshold: float = 0.50,
        moderate_alignment_response_threshold: float = 0.40,
    ) -> None:
        self.threshold = float(threshold)
        self.gray_mad_threshold = max(0.0, float(gray_mad_threshold))
        self.edge_change_threshold = max(0.0, float(edge_change_threshold))
        self.alignment_response_threshold = min(
            1.0, max(0.0, float(alignment_response_threshold))
        )
        self.moderate_histogram_threshold = (
            None
            if moderate_histogram_threshold is None
            else max(0.0, float(moderate_histogram_threshold))
        )
        self.moderate_gray_mad_threshold = max(
            0.0, float(moderate_gray_mad_threshold)
        )
        self.moderate_edge_change_threshold = max(
            0.0, float(moderate_edge_change_threshold)
        )
        self.moderate_alignment_response_threshold = min(
            1.0, max(0.0, float(moderate_alignment_response_threshold))
        )
        self.minimum_frames = max(1, int(round(float(minimum_seconds) * fps)))
        self._previous_hist: Optional[np.ndarray] = None
        self._previous_gray: Optional[np.ndarray] = None
        self._previous_edges: Optional[np.ndarray] = None
        self._last_cut_frame = 0
        self.shot_index = 0

    def reset(self) -> None:
        self._previous_hist = None
        self._previous_gray = None
        self._previous_edges = None
        self._last_cut_frame = 0
        self.shot_index = 0

    def update(self, rgb: np.ndarray, frame_idx: int) -> bool:
        small = cv2.resize(rgb, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 72, 144)

        if self._previous_hist is None:
            self._previous_hist = hist
            self._previous_gray = gray
            self._previous_edges = edges
            return False

        histogram_distance = cv2.compareHist(
            self._previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA
        )
        assert self._previous_gray is not None
        assert self._previous_edges is not None
        gray_mad = float(np.mean(cv2.absdiff(self._previous_gray, gray))) / 255.0
        previous_edge_mask = self._previous_edges > 0
        edge_mask = edges > 0
        edge_union = int(np.count_nonzero(previous_edge_mask | edge_mask))
        edge_disagreement = int(np.count_nonzero(previous_edge_mask ^ edge_mask))
        edge_change = edge_disagreement / max(1, edge_union)
        _, alignment_response = cv2.phaseCorrelate(
            self._previous_gray.astype(np.float32), gray.astype(np.float32)
        )

        self._previous_hist = hist
        self._previous_gray = gray
        self._previous_edges = edges

        enough_spacing = frame_idx - self._last_cut_frame >= self.minimum_frames
        secondary_cut = (
            gray_mad >= self.gray_mad_threshold
            and edge_change >= self.edge_change_threshold
            and alignment_response <= self.alignment_response_threshold
        )
        moderate_cut = (
            self.moderate_histogram_threshold is not None
            and histogram_distance >= self.moderate_histogram_threshold
            and gray_mad >= self.moderate_gray_mad_threshold
            and edge_change >= self.moderate_edge_change_threshold
            and alignment_response
            <= self.moderate_alignment_response_threshold
        )
        if enough_spacing and (
            histogram_distance >= self.threshold
            or secondary_cut
            or moderate_cut
        ):
            self._last_cut_frame = frame_idx
            self.shot_index += 1
            return True
        return False
