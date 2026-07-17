from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .geometry import box_center, center_distance, clamp, iou, sanitize_box
from .types import BBox, Detection, TrackView


@dataclass
class _Track:
    numeric_id: int
    label: str
    score: float
    box: BBox
    hits: int
    misses: int
    last_detection_frame: int
    vx_per_frame: float = 0.0
    vy_per_frame: float = 0.0

    @property
    def track_id(self) -> str:
        return f"{self.label}:{self.numeric_id}"


class TrackManager:
    """Small deterministic detector-to-detector association tracker."""

    def __init__(
        self,
        *,
        max_missed_updates: int = 4,
        match_center_distance: float = 0.18,
        min_iou: float = 0.03,
        box_alpha: float = 0.68,
    ) -> None:
        self.max_missed_updates = int(max_missed_updates)
        self.match_center_distance = float(match_center_distance)
        self.min_iou = float(min_iou)
        self.box_alpha = float(box_alpha)
        self._next_id = 1
        self._tracks: Dict[int, _Track] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, detections: Iterable[Detection], frame_idx: int) -> List[TrackView]:
        detections = list(detections)
        unmatched_tracks = set(self._tracks)
        matched_detection_indices = set()

        # Globally sort candidate pairs so association is deterministic.
        pairs: List[tuple[float, int, int]] = []
        for det_idx, detection in enumerate(detections):
            for track_id, track in self._tracks.items():
                if track.label != detection.label:
                    continue
                overlap = iou(track.box, detection.box)
                distance = center_distance(track.box, detection.box)
                if overlap < self.min_iou and distance > self.match_center_distance:
                    continue
                cost = (1.0 - overlap) + 1.5 * distance
                pairs.append((cost, track_id, det_idx))

        for _, track_id, det_idx in sorted(pairs):
            if track_id not in unmatched_tracks or det_idx in matched_detection_indices:
                continue
            track = self._tracks[track_id]
            detection = detections[det_idx]
            old_cx, old_cy = box_center(track.box)
            new_cx, new_cy = box_center(detection.box)
            frame_delta = max(1, frame_idx - track.last_detection_frame)
            alpha = self.box_alpha
            blended = sanitize_box(
                tuple(
                    alpha * new + (1.0 - alpha) * old
                    for new, old in zip(detection.box, track.box)
                )  # type: ignore[arg-type]
            )
            track.vx_per_frame = (new_cx - old_cx) / frame_delta
            track.vy_per_frame = (new_cy - old_cy) / frame_delta
            track.box = blended
            track.score = 0.7 * float(detection.score) + 0.3 * track.score
            track.hits += 1
            track.misses = 0
            track.last_detection_frame = frame_idx
            unmatched_tracks.remove(track_id)
            matched_detection_indices.add(det_idx)

        for track_id in unmatched_tracks:
            track = self._tracks[track_id]
            track.misses += 1
            track.score *= 0.90

        for det_idx, detection in enumerate(detections):
            if det_idx in matched_detection_indices:
                continue
            numeric_id = self._next_id
            self._next_id += 1
            self._tracks[numeric_id] = _Track(
                numeric_id=numeric_id,
                label=detection.label,
                score=float(detection.score),
                box=sanitize_box(detection.box),
                hits=1,
                misses=0,
                last_detection_frame=frame_idx,
            )

        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.misses <= self.max_missed_updates
        }
        return self.snapshot(frame_idx)

    def snapshot(self, frame_idx: int) -> List[TrackView]:
        views: List[TrackView] = []
        for track in self._tracks.values():
            delta = max(0, frame_idx - track.last_detection_frame)
            if delta:
                x1, y1, x2, y2 = track.box
                dx = clamp(track.vx_per_frame * delta, -0.08, 0.08)
                dy = clamp(track.vy_per_frame * delta, -0.08, 0.08)
                width = x2 - x1
                height = y2 - y1
                px1 = clamp(x1 + dx, 0.0, 1.0 - width)
                py1 = clamp(y1 + dy, 0.0, 1.0 - height)
                box = sanitize_box((px1, py1, px1 + width, py1 + height))
            else:
                box = track.box
            views.append(
                TrackView(
                    track_id=track.track_id,
                    label=track.label,
                    score=max(0.0, min(1.0, track.score * (0.97**delta))),
                    box=box,
                    hits=track.hits,
                    misses=track.misses,
                )
            )
        return sorted(views, key=lambda item: item.track_id)
