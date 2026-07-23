from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

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
    """Deterministic detector-to-detector association tracker.

    Association uses bounded predicted boxes. The normal ball gate remains
    narrow; a wider reacquisition gate is available only for an established
    ball track and a high-confidence detection after a miss.
    """

    def __init__(
        self,
        *,
        max_missed_updates: int = 4,
        max_missed_updates_by_label: Optional[Mapping[str, int]] = None,
        match_center_distance: float = 0.18,
        ball_match_center_distance: float = 0.18,
        ball_reacquire_center_distance: float = 0.28,
        ball_reacquire_min_confidence: float = 0.55,
        ball_reacquire_min_hits: int = 2,
        min_iou: float = 0.03,
        box_alpha: float = 0.68,
        ball_box_alpha: float = 0.90,
        box_alpha_by_label: Optional[Mapping[str, float]] = None,
        velocity_alpha_by_label: Optional[Mapping[str, float]] = None,
        prediction_max_frames: int = 10,
        ball_prediction_max_frames: int = 22,
        prediction_max_distance: float = 0.10,
    ) -> None:
        self.max_missed_updates = max(0, int(max_missed_updates))
        self.max_missed_updates_by_label = {
            str(label): max(0, int(value))
            for label, value in (max_missed_updates_by_label or {}).items()
        }
        self.match_center_distance = float(match_center_distance)
        self.ball_match_center_distance = float(ball_match_center_distance)
        self.ball_reacquire_center_distance = max(
            self.ball_match_center_distance,
            float(ball_reacquire_center_distance),
        )
        self.ball_reacquire_min_confidence = float(ball_reacquire_min_confidence)
        self.ball_reacquire_min_hits = max(1, int(ball_reacquire_min_hits))
        self.min_iou = float(min_iou)
        self.box_alpha = float(box_alpha)
        self.ball_box_alpha = float(ball_box_alpha)
        self.box_alpha_by_label = {
            str(label): clamp(float(value), 0.0, 1.0)
            for label, value in (box_alpha_by_label or {}).items()
        }
        self.velocity_alpha_by_label = {
            str(label): clamp(float(value), 0.0, 1.0)
            for label, value in (velocity_alpha_by_label or {}).items()
        }
        self.prediction_max_frames = max(0, int(prediction_max_frames))
        self.ball_prediction_max_frames = max(0, int(ball_prediction_max_frames))
        self.prediction_max_distance = max(0.0, float(prediction_max_distance))
        self._next_id = 1
        self._tracks: Dict[int, _Track] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def _max_missed_for(self, label: str) -> int:
        return self.max_missed_updates_by_label.get(
            label, self.max_missed_updates
        )

    def _box_alpha_for(self, label: str) -> float:
        if label in self.box_alpha_by_label:
            return self.box_alpha_by_label[label]
        return self.ball_box_alpha if label == "ball" else self.box_alpha

    def _velocity_alpha_for(self, label: str) -> float:
        if label in self.velocity_alpha_by_label:
            return self.velocity_alpha_by_label[label]
        return 0.78 if label == "ball" else 0.58

    def _predicted_box(self, track: _Track, frame_idx: int) -> BBox:
        delta = max(0, frame_idx - track.last_detection_frame)
        if delta <= 0:
            return track.box
        x1, y1, x2, y2 = track.box
        horizon = (
            self.ball_prediction_max_frames
            if track.label == "ball"
            else self.prediction_max_frames
        )
        prediction_delta = min(delta, horizon)
        dx = clamp(
            track.vx_per_frame * prediction_delta,
            -self.prediction_max_distance,
            self.prediction_max_distance,
        )
        dy = clamp(
            track.vy_per_frame * prediction_delta,
            -self.prediction_max_distance,
            self.prediction_max_distance,
        )
        width = x2 - x1
        height = y2 - y1
        px1 = clamp(x1 + dx, 0.0, 1.0 - width)
        py1 = clamp(y1 + dy, 0.0, 1.0 - height)
        return sanitize_box((px1, py1, px1 + width, py1 + height))

    def update(self, detections: Iterable[Detection], frame_idx: int) -> List[TrackView]:
        detections = list(detections)
        unmatched_tracks = set(self._tracks)
        matched_detection_indices = set()

        pairs: List[tuple[float, int, int]] = []
        for det_idx, detection in enumerate(detections):
            for track_id, track in self._tracks.items():
                if track.label != detection.label:
                    continue
                predicted_box = self._predicted_box(track, frame_idx)
                overlap = iou(predicted_box, detection.box)
                distance = center_distance(predicted_box, detection.box)
                if track.label == "ball":
                    distance_limit = self.ball_match_center_distance
                    if (
                        track.misses > 0
                        and track.hits >= self.ball_reacquire_min_hits
                        and float(detection.score) >= self.ball_reacquire_min_confidence
                    ):
                        distance_limit = self.ball_reacquire_center_distance
                else:
                    distance_limit = self.match_center_distance
                if overlap < self.min_iou and distance > distance_limit:
                    continue
                distance_weight = 0.95 if track.label == "ball" else 1.5
                cost = (1.0 - overlap) + distance_weight * distance + 0.04 * track.misses
                pairs.append((cost, track_id, det_idx))

        for _, track_id, det_idx in sorted(pairs):
            if track_id not in unmatched_tracks or det_idx in matched_detection_indices:
                continue
            track = self._tracks[track_id]
            detection = detections[det_idx]
            old_cx, old_cy = box_center(track.box)
            new_cx, new_cy = box_center(detection.box)
            frame_delta = max(1, frame_idx - track.last_detection_frame)
            alpha = self._box_alpha_for(track.label)
            predicted_box = self._predicted_box(track, frame_idx)
            blended = sanitize_box(
                tuple(
                    alpha * new + (1.0 - alpha) * predicted
                    for new, predicted in zip(detection.box, predicted_box)
                )  # type: ignore[arg-type]
            )
            measured_vx = (new_cx - old_cx) / frame_delta
            measured_vy = (new_cy - old_cy) / frame_delta
            velocity_alpha = self._velocity_alpha_for(track.label)
            track.vx_per_frame = (
                velocity_alpha * measured_vx
                + (1.0 - velocity_alpha) * track.vx_per_frame
            )
            track.vy_per_frame = (
                velocity_alpha * measured_vy
                + (1.0 - velocity_alpha) * track.vy_per_frame
            )
            track.box = blended
            track.score = 0.75 * float(detection.score) + 0.25 * track.score
            track.hits += 1
            track.misses = 0
            track.last_detection_frame = frame_idx
            unmatched_tracks.remove(track_id)
            matched_detection_indices.add(det_idx)

        for track_id in unmatched_tracks:
            track = self._tracks[track_id]
            track.misses += 1
            track.score *= 0.92 if track.label == "ball" else 0.88

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
            if track.misses <= self._max_missed_for(track.label)
        }
        return self.snapshot(frame_idx)

    def snapshot(self, frame_idx: int) -> List[TrackView]:
        views: List[TrackView] = []
        for track in self._tracks.values():
            delta = max(0, frame_idx - track.last_detection_frame)
            views.append(
                TrackView(
                    track_id=track.track_id,
                    label=track.label,
                    score=max(0.0, min(1.0, track.score * (0.975**delta))),
                    box=self._predicted_box(track, frame_idx),
                    hits=track.hits,
                    misses=track.misses,
                    last_detection_frame=track.last_detection_frame,
                    vx_per_frame=track.vx_per_frame,
                    vy_per_frame=track.vy_per_frame,
                    observed_this_frame=track.last_detection_frame == frame_idx,
                )
            )
        return sorted(views, key=lambda item: item.track_id)
