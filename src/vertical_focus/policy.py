from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .geometry import (
    box_area,
    box_center,
    clamp,
    iou,
    point_box_distance,
    union_box,
    weighted_center_x,
)
from .types import Detection, FocusBox, FocusCandidate, SelectedFocus, TrackView


@dataclass(frozen=True)
class _PolicyTiming:
    min_hold_s: float
    lost_hold_s: float
    switch_margin: float


class FocusPolicy:
    """Shot-local sports/movie focus selection with identity hysteresis."""

    def __init__(self, mode: str, config: Dict) -> None:
        if mode not in {"sports", "movie"}:
            raise ValueError(f"Unsupported focus policy: {mode!r}")
        self.mode = mode
        self.config = config
        temporal = config.get("temporal", {})
        self.timing = _PolicyTiming(
            min_hold_s=float(temporal.get("min_hold_seconds", 0.6)),
            lost_hold_s=float(temporal.get("lost_hold_seconds", 0.35)),
            switch_margin=float(temporal.get("switch_margin", 0.14)),
        )
        self._current: Optional[SelectedFocus] = None
        self._current_since_s = 0.0
        self._last_seen_s = 0.0

    def reset(self) -> None:
        self._current = None
        self._current_since_s = 0.0
        self._last_seen_s = 0.0

    def select(
        self,
        tracks: Sequence[TrackView],
        motion: Optional[Detection],
        *,
        crop_width: float,
        time_s: float,
    ) -> SelectedFocus:
        if self.mode == "sports":
            candidates = self._sports_candidates(tracks, motion, crop_width)
        else:
            candidates = self._movie_candidates(tracks, motion, crop_width)

        candidates.append(FocusCandidate("safe_center", "safe_center", 0.05, 0.5, ()))
        by_key = {candidate.key: candidate for candidate in candidates}
        top = max(candidates, key=lambda candidate: (candidate.score, candidate.key))

        if self._current is None:
            return self._switch_to(top, time_s)

        current_candidate = by_key.get(self._current.key)
        if current_candidate is not None:
            self._last_seen_s = time_s
            current = self._to_selected(current_candidate)
            self._current = current
            if top.key == current.key:
                return current
            held_for = time_s - self._current_since_s
            if held_for < self.timing.min_hold_s:
                return current
            if top.score < current.score + self.timing.switch_margin:
                return current
            return self._switch_to(top, time_s)

        if time_s - self._last_seen_s <= self.timing.lost_hold_s:
            # Preserve the previous identity briefly through missed detections.
            return SelectedFocus(
                key=self._current.key,
                label=self._current.label,
                score=max(0.0, self._current.score * 0.96),
                center_x=self._current.center_x,
                boxes=self._current.boxes,
            )

        return self._switch_to(top, time_s)

    def _switch_to(self, candidate: FocusCandidate, time_s: float) -> SelectedFocus:
        selected = self._to_selected(candidate)
        self._current = selected
        self._current_since_s = time_s
        self._last_seen_s = time_s
        return selected

    @staticmethod
    def _to_selected(candidate: FocusCandidate) -> SelectedFocus:
        return SelectedFocus(
            key=candidate.key,
            label=candidate.label,
            score=clamp(candidate.score, 0.0, 1.0),
            center_x=clamp(candidate.center_x, 0.0, 1.0),
            boxes=candidate.boxes,
        )

    @staticmethod
    def _focus_box(track: TrackView) -> FocusBox:
        return FocusBox(track.track_id, track.label, track.score, track.box)

    @staticmethod
    def _motion_box(motion: Detection) -> FocusBox:
        return FocusBox("motion", "motion", motion.score, motion.box)

    def _track_salience(self, track: TrackView, motion: Optional[Detection], *, movie: bool) -> float:
        area = box_area(track.box)
        center_x, center_y = box_center(track.box)
        center_prior = 1.0 - min(1.0, abs(center_x - 0.5) * 1.7)
        if track.label == "face":
            area_reference = 0.035 if movie else 0.05
        else:
            area_reference = 0.11 if movie else 0.075
        area_score = min(1.0, math.sqrt(max(0.0, area) / area_reference))
        motion_score = 0.0
        if motion is not None:
            motion_score = max(iou(track.box, motion.box), math.exp(-8.0 * abs(center_x - box_center(motion.box)[0])))
        stable = min(1.0, track.hits / 3.0)
        vertical_prior = clamp((center_y - 0.20) / 0.65, 0.0, 1.0)
        if movie:
            return 0.32 * track.score + 0.34 * area_score + 0.14 * center_prior + 0.12 * stable + 0.08 * motion_score
        return 0.30 * track.score + 0.24 * area_score + 0.10 * center_prior + 0.14 * stable + 0.14 * motion_score + 0.08 * vertical_prior

    def _sports_candidates(
        self,
        tracks: Sequence[TrackView],
        motion: Optional[Detection],
        crop_width: float,
    ) -> List[FocusCandidate]:
        cfg = self.config.get("selection", {})
        min_person_area = float(cfg.get("min_person_area", 0.0025))
        direct_ball_area = float(cfg.get("direct_ball_min_area", 0.004))
        ball_player_radius = float(cfg.get("ball_player_radius", 0.22))
        group_radius = float(cfg.get("group_radius", 0.24))
        max_group = int(cfg.get("max_group_size", 3))
        min_hits = int(cfg.get("min_track_hits", 2))

        persons = [t for t in tracks if t.label == "person" and box_area(t.box) >= min_person_area]
        balls = [t for t in tracks if t.label == "ball"]
        faces = [t for t in tracks if t.label == "face"]
        candidates: List[FocusCandidate] = []

        ball = max(balls, key=lambda t: t.score * (0.2 + math.sqrt(max(box_area(t.box), 1e-8))), default=None)
        ball_center = box_center(ball.box) if ball is not None else None

        person_scores: Dict[str, float] = {}
        for person in persons:
            score = self._track_salience(person, motion, movie=False)
            # Small upper-frame detections are usually crowd/bench noise.
            if box_area(person.box) < 0.008 and person.box[3] < 0.58:
                score *= 0.48
            person_scores[person.track_id] = score

        person_areas = sorted(box_area(person.box) for person in persons)
        median_person_area = person_areas[len(person_areas) // 2] if person_areas else 0.0
        crowd_mode = (
            ball is None
            and len(persons) >= int(cfg.get("crowd_min_people", 4))
            and median_person_area <= float(cfg.get("crowd_max_median_area", 0.015))
            and (motion is None or motion.score < float(cfg.get("crowd_max_motion_score", 0.20)))
        )
        if crowd_mode:
            crowd = sorted(
                persons,
                key=lambda person: (box_area(person.box) * person.score, person.track_id),
                reverse=True,
            )[:max_group]
            candidates.append(
                FocusCandidate(
                    key="crowd_group:" + ":".join(sorted(person.track_id for person in crowd)),
                    label="crowd_group",
                    score=float(cfg.get("crowd_group_score", 0.42)),
                    center_x=weighted_center_x(tuple((person.box, max(box_area(person.box), 1e-6)) for person in crowd)),
                    boxes=tuple(self._focus_box(person) for person in crowd),
                )
            )

        if ball is not None and ball_center is not None:
            associated: List[tuple[float, TrackView]] = []
            for person in persons:
                distance = point_box_distance(ball_center[0], ball_center[1], person.box)
                association = math.exp(-distance / max(0.02, ball_player_radius))
                score = 0.45 * association + 0.40 * person_scores[person.track_id] + 0.15 * ball.score
                associated.append((score, person))
            if associated:
                association_score, player = max(associated, key=lambda item: item[0])
                if association_score >= float(cfg.get("min_ball_player_score", 0.48)):
                    center_x = weighted_center_x(((player.box, 0.84), (ball.box, 0.16)))
                    boxes = (self._focus_box(player), self._focus_box(ball))
                    candidates.append(
                        FocusCandidate(
                            key=f"player_near_ball:{player.track_id}:{ball.track_id}",
                            label="player_near_ball",
                            score=min(1.0, association_score + 0.12),
                            center_x=center_x,
                            boxes=boxes,
                        )
                    )

            if box_area(ball.box) >= direct_ball_area:
                candidates.append(
                    FocusCandidate(
                        key=f"ball:{ball.track_id}",
                        label="ball",
                        score=0.50 * ball.score + 0.20,
                        center_x=ball_center[0],
                        boxes=(self._focus_box(ball),),
                    )
                )

        anchor_x: Optional[float] = ball_center[0] if ball_center else None
        if anchor_x is None and motion is not None:
            anchor_x = box_center(motion.box)[0]
        if anchor_x is not None and persons:
            group = [p for p in persons if abs(box_center(p.box)[0] - anchor_x) <= group_radius]
            group = sorted(group, key=lambda p: person_scores[p.track_id], reverse=True)[:max_group]
            if len(group) >= 2:
                boxes = tuple(self._focus_box(person) for person in group)
                center_x = weighted_center_x(tuple((p.box, person_scores[p.track_id]) for p in group))
                group_score = sum(person_scores[p.track_id] for p in group) / len(group)
                candidates.append(
                    FocusCandidate(
                        key="action_group:" + ":".join(sorted(p.track_id for p in group)),
                        label="action_group",
                        score=min(1.0, group_score + 0.13),
                        center_x=center_x,
                        boxes=boxes,
                    )
                )

        for person in persons:
            if crowd_mode and box_area(person.box) <= float(cfg.get("crowd_max_median_area", 0.015)):
                continue
            if person.hits < min_hits and (self._current is None or person.track_id not in self._current.key):
                continue
            candidates.append(
                FocusCandidate(
                    key=f"active_player:{person.track_id}",
                    label="active_player",
                    score=person_scores[person.track_id],
                    center_x=box_center(person.box)[0],
                    boxes=(self._focus_box(person),),
                )
            )

        announcer_min_area = float(cfg.get("announcer_face_min_area", 0.025))
        for face in faces:
            if box_area(face.box) < announcer_min_area:
                continue
            score = self._track_salience(face, motion, movie=True)
            candidates.append(
                FocusCandidate(
                    key=f"announcer_face:{face.track_id}",
                    label="announcer_face",
                    score=min(1.0, score + (0.10 if ball is None else 0.0)),
                    center_x=box_center(face.box)[0],
                    boxes=(self._focus_box(face),),
                )
            )

        if motion is not None and motion.score >= float(cfg.get("min_motion_score", 0.08)):
            candidates.append(
                FocusCandidate(
                    key="action_region",
                    label="action_region",
                    score=0.30 + 0.35 * motion.score,
                    center_x=box_center(motion.box)[0],
                    boxes=(self._motion_box(motion),),
                )
            )
        return candidates

    def _movie_candidates(
        self,
        tracks: Sequence[TrackView],
        motion: Optional[Detection],
        crop_width: float,
    ) -> List[FocusCandidate]:
        cfg = self.config.get("selection", {})
        min_hits = int(cfg.get("min_track_hits", 2))
        min_face_area = float(cfg.get("min_face_area", 0.0015))
        faces = [t for t in tracks if t.label == "face" and box_area(t.box) >= min_face_area]
        persons = [t for t in tracks if t.label == "person"]
        candidates: List[FocusCandidate] = []

        face_scores = {face.track_id: self._track_salience(face, motion, movie=True) for face in faces}
        stable_faces = [
            face
            for face in faces
            if face.hits >= min_hits or (self._current is not None and face.track_id in self._current.key)
        ]
        stable_faces.sort(key=lambda face: face_scores[face.track_id], reverse=True)

        if len(stable_faces) >= 2:
            top_score = face_scores[stable_faces[0].track_id]
            comparable = [
                face
                for face in stable_faces[:3]
                if face_scores[face.track_id] >= top_score * float(cfg.get("group_score_ratio", 0.72))
            ]
            if len(comparable) >= 2:
                combined = union_box(face.box for face in comparable)
                max_group_width = max(crop_width * float(cfg.get("group_crop_width_factor", 1.05)), 0.18)
                if combined[2] - combined[0] <= max_group_width:
                    boxes = tuple(self._focus_box(face) for face in comparable)
                    center_x = weighted_center_x(tuple((face.box, face_scores[face.track_id]) for face in comparable))
                    candidates.append(
                        FocusCandidate(
                            key="interaction_group:" + ":".join(sorted(face.track_id for face in comparable)),
                            label="interaction_group",
                            score=min(1.0, sum(face_scores[f.track_id] for f in comparable) / len(comparable) + 0.10),
                            center_x=center_x,
                            boxes=boxes,
                        )
                    )

        for face in stable_faces:
            candidates.append(
                FocusCandidate(
                    key=f"primary_face:{face.track_id}",
                    label="primary_face",
                    score=face_scores[face.track_id],
                    center_x=box_center(face.box)[0],
                    boxes=(self._focus_box(face),),
                )
            )

        if not stable_faces:
            person_scores = {person.track_id: self._track_salience(person, motion, movie=True) for person in persons}
            for person in persons:
                if person.hits < min_hits and (self._current is None or person.track_id not in self._current.key):
                    continue
                candidates.append(
                    FocusCandidate(
                        key=f"primary_person:{person.track_id}",
                        label="primary_person",
                        score=0.86 * person_scores[person.track_id],
                        center_x=box_center(person.box)[0],
                        boxes=(self._focus_box(person),),
                    )
                )

        if motion is not None and motion.score >= float(cfg.get("min_motion_score", 0.10)):
            candidates.append(
                FocusCandidate(
                    key="action_region",
                    label="action_region",
                    score=0.26 + 0.34 * motion.score,
                    center_x=box_center(motion.box)[0],
                    boxes=(self._motion_box(motion),),
                )
            )
        return candidates
