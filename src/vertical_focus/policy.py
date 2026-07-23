from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
class _TemporalSettings:
    min_hold_s: float
    lost_hold_s: float
    switch_margin: float
    candidate_persistence_s: float
    hold_last_on_empty: bool


@dataclass
class _BallMemory:
    track_id: str
    center_x: float
    center_y: float
    vx_per_s: float
    vy_per_s: float
    last_observed_s: float
    last_hits: int
    box: Tuple[float, float, float, float]
    score: float

    def predicted_center(
        self,
        time_s: float,
        horizon_s: float,
        max_displacement: float,
    ) -> tuple[float, float]:
        dt = clamp(time_s - self.last_observed_s, 0.0, horizon_s)
        limit = max(0.0, float(max_displacement))
        dx = clamp(self.vx_per_s * dt, -limit, limit)
        dy = clamp(self.vy_per_s * dt, -limit, limit)
        return (
            clamp(self.center_x + dx, 0.0, 1.0),
            clamp(self.center_y + dy, 0.0, 1.0),
        )


@dataclass(frozen=True)
class _CharacterEvidence:
    candidate: FocusCandidate
    person: Optional[TrackView]
    face_or_head: Optional[TrackView]
    anchor_id: str


class _PolicyBase:
    """Shared temporal infrastructure, not shared behavior policy."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        temporal = config.get("temporal", {})
        self.timing = _TemporalSettings(
            min_hold_s=float(temporal.get("min_hold_seconds", 0.6)),
            lost_hold_s=float(temporal.get("lost_hold_seconds", 0.35)),
            switch_margin=float(temporal.get("switch_margin", 0.14)),
            candidate_persistence_s=float(
                temporal.get("candidate_persistence_seconds", 0.0)
            ),
            hold_last_on_empty=bool(temporal.get("hold_last_on_empty", True)),
        )
        self._current: Optional[SelectedFocus] = None
        self._current_since_s = 0.0
        self._last_seen_s = 0.0
        self._pending_key: Optional[str] = None
        self._pending_since_s = 0.0

    def reset(self) -> None:
        self._current = None
        self._current_since_s = 0.0
        self._last_seen_s = 0.0
        self._pending_key = None
        self._pending_since_s = 0.0

    @staticmethod
    def _focus_box(track: TrackView) -> FocusBox:
        return FocusBox(track.track_id, track.label, track.score, track.box)

    @staticmethod
    def _motion_box(motion: Detection) -> FocusBox:
        return FocusBox("motion", "motion", motion.score, motion.box)

    @staticmethod
    def _to_selected(candidate: FocusCandidate) -> SelectedFocus:
        return SelectedFocus(
            key=candidate.key,
            label=candidate.label,
            score=clamp(candidate.score, 0.0, 1.0),
            center_x=clamp(candidate.center_x, 0.0, 1.0),
            boxes=candidate.boxes,
            scene_state=candidate.scene_state,
            evidence_kind=candidate.evidence_kind,
            smoothing_regime=candidate.smoothing_regime,
        )

    def _switch_to(self, candidate: FocusCandidate, time_s: float) -> SelectedFocus:
        selected = self._to_selected(candidate)
        self._current = selected
        self._current_since_s = time_s
        self._last_seen_s = time_s
        self._pending_key = None
        return selected

    def _hold_current(
        self,
        *,
        scene_state: str,
        evidence_kind: str = "held_composition",
        label: Optional[str] = None,
        preserve_boxes: bool = True,
    ) -> SelectedFocus:
        if self._current is None:
            return SelectedFocus(
                key=f"scene_hold:{scene_state}",
                label=label or "safe_center",
                score=0.10,
                center_x=0.5,
                boxes=(),
                scene_state=scene_state,
                evidence_kind="safe_center",
                smoothing_regime="locked",
            )
        held = SelectedFocus(
            key=self._current.key,
            label=label or self._current.label,
            score=max(0.01, self._current.score * 0.995),
            center_x=self._current.center_x,
            boxes=self._current.boxes if preserve_boxes else (),
            scene_state=scene_state,
            evidence_kind=evidence_kind,
            smoothing_regime="locked",
        )
        self._current = held
        return held

    def _choose_with_hysteresis(
        self,
        candidates: Sequence[FocusCandidate],
        *,
        time_s: float,
        min_hold_s: Optional[float] = None,
        margin: Optional[float] = None,
        persistence_s: Optional[float] = None,
        immediate_keys: Iterable[str] = (),
    ) -> Optional[SelectedFocus]:
        if not candidates:
            return None
        top = max(candidates, key=lambda item: (item.score, item.key))
        if self._current is None:
            return self._switch_to(top, time_s)

        by_key = {candidate.key: candidate for candidate in candidates}
        current_candidate = by_key.get(self._current.key)
        immediate = top.key in set(immediate_keys)
        if current_candidate is not None:
            self._last_seen_s = time_s
            current = self._to_selected(current_candidate)
            self._current = current
            if top.key == current.key:
                self._pending_key = None
                return current
            if immediate:
                return self._switch_to(top, time_s)
            held_for = time_s - self._current_since_s
            effective_hold = self.timing.min_hold_s if min_hold_s is None else min_hold_s
            effective_margin = self.timing.switch_margin if margin is None else margin
            if held_for < effective_hold or top.score < current.score + effective_margin:
                self._pending_key = None
                return current
        else:
            if immediate:
                return self._switch_to(top, time_s)
            if time_s - self._last_seen_s <= self.timing.lost_hold_s:
                return self._hold_current(
                    scene_state=self._current.scene_state,
                    evidence_kind="temporarily_lost",
                )

        effective_persistence = (
            self.timing.candidate_persistence_s
            if persistence_s is None
            else persistence_s
        )
        if effective_persistence > 0.0:
            if self._pending_key != top.key:
                self._pending_key = top.key
                self._pending_since_s = time_s
                return self._current
            if time_s - self._pending_since_s < effective_persistence:
                return self._current
        return self._switch_to(top, time_s)


class SportsFocusPolicy(_PolicyBase):
    """Gameplay-state and trusted-ball-continuity sports policy."""

    LIVE_STATES = frozenset({"live_gameplay", "replay_like"})

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.selection = config.get("selection", {})
        self.state_cfg = config.get("scene_state", {})
        self._scene_state = "weak_evidence"
        self._pending_scene_state: Optional[str] = None
        self._pending_scene_since_s = 0.0
        self._last_live_evidence_s = -1e9
        self._ball_memory: Optional[_BallMemory] = None
        self._gameplay_anchor_x: Optional[float] = None
        self._pending_gameplay_anchor_x: Optional[float] = None
        self._pending_gameplay_anchor_since_s = 0.0

    def reset(self) -> None:
        super().reset()
        self._scene_state = "weak_evidence"
        self._pending_scene_state = None
        self._pending_scene_since_s = 0.0
        self._last_live_evidence_s = -1e9
        self._ball_memory = None
        self._gameplay_anchor_x = None
        self._pending_gameplay_anchor_x = None
        self._pending_gameplay_anchor_since_s = 0.0

    def select(
        self,
        tracks: Sequence[TrackView],
        motion: Optional[Detection],
        *,
        crop_width: float,
        time_s: float,
    ) -> SelectedFocus:
        del crop_width
        persons = self._sports_persons(tracks)
        faces = [
            track
            for track in tracks
            if track.label in {"face", "head"} and track.misses <= 1
        ]
        ball_candidate = self._trusted_ball_candidate(tracks, persons, time_s)
        instant_state, urgent_live = self._classify_scene(
            persons, faces, motion, ball_candidate
        )
        state = self._update_scene_state(
            instant_state, time_s, urgent_live=urgent_live
        )

        if ball_candidate is not None:
            if state == "replay_like" and ball_candidate.smoothing_regime == "fast_reacquire":
                ball_candidate = replace(ball_candidate, smoothing_regime="ball_follow")
            ball_candidate = replace(ball_candidate, scene_state=state)
            self._gameplay_anchor_x = ball_candidate.center_x
            self._pending_gameplay_anchor_x = None
            if self._current is None or self._current.key != ball_candidate.key:
                return self._switch_to(ball_candidate, time_s)
            selected = self._to_selected(ball_candidate)
            self._current = selected
            self._last_seen_s = time_s
            return selected

        if state in self.LIVE_STATES:
            gameplay = self._gameplay_region_candidate(persons, motion, state, time_s)
            if gameplay is not None:
                if self._current is not None and self._current.key == "sports_ball":
                    return self._switch_to(gameplay, time_s)
                selected = self._choose_with_hysteresis(
                    [gameplay],
                    time_s=time_s,
                    min_hold_s=float(
                        self.config.get("temporal", {}).get(
                            "gameplay_min_hold_seconds", 0.28
                        )
                    ),
                    margin=0.04,
                    persistence_s=0.0,
                )
                if selected is not None:
                    return selected
            return self._hold_current(
                scene_state=state,
                evidence_kind="live_without_local_evidence",
                label="gameplay_cluster",
                preserve_boxes=False,
            )

        self._gameplay_anchor_x = None
        self._pending_gameplay_anchor_x = None
        temporal = self.config.get("temporal", {})
        if state == "closeup_player":
            selected = self._choose_with_hysteresis(
                self._closeup_candidates(persons, state),
                time_s=time_s,
                min_hold_s=float(temporal.get("non_gameplay_min_hold_seconds", 0.70)),
                margin=float(temporal.get("non_gameplay_switch_margin", 0.18)),
                persistence_s=float(
                    temporal.get("non_gameplay_persistence_seconds", 0.35)
                ),
            )
            if selected is not None:
                return selected

        if state == "announcer_or_studio":
            selected = self._choose_with_hysteresis(
                self._announcer_candidates(persons, faces, state),
                time_s=time_s,
                min_hold_s=float(temporal.get("non_gameplay_min_hold_seconds", 0.70)),
                margin=float(temporal.get("non_gameplay_switch_margin", 0.18)),
                persistence_s=float(
                    temporal.get("non_gameplay_persistence_seconds", 0.35)
                ),
            )
            if selected is not None:
                return selected

        locked_label = "safe_center" if state == "weak_evidence" else state
        return self._hold_current(
            scene_state=state,
            label=locked_label,
            preserve_boxes=False,
        )

    def _sports_persons(self, tracks: Sequence[TrackView]) -> List[TrackView]:
        min_area = float(self.selection.get("min_person_area", 0.0025))
        return [
            track
            for track in tracks
            if track.label == "person"
            and box_area(track.box) >= min_area
            and track.misses <= 1
        ]

    def _person_salience(self, person: TrackView) -> float:
        area = box_area(person.box)
        center_x, center_y = box_center(person.box)
        area_score = min(1.0, math.sqrt(max(area, 0.0) / 0.08))
        center_prior = 1.0 - min(1.0, abs(center_x - 0.5) * 1.6)
        vertical_prior = clamp((center_y - 0.18) / 0.68, 0.0, 1.0)
        stable = min(1.0, person.hits / 4.0)
        return (
            0.31 * person.score
            + 0.30 * area_score
            + 0.15 * center_prior
            + 0.14 * stable
            + 0.10 * vertical_prior
        )

    def _nearest_person(
        self,
        center: tuple[float, float],
        persons: Sequence[TrackView],
    ) -> tuple[Optional[TrackView], float, float]:
        radius = max(0.02, float(self.selection.get("ball_player_radius", 0.18)))
        best: Optional[tuple[float, float, TrackView]] = None
        for person in persons:
            distance = point_box_distance(center[0], center[1], person.box)
            association = math.exp(-distance / radius)
            score = 0.70 * association + 0.30 * self._person_salience(person)
            row = (score, distance, person)
            if best is None or row[0] > best[0]:
                best = row
        if best is None:
            return None, 1.0, 0.0
        return best[2], best[1], best[0]

    def _ball_context_persons(
        self, persons: Sequence[TrackView]
    ) -> List[TrackView]:
        """Exclude tiny border people from ball-possession evidence.

        Wide broadcast frames contain spectators, bench fragments, and clipped
        detections along the lower border.  A sports-ball false positive inside
        one of those boxes can otherwise satisfy the same proximity rule as a
        real dribble.  Full-height court players remain valid even when their
        feet touch the frame boundary.
        """

        cfg = self.selection
        max_top = float(cfg.get("ball_context_person_max_top", 0.82))
        max_center_y = float(
            cfg.get("ball_context_person_max_center_y", 0.90)
        )
        max_bottom = float(
            cfg.get("ball_context_person_max_bottom", 0.985)
        )
        min_clipped_height = float(
            cfg.get("ball_context_min_clipped_height", 0.16)
        )
        valid: List[TrackView] = []
        for person in persons:
            top = float(person.box[1])
            bottom = float(person.box[3])
            center_y = 0.5 * (top + bottom)
            height = max(0.0, bottom - top)
            if top > max_top or center_y > max_center_y:
                continue
            if bottom > max_bottom and height < min_clipped_height:
                continue
            valid.append(person)
        return valid

    def _ball_path(self, time_s: float) -> Optional[tuple[float, float]]:
        if self._ball_memory is None:
            return None
        age = time_s - self._ball_memory.last_observed_s
        if age > float(self.selection.get("ball_forget_seconds", 0.65)):
            return None
        return self._ball_memory.predicted_center(
            time_s,
            float(self.selection.get("ball_prediction_seconds", 0.30)),
            float(self.selection.get("ball_prediction_max_displacement", 0.10)),
        )

    def _trusted_ball_candidate(
        self,
        tracks: Sequence[TrackView],
        persons: Sequence[TrackView],
        time_s: float,
    ) -> Optional[FocusCandidate]:
        cfg = self.selection
        min_area = float(cfg.get("ball_min_area", 0.00001))
        max_area = float(cfg.get("ball_max_area", 0.030))
        balls = [
            track
            for track in tracks
            if track.label == "ball"
            and min_area <= box_area(track.box) <= max_area
            and track.misses <= int(cfg.get("ball_max_misses", 4))
        ]
        if not balls and self._ball_memory is None:
            return None

        observed_areas = [box_area(ball.box) for ball in balls if ball.observed]
        raw_reference_area = max(
            observed_areas or [box_area(ball.box) for ball in balls] or [min_area]
        )
        reference_area = min(
            raw_reference_area,
            float(cfg.get("ball_context_reference_area_cap", 0.0008)),
        )
        predicted_path = self._ball_path(time_s)
        memory = self._ball_memory
        prediction_seconds = float(cfg.get("ball_prediction_seconds", 0.30))
        prediction_max_displacement = float(
            cfg.get("ball_prediction_max_displacement", 0.10)
        )
        forget_seconds = float(cfg.get("ball_forget_seconds", 0.65))
        memory_age = (
            time_s - memory.last_observed_s if memory is not None else float("inf")
        )
        memory_fresh = memory is not None and memory_age <= forget_seconds

        context_persons = self._ball_context_persons(persons)
        ranked: List[
            tuple[float, TrackView, Optional[TrackView], float, float, bool]
        ] = []
        for ball in balls:
            area = box_area(ball.box)
            center = box_center(ball.box)
            player, player_distance, context_score = self._nearest_person(
                center, context_persons
            )
            context_area_ok = area >= float(
                cfg.get("ball_context_min_area", 0.00015)
            )
            persistent_area_ok = area >= float(
                cfg.get("ball_persistent_min_area", 0.000075)
            )
            same_track = memory is not None and ball.track_id == memory.track_id
            path_distance = (
                math.hypot(
                    center[0] - predicted_path[0], center[1] - predicted_path[1]
                )
                if predicted_path is not None
                else 0.0
            )
            near_path = (
                predicted_path is not None
                and path_distance <= float(cfg.get("ball_reacquire_distance", 0.18))
            )
            observed = ball.observed
            new_or_far = not same_track and not near_path
            if observed and new_or_far:
                if center[1] > float(cfg.get("ball_new_max_center_y", 0.92)):
                    continue
                if ball.box[3] > float(cfg.get("ball_new_max_bottom", 0.985)):
                    continue
                if area > float(cfg.get("ball_live_max_area", 0.008)):
                    continue

            if player is not None:
                person_area = max(box_area(player.box), 1e-8)
                if area / person_area > float(
                    cfg.get("ball_max_person_area_ratio", 0.18)
                ):
                    player = None
                    player_distance = 1.0
                    context_score = 0.0

            context_close = player_distance <= float(
                cfg.get("ball_new_context_radius", 0.18)
            )
            tight_context = player_distance <= float(
                cfg.get("ball_tight_context_radius", 0.03)
            )

            velocity = math.hypot(ball.vx_per_frame, ball.vy_per_frame)
            low_confidence_motion = (
                float(cfg.get("ball_low_confidence_min_velocity_per_frame", 0.00035))
                <= velocity
                <= float(cfg.get("ball_low_confidence_max_velocity_per_frame", 0.060))
            )
            context_velocity_max = float(
                cfg.get("ball_context_max_velocity_per_frame", 0.012)
            )
            contextual_motion = (
                tight_context
                and context_area_ok
                and velocity >= float(
                    cfg.get("ball_low_confidence_min_velocity_per_frame", 0.00035)
                )
                and velocity <= context_velocity_max
            )
            contextual_geometry = tight_context and context_area_ok
            area_competitive = area >= float(
                cfg.get("ball_context_min_relative_area", 0.50)
            ) * reference_area
            immediate_context = (
                contextual_geometry
                and area_competitive
                and (
                    ball.score
                    >= float(cfg.get("ball_context_immediate_confidence", 0.30))
                    or (
                        ball.hits >= 2
                        and ball.score
                        >= float(cfg.get("ball_context_two_hit_confidence", 0.12))
                        and contextual_motion
                    )
                    or (
                        ball.hits >= 3
                        and ball.score
                        >= float(cfg.get("ball_context_continue_confidence", 0.07))
                        and contextual_motion
                    )
                )
            )
            eligible = False

            if same_track:
                if observed:
                    normal_continue = (
                        ball.hits >= int(cfg.get("ball_min_track_hits", 2))
                        and ball.score >= float(cfg.get("ball_continue_confidence", 0.16))
                        and (
                            predicted_path is None
                            or path_distance
                            <= float(cfg.get("ball_continue_distance", 0.12))
                        )
                    )
                    contextual_continue = (
                        ball.hits >= 2
                        and ball.score
                        >= float(cfg.get("ball_context_continue_confidence", 0.07))
                        and contextual_motion
                        and (
                            predicted_path is None
                            or path_distance
                            <= float(cfg.get("ball_reacquire_distance", 0.18))
                        )
                    )
                    high_conf_motion = (
                        ball.hits >= 2
                        and ball.score >= float(cfg.get("ball_fast_min_confidence", 0.55))
                        and near_path
                    )
                    eligible = normal_continue or contextual_continue or high_conf_motion
                else:
                    eligible = memory_age <= prediction_seconds
            elif observed and near_path:
                # A new numeric ID near the semantic path is not sufficient by
                # itself. Low-confidence static court marks can survive many
                # tracker updates. Require motion, tight player interaction, or
                # a genuinely high detector score before transferring identity.
                near_path_transfer = (
                    ball.score >= float(cfg.get("ball_new_high_confidence", 0.55))
                    or (
                        ball.hits >= 3
                        and ball.score
                        >= float(
                            cfg.get("ball_context_continue_confidence", 0.07)
                        )
                        and persistent_area_ok
                        and low_confidence_motion
                    )
                    or immediate_context
                )
                eligible = ball.hits >= 2 and near_path_transfer
            elif observed and (not memory_fresh or memory is None):
                hits = ball.hits
                score = ball.score
                strong_context = player_distance <= float(
                    cfg.get("ball_low_confidence_context_radius", 0.10)
                )
                eligible = immediate_context or (
                    (
                        context_close
                        and hits >= 2
                        and score >= float(cfg.get("ball_new_context_confidence", 0.35))
                        and (
                            contextual_geometry
                            or (persistent_area_ok and low_confidence_motion)
                        )
                    )
                    or (
                        hits >= 3
                        and score
                        >= float(cfg.get("ball_new_persistent_confidence", 0.20))
                        and persistent_area_ok
                        and low_confidence_motion
                        and (strong_context or hits >= 4)
                    )
                    or (
                        hits >= 3
                        and score
                        >= float(cfg.get("ball_context_continue_confidence", 0.07))
                        and contextual_motion
                    )
                    or (
                        hits >= 2
                        and score >= float(cfg.get("ball_new_high_confidence", 0.55))
                    )
                )
            elif observed and memory_fresh:
                high_confidence_reacquire = (
                    context_close
                    and ball.hits >= 2
                    and ball.score >= float(cfg.get("ball_new_high_confidence", 0.55))
                )
                contextual_reacquire = (
                    ball.hits >= 3
                    and ball.score
                    >= float(cfg.get("ball_far_context_confidence", 0.20))
                    and contextual_motion
                )
                # Strong ball-player geometry can invalidate a weak stale path
                # immediately. Otherwise preserve the gap requirement so a
                # single distant false positive cannot yank the crop.
                eligible = immediate_context or (
                    (high_confidence_reacquire or contextual_reacquire)
                    and memory_age
                    >= float(cfg.get("ball_far_reacquire_min_gap_seconds", 0.20))
                )

            if not eligible:
                continue
            continuity = (
                math.exp(
                    -path_distance
                    / max(0.02, float(cfg.get("ball_reacquire_distance", 0.18)))
                )
                if predicted_path is not None
                else 0.55
            )
            relative_area = math.sqrt(clamp(area / max(reference_area, min_area), 0.0, 1.0))
            close_context_score = math.exp(
                -player_distance
                / max(0.01, float(cfg.get("ball_tight_context_radius", 0.03)))
            )
            motion_floor = float(
                cfg.get("ball_low_confidence_min_velocity_per_frame", 0.00035)
            )
            motion_reference = max(
                motion_floor + 1e-6,
                float(cfg.get("ball_motion_reference_per_frame", 0.006)),
            )
            motion_support = clamp(
                (velocity - motion_floor) / (motion_reference - motion_floor),
                0.0,
                1.0,
            )
            quality = (
                0.28 * ball.score
                + 0.11 * min(1.0, ball.hits / 4.0)
                + 0.17 * continuity
                + 0.14 * context_score
                + 0.14 * relative_area
                + 0.10 * motion_support
                + (0.11 if same_track else 0.0)
                + (0.04 if observed else -0.04)
                + 0.07 * close_context_score
            )
            ranked.append(
                (quality, ball, player, player_distance, path_distance, same_track)
            )

        chosen = max(ranked, key=lambda row: (row[0], row[1].score), default=None)
        if chosen is None:
            if memory is None or memory_age > prediction_seconds:
                return None
            predicted_x, predicted_y = memory.predicted_center(
                time_s,
                prediction_seconds,
                prediction_max_displacement,
            )
            width = memory.box[2] - memory.box[0]
            height = memory.box[3] - memory.box[1]
            x1 = clamp(predicted_x - width / 2.0, 0.0, 1.0 - width)
            y1 = clamp(predicted_y - height / 2.0, 0.0, 1.0 - height)
            decay = clamp(
                1.0 - memory_age / max(prediction_seconds, 1e-6), 0.0, 1.0
            )
            return FocusCandidate(
                key="sports_ball",
                label="ball_focus",
                score=0.72 + 0.20 * decay,
                center_x=predicted_x,
                boxes=(
                    FocusBox(
                        memory.track_id,
                        "ball",
                        memory.score * (0.85 + 0.15 * decay),
                        (x1, y1, x1 + width, y1 + height),
                    ),
                ),
                scene_state="live_gameplay",
                evidence_kind="predicted_ball",
                smoothing_regime="ball_follow",
            )

        quality, ball, player, player_distance, path_distance, same_track = chosen
        center_x, center_y = box_center(ball.box)
        area = box_area(ball.box)
        memory_before = self._ball_memory
        gap_before = (
            time_s - memory_before.last_observed_s
            if memory_before is not None
            else 0.0
        )
        reacquired = (
            ball.observed
            and ball.hits >= 2
            and memory_before is not None
            and (
                ball.track_id != memory_before.track_id
                or gap_before >= float(cfg.get("ball_reacquire_gap_seconds", 0.15))
            )
        )
        long_consistent_move = (
            ball.observed
            and memory_before is not None
            and same_track
            and path_distance >= float(cfg.get("ball_fast_error_threshold", 0.08))
            and ball.score >= float(cfg.get("ball_fast_min_confidence", 0.55))
        )

        if ball.observed and (
            memory_before is None
            or ball.track_id != memory_before.track_id
            or ball.hits != memory_before.last_hits
        ):
            vx = 0.0
            vy = 0.0
            if memory_before is not None and same_track:
                dt = max(1e-3, time_s - memory_before.last_observed_s)
                max_velocity = float(cfg.get("ball_max_velocity_per_second", 0.90))
                measured_vx = clamp(
                    (center_x - memory_before.center_x) / dt,
                    -max_velocity,
                    max_velocity,
                )
                measured_vy = clamp(
                    (center_y - memory_before.center_y) / dt,
                    -max_velocity,
                    max_velocity,
                )
                velocity_alpha = clamp(
                    float(cfg.get("ball_velocity_alpha", 0.45)), 0.0, 1.0
                )
                vx = (
                    velocity_alpha * measured_vx
                    + (1.0 - velocity_alpha) * memory_before.vx_per_s
                )
                vy = (
                    velocity_alpha * measured_vy
                    + (1.0 - velocity_alpha) * memory_before.vy_per_s
                )
            self._ball_memory = _BallMemory(
                track_id=ball.track_id,
                center_x=center_x,
                center_y=center_y,
                vx_per_s=vx,
                vy_per_s=vy,
                last_observed_s=time_s,
                last_hits=ball.hits,
                box=ball.box,
                score=ball.score,
            )

        # A small bounded velocity lead compensates for detector cadence and
        # causal camera acceleration without allowing a stale or reacquired
        # track to project across the court.
        if (
            ball.observed
            and same_track
            and not reacquired
            and ball.hits >= int(cfg.get("ball_lead_min_hits", 3))
            and self._ball_memory is not None
        ):
            lead = clamp(
                self._ball_memory.vx_per_s
                * float(cfg.get("ball_lead_seconds", 0.06)),
                -float(cfg.get("ball_lead_max_offset", 0.045)),
                float(cfg.get("ball_lead_max_offset", 0.045)),
            )
            center_x = clamp(center_x + lead, 0.0, 1.0)

        # Player proximity remains contextual evidence for trust and for a tiny
        # composition offset.  Only the ball is exposed as the selected focus
        # box so the QA overlay and compact focus_bbox track do not flicker
        # between the ball and changing nearby players.
        boxes = [self._focus_box(ball)]
        if player is not None and player_distance <= float(
            cfg.get("ball_player_context_max_distance", 0.24)
        ):
            ball_weight = clamp(float(cfg.get("ball_focus_weight", 0.96)), 0.90, 1.0)
            player_x = box_center(player.box)[0]
            raw_offset = (1.0 - ball_weight) * (player_x - center_x)
            offset = clamp(
                raw_offset,
                -float(cfg.get("ball_context_offset_max", 0.03)),
                float(cfg.get("ball_context_offset_max", 0.03)),
            )
            center_x = clamp(center_x + offset, 0.0, 1.0)

        target_error = abs(
            center_x - (self._current.center_x if self._current else center_x)
        )
        tight_context = player_distance <= float(
            cfg.get("ball_tight_context_radius", 0.03)
        )
        context_fast = (
            tight_context
            and area >= float(cfg.get("ball_context_min_area", 0.00015))
            and ball.hits >= 3
            and ball.score >= float(cfg.get("ball_far_context_confidence", 0.20))
        )
        fast_evidence = (
            ball.score >= float(cfg.get("ball_fast_min_confidence", 0.55))
            or context_fast
        )
        fast = (
            ball.observed
            and (reacquired or long_consistent_move)
            and fast_evidence
            and target_error >= float(cfg.get("ball_fast_error_threshold", 0.08))
        )
        return FocusCandidate(
            key="sports_ball",
            label="ball_focus",
            score=clamp(0.92 + 0.08 * quality, 0.0, 1.0),
            center_x=center_x,
            boxes=tuple(boxes),
            scene_state="live_gameplay",
            evidence_kind=(
                "reacquired_ball"
                if reacquired
                else ("observed_ball" if ball.observed else "predicted_ball")
            ),
            smoothing_regime="fast_reacquire" if fast else "ball_follow",
        )

    def _classify_scene(
        self,
        persons: Sequence[TrackView],
        faces: Sequence[TrackView],
        motion: Optional[Detection],
        ball_candidate: Optional[FocusCandidate],
    ) -> tuple[str, bool]:
        motion_score = float(motion.score) if motion is not None else 0.0
        areas = sorted(box_area(person.box) for person in persons)
        median_area = areas[len(areas) // 2] if areas else 0.0
        max_area = max(areas, default=0.0)

        if ball_candidate is not None:
            replay_like = (
                len(persons) <= int(self.state_cfg.get("replay_max_people", 3))
                and median_area
                >= float(self.state_cfg.get("replay_min_median_person_area", 0.020))
                and motion_score
                <= float(self.state_cfg.get("replay_max_motion_score", 0.18))
            )
            return ("replay_like" if replay_like else "live_gameplay", True)

        gameplay = (
            len(persons) >= int(self.state_cfg.get("gameplay_min_people", 4))
            and motion_score
            >= float(self.state_cfg.get("gameplay_min_motion_score", 0.12))
        ) or (
            len(persons) >= int(self.state_cfg.get("gameplay_force_people", 7))
            and motion_score
            >= float(self.state_cfg.get("gameplay_force_min_motion_score", 0.07))
        )
        if gameplay:
            return "live_gameplay", False

        announcer = (
            0 < len(persons) <= int(self.state_cfg.get("announcer_max_people", 2))
            and max_area >= float(self.state_cfg.get("announcer_min_person_area", 0.055))
            and motion_score
            <= float(self.state_cfg.get("announcer_max_motion_score", 0.08))
            and bool(faces)
        )
        if announcer:
            return "announcer_or_studio", False

        closeup = (
            0 < len(persons) <= int(self.state_cfg.get("closeup_max_people", 3))
            and max_area >= float(self.state_cfg.get("closeup_min_person_area", 0.045))
        )
        if closeup:
            return "closeup_player", False

        if (
            len(persons) >= int(self.state_cfg.get("crowd_min_people", 4))
            and median_area
            <= float(self.state_cfg.get("crowd_max_median_area", 0.015))
            and motion_score
            <= float(self.state_cfg.get("crowd_max_motion_score", 0.10))
        ):
            return "crowd_or_idle", False

        if (
            len(persons) >= int(self.state_cfg.get("timeout_min_people", 3))
            and motion_score
            <= float(self.state_cfg.get("timeout_max_motion_score", 0.08))
        ):
            return "timeout_or_bench", False

        if not persons and not faces and motion_score <= float(
            self.state_cfg.get("static_max_motion_score", 0.04)
        ):
            return "graphic_or_static", False
        return "weak_evidence", False

    def _update_scene_state(
        self,
        instant_state: str,
        time_s: float,
        *,
        urgent_live: bool,
    ) -> str:
        if instant_state in self.LIVE_STATES:
            self._last_live_evidence_s = time_s
        if (
            self._scene_state in self.LIVE_STATES
            and instant_state not in self.LIVE_STATES
            and time_s - self._last_live_evidence_s
            <= float(self.state_cfg.get("live_exit_hold_seconds", 0.80))
        ):
            return self._scene_state
        if instant_state == self._scene_state:
            self._pending_scene_state = None
            return self._scene_state
        if urgent_live:
            self._scene_state = instant_state
            self._pending_scene_state = None
            return self._scene_state
        if self._pending_scene_state != instant_state:
            self._pending_scene_state = instant_state
            self._pending_scene_since_s = time_s
            return self._scene_state
        required = (
            float(self.state_cfg.get("gameplay_enter_seconds", 0.25))
            if instant_state in self.LIVE_STATES
            else float(self.state_cfg.get("state_persistence_seconds", 0.35))
        )
        if time_s - self._pending_scene_since_s >= required:
            self._scene_state = instant_state
            self._pending_scene_state = None
        return self._scene_state

    def _stabilize_gameplay_center(
        self,
        raw_center_x: float,
        time_s: float,
        *,
        ball_anchored: bool,
    ) -> float:
        cfg = self.selection
        raw = clamp(raw_center_x, 0.0, 1.0)
        if self._gameplay_anchor_x is None:
            self._gameplay_anchor_x = raw
            self._pending_gameplay_anchor_x = None
            return raw

        prefix = "ball_anchored" if ball_anchored else "gameplay_anchor"
        deadband = float(cfg.get(f"{prefix}_deadband", 0.035 if ball_anchored else 0.060))
        persistence = float(
            cfg.get(f"{prefix}_persistence_seconds", 0.12 if ball_anchored else 0.30)
        )
        max_step = float(cfg.get(f"{prefix}_max_step", 0.16 if ball_anchored else 0.12))
        if abs(raw - self._gameplay_anchor_x) <= deadband:
            self._pending_gameplay_anchor_x = None
            return self._gameplay_anchor_x

        if (
            self._pending_gameplay_anchor_x is None
            or abs(raw - self._pending_gameplay_anchor_x) > deadband * 0.50
        ):
            self._pending_gameplay_anchor_x = raw
            self._pending_gameplay_anchor_since_s = time_s
            return self._gameplay_anchor_x

        if time_s - self._pending_gameplay_anchor_since_s < persistence:
            return self._gameplay_anchor_x

        delta = clamp(
            self._pending_gameplay_anchor_x - self._gameplay_anchor_x,
            -max_step,
            max_step,
        )
        self._gameplay_anchor_x = clamp(self._gameplay_anchor_x + delta, 0.0, 1.0)
        self._pending_gameplay_anchor_x = None
        return self._gameplay_anchor_x

    def _gameplay_region_candidate(
        self,
        persons: Sequence[TrackView],
        motion: Optional[Detection],
        state: str,
        time_s: float,
    ) -> Optional[FocusCandidate]:
        if not persons or motion is None:
            return None
        motion_score = float(motion.score)
        if motion_score < float(self.state_cfg.get("gameplay_min_motion_score", 0.12)):
            return None

        path = self._ball_path(time_s)
        anchor_x = path[0] if path is not None else box_center(motion.box)[0]
        radius = float(self.selection.get("gameplay_cluster_radius", 0.24))
        cluster = [
            person
            for person in persons
            if abs(box_center(person.box)[0] - anchor_x) <= radius
        ]
        if len(cluster) < 2:
            cluster = sorted(persons, key=self._person_salience, reverse=True)[:4]
        else:
            cluster = sorted(cluster, key=self._person_salience, reverse=True)[:4]
        if not cluster:
            return None
        player_center = weighted_center_x(
            tuple(
                (
                    person.box,
                    max(0.05, self._person_salience(person))
                    * math.sqrt(max(box_area(person.box), 1e-6)),
                )
                for person in cluster
            )
        )
        if path is not None:
            raw_center_x = (
                0.78 * anchor_x
                + 0.16 * player_center
                + 0.06 * box_center(motion.box)[0]
            )
            evidence_kind = "ball_anchored_action"
        else:
            raw_center_x = 0.68 * player_center + 0.32 * box_center(motion.box)[0]
            evidence_kind = "gameplay_region"
        center_x = self._stabilize_gameplay_center(
            raw_center_x,
            time_s,
            ball_anchored=path is not None,
        )
        return FocusCandidate(
            key="gameplay_cluster",
            label="gameplay_cluster",
            score=min(0.68, 0.50 + 0.12 * motion_score + 0.01 * len(cluster)),
            center_x=clamp(center_x, 0.0, 1.0),
            boxes=tuple(self._focus_box(person) for person in cluster),
            scene_state=state,
            evidence_kind=evidence_kind,
            smoothing_regime="normal_follow",
        )

    def _closeup_candidates(
        self, persons: Sequence[TrackView], state: str
    ) -> List[FocusCandidate]:
        min_hits = int(self.selection.get("min_track_hits", 3))
        min_area = float(self.state_cfg.get("closeup_min_person_area", 0.045))
        candidates: List[FocusCandidate] = []
        for person in persons:
            if box_area(person.box) < min_area or person.hits < min_hits:
                continue
            candidates.append(
                FocusCandidate(
                    key=f"closeup_player:{person.track_id}",
                    label="closeup_player",
                    score=min(0.66, 0.42 + 0.24 * self._person_salience(person)),
                    center_x=box_center(person.box)[0],
                    boxes=(self._focus_box(person),),
                    scene_state=state,
                    evidence_kind="closeup_person",
                    smoothing_regime="normal_follow",
                )
            )
        return candidates

    def _announcer_candidates(
        self,
        persons: Sequence[TrackView],
        faces: Sequence[TrackView],
        state: str,
    ) -> List[FocusCandidate]:
        min_hits = int(self.selection.get("min_track_hits", 3))
        candidates: List[FocusCandidate] = []
        for head in faces:
            if head.hits < min_hits or box_area(head.box) < float(
                self.state_cfg.get("announcer_min_head_area", 0.020)
            ):
                continue
            label_score = 0.66 if head.label == "face" else 0.52
            candidates.append(
                FocusCandidate(
                    key=f"announcer:{head.track_id}",
                    label="announcer_or_studio",
                    score=min(0.70, label_score + 0.08 * head.score),
                    center_x=box_center(head.box)[0],
                    boxes=(self._focus_box(head),),
                    scene_state=state,
                    evidence_kind=(
                        "announcer_face" if head.label == "face" else "announcer_head"
                    ),
                    smoothing_regime="normal_follow",
                )
            )
        if not candidates:
            candidates.extend(self._closeup_candidates(persons, state))
        return candidates


class MovieFocusPolicy(_PolicyBase):
    """Persistent-character movie policy with strong composition holds."""

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.selection = config.get("selection", {})
        self._stable_key: Optional[str] = None
        self._last_raw_target_x = 0.5
        self._stable_since_s = 0.0
        self._lock_anchor_x: Optional[float] = None
        self._unlock_pending_since_s: Optional[float] = None
        self._shot_start_s: Optional[float] = None

    def reset(self) -> None:
        super().reset()
        self._stable_key = None
        self._last_raw_target_x = 0.5
        self._stable_since_s = 0.0
        self._lock_anchor_x = None
        self._unlock_pending_since_s = None
        self._shot_start_s = None

    def select(
        self,
        tracks: Sequence[TrackView],
        motion: Optional[Detection],
        *,
        crop_width: float,
        time_s: float,
    ) -> SelectedFocus:
        if self._shot_start_s is None:
            self._shot_start_s = time_s
        initialization_window = float(
            self.config.get("temporal", {}).get("cut_initialization_seconds", 0.75)
        )
        initializing = (
            self._current is None
            or (
                self._current.evidence_kind == "safe_center"
                and time_s - self._shot_start_s <= initialization_window
            )
        )
        characters = self._character_evidence(tracks, initializing=initializing)
        candidates = [item.candidate for item in characters]
        group = self._interaction_group(characters, crop_width)
        if group is not None:
            candidates.append(group)

        if not candidates and motion is not None and motion.score >= float(
            self.selection.get("min_motion_score", 0.16)
        ):
            candidates.append(
                FocusCandidate(
                    key="movie_action_region",
                    label="action_region",
                    score=min(0.58, 0.26 + 0.30 * motion.score),
                    center_x=box_center(motion.box)[0],
                    boxes=(self._motion_box(motion),),
                    scene_state="movie_action",
                    evidence_kind="motion_only",
                    smoothing_regime="compose",
                )
            )

        if not candidates:
            if self._current is not None and self.timing.hold_last_on_empty:
                box_hold_s = float(
                    self.config.get("temporal", {}).get(
                        "focus_box_hold_seconds", 0.18
                    )
                )
                preserve_boxes = time_s - self._last_seen_s <= box_hold_s
                return self._hold_current(
                    scene_state="movie_hold",
                    evidence_kind=(
                        "short_focus_box_hold"
                        if preserve_boxes
                        else "no_evidence_hold"
                    ),
                    preserve_boxes=preserve_boxes,
                )
            centered = FocusCandidate(
                key="movie_safe_center",
                label="safe_center",
                score=0.05,
                center_x=0.5,
                boxes=(),
                scene_state="movie_weak_evidence",
                evidence_kind="safe_center",
                smoothing_regime="locked",
            )
            return self._switch_to(centered, time_s)

        top = max(candidates, key=lambda item: (item.score, item.key))
        if self._current is None:
            top = replace(top, smoothing_regime="cut_reset")
            selected = self._switch_to(top, time_s)
            self._reset_stability(selected.key, selected.center_x, time_s)
            self._lock_anchor_x = selected.center_x
            return selected

        if self._current.evidence_kind == "safe_center":
            direct_regime = (
                "cut_reset"
                if time_s - self._shot_start_s <= initialization_window
                else "confirmed_reframe"
            )
            top = replace(top, smoothing_regime=direct_regime)
            selected = self._switch_to(top, time_s)
            self._reset_stability(selected.key, selected.center_x, time_s)
            self._lock_anchor_x = selected.center_x
            return selected

        by_key = {candidate.key: candidate for candidate in candidates}
        current_candidate = by_key.get(self._current.key)
        if current_candidate is not None:
            self._last_seen_s = time_s
            if top.key == current_candidate.key:
                selected = self._apply_stability(current_candidate, time_s)
                self._current = selected
                self._pending_key = None
                return selected

            held_for = time_s - self._current_since_s
            margin, persistence = self._movie_switch_requirements(
                current_candidate, top
            )
            if (
                held_for < self.timing.min_hold_s
                or top.score < current_candidate.score + margin
            ):
                self._pending_key = None
                selected = self._apply_stability(current_candidate, time_s)
                self._current = selected
                return selected
            if self._pending_key != top.key:
                self._pending_key = top.key
                self._pending_since_s = time_s
                selected = self._apply_stability(current_candidate, time_s)
                self._current = selected
                return selected
            if time_s - self._pending_since_s < persistence:
                selected = self._apply_stability(current_candidate, time_s)
                self._current = selected
                return selected
        else:
            lost_for = time_s - self._last_seen_s
            strong_replacement = (
                top.evidence_kind == "frontal_face"
                and lost_for
                >= float(
                    self.config.get("temporal", {}).get(
                        "strong_face_replacement_delay_seconds", 0.15
                    )
                )
                and top.score
                >= self._current.score
                - float(
                    self.config.get("temporal", {}).get(
                        "strong_face_replacement_score_tolerance", 0.05
                    )
                )
            )
            if strong_replacement:
                top = replace(top, smoothing_regime="confirmed_reframe")
                selected = self._switch_to(top, time_s)
                self._reset_stability(selected.key, selected.center_x, time_s)
                return selected
            if lost_for <= self.timing.lost_hold_s:
                box_hold_s = float(
                    self.config.get("temporal", {}).get(
                        "focus_box_hold_seconds", 0.18
                    )
                )
                preserve_boxes = lost_for <= box_hold_s
                return self._hold_current(
                    scene_state="movie_hold",
                    evidence_kind=(
                        "short_focus_box_hold"
                        if preserve_boxes
                        else "temporary_face_or_person_loss"
                    ),
                    preserve_boxes=preserve_boxes,
                )
            persistence = float(
                self.config.get("temporal", {}).get(
                    "new_subject_persistence_seconds", 0.45
                )
            )
            if self._pending_key != top.key:
                self._pending_key = top.key
                self._pending_since_s = time_s
                return self._hold_current(
                    scene_state="movie_hold",
                    evidence_kind="pending_subject",
                    preserve_boxes=False,
                )
            if time_s - self._pending_since_s < persistence:
                return self._hold_current(
                    scene_state="movie_hold",
                    evidence_kind="pending_subject",
                    preserve_boxes=False,
                )

        top = replace(top, smoothing_regime="confirmed_reframe")
        selected = self._switch_to(top, time_s)
        self._reset_stability(selected.key, selected.center_x, time_s)
        return selected

    def _movie_switch_requirements(
        self, current: FocusCandidate, top: FocusCandidate
    ) -> tuple[float, float]:
        temporal = self.config.get("temporal", {})
        margin = float(temporal.get("switch_margin", 0.22))
        persistence = float(temporal.get("new_subject_persistence_seconds", 0.45))
        current_anchor = self._anchor_from_key(current.key)
        top_anchor = self._anchor_from_key(top.key)
        if current_anchor is not None and top_anchor == current_anchor:
            margin = float(temporal.get("same_character_switch_margin", 0.08))
            persistence = float(
                temporal.get("same_character_persistence_seconds", 0.20)
            )
        if top.evidence_kind in {"small_face", "unpaired_face"}:
            margin = max(
                margin, float(temporal.get("small_face_switch_margin", 0.30))
            )
            persistence = max(
                persistence,
                float(temporal.get("small_face_persistence_seconds", 0.45)),
            )
        return margin, persistence

    @staticmethod
    def _anchor_from_key(key: str) -> Optional[str]:
        for prefix in ("character:", "interaction_group:"):
            if key.startswith(prefix):
                return key[len(prefix) :]
        return None

    def _reset_stability(self, key: str, center_x: float, time_s: float) -> None:
        self._stable_key = key
        self._last_raw_target_x = center_x
        self._stable_since_s = time_s
        self._lock_anchor_x = None
        self._unlock_pending_since_s = None

    def _apply_stability(
        self, candidate: FocusCandidate, time_s: float
    ) -> SelectedFocus:
        temporal = self.config.get("temporal", {})
        jitter_delta = float(temporal.get("stable_target_delta", 0.010))
        hold_after = float(temporal.get("stable_hold_seconds", 0.30))
        unlock_error = float(temporal.get("unlock_error", 0.035))
        unlock_persistence = float(
            temporal.get("unlock_persistence_seconds", 0.15)
        )

        if self._stable_key != candidate.key:
            self._reset_stability(candidate.key, candidate.center_x, time_s)
            return self._to_selected(
                replace(candidate, smoothing_regime="confirmed_reframe")
            )

        raw_delta = abs(candidate.center_x - self._last_raw_target_x)
        if raw_delta > jitter_delta:
            self._stable_since_s = time_s
        self._last_raw_target_x = candidate.center_x

        if self._lock_anchor_x is None and time_s - self._stable_since_s >= hold_after:
            self._lock_anchor_x = (
                self._current.center_x if self._current is not None else candidate.center_x
            )
            self._unlock_pending_since_s = None

        if self._lock_anchor_x is not None:
            error = abs(candidate.center_x - self._lock_anchor_x)
            if error <= unlock_error:
                self._unlock_pending_since_s = None
                return self._to_selected(
                    replace(
                        candidate,
                        center_x=self._lock_anchor_x,
                        smoothing_regime="settle_then_lock",
                    )
                )
            if self._unlock_pending_since_s is None:
                self._unlock_pending_since_s = time_s
                return self._to_selected(
                    replace(
                        candidate,
                        center_x=self._lock_anchor_x,
                        smoothing_regime="settle_then_lock",
                    )
                )
            if time_s - self._unlock_pending_since_s < unlock_persistence:
                return self._to_selected(
                    replace(
                        candidate,
                        center_x=self._lock_anchor_x,
                        smoothing_regime="settle_then_lock",
                    )
                )
            self._lock_anchor_x = None
            self._unlock_pending_since_s = None
            self._stable_since_s = time_s
            return self._to_selected(
                replace(candidate, smoothing_regime="confirmed_reframe")
            )

        return self._to_selected(replace(candidate, smoothing_regime="compose"))

    def _character_evidence(
        self,
        tracks: Sequence[TrackView],
        *,
        initializing: bool = False,
    ) -> List[_CharacterEvidence]:
        cfg = self.selection
        min_hits = int(
            cfg.get("cut_initial_min_track_hits", 1)
            if initializing
            else cfg.get("min_track_hits", 3)
        )
        min_person_area = float(cfg.get("min_person_area", 0.004))
        persons = [
            track
            for track in tracks
            if track.label == "person"
            and box_area(track.box) >= min_person_area
            and track.misses <= int(cfg.get("person_max_misses", 2))
        ]
        faces = [
            track
            for track in tracks
            if track.label == "face" and track.misses <= 1
        ]
        heads = [
            track
            for track in tracks
            if track.label == "head" and track.misses <= 1
        ]
        current_anchor = self._anchor_from_key(self._current.key) if self._current else None

        paired_faces = self._pair_regions(faces, persons)
        paired_heads = self._pair_regions(heads, persons)
        has_frontal_face = bool(paired_faces)
        evidence: List[_CharacterEvidence] = []
        used_regions = set()

        for person in persons:
            continuation_anchor = self._continuation_anchor(person)
            anchor_id = continuation_anchor or person.track_id
            is_current = current_anchor == anchor_id
            if person.hits < min_hits and not is_current:
                continue
            face = paired_faces.get(person.track_id)
            head = paired_heads.get(person.track_id)
            region = face or head
            if region is not None:
                used_regions.add(region.track_id)
            person_score = self._movie_person_score(person, is_current=is_current)
            center_x = box_center(person.box)[0]
            # A face/head is evidence for identity and scoring, but the stable
            # enclosing person box is the user-visible focus region.  Exposing
            # both regions made the overlay and compact focus_bbox track flicker
            # between head and body geometry even when the selected character
            # did not change.
            boxes: List[FocusBox] = [self._focus_box(person)]
            evidence_kind = "person"
            label = "primary_person"
            score = 0.82 * person_score

            if face is not None:
                face_area = box_area(face.box)
                face_score = self._movie_face_score(face, is_current=is_current)
                small = face_area < float(
                    cfg.get("small_face_preemption_area", 0.006)
                )
                paired_person_large = box_area(person.box) >= float(
                    cfg.get("small_face_min_person_area", 0.030)
                )
                if face_area >= float(cfg.get("current_face_min_area", 0.0012)) and (
                    is_current or face_area >= float(cfg.get("new_face_min_area", 0.0020))
                ):
                    face_weight = clamp(
                        float(cfg.get("face_center_weight", 0.72)), 0.0, 1.0
                    )
                    center_x = (
                        face_weight * box_center(face.box)[0]
                        + (1.0 - face_weight) * center_x
                    )
                    label = "primary_face"
                    evidence_kind = (
                        "small_face" if small and not paired_person_large else "frontal_face"
                    )
                    score = 0.62 * face_score + 0.38 * person_score
                    if evidence_kind == "frontal_face":
                        score += float(cfg.get("frontal_face_bonus", 0.10))
                    if small and not paired_person_large and not is_current:
                        score *= float(cfg.get("small_background_score_scale", 0.62))
            elif head is not None:
                head_weight = clamp(
                    float(cfg.get("head_center_weight", 0.60)), 0.0, 1.0
                )
                center_x = (
                    head_weight * box_center(head.box)[0]
                    + (1.0 - head_weight) * center_x
                )
                label = "primary_head"
                evidence_kind = "head"
                score = (
                    0.46 * self._movie_face_score(head, is_current=is_current)
                    + 0.54 * person_score
                )
                if has_frontal_face and not is_current:
                    score *= float(cfg.get("head_when_face_exists_scale", 0.82))
            else:
                person_area = box_area(person.box)
                person_width = person.box[2] - person.box[0]
                if has_frontal_face and not is_current:
                    score *= float(cfg.get("person_only_when_face_exists_scale", 0.68))
                if (
                    person_area
                    >= float(cfg.get("oversized_person_area", 0.18))
                    or person_width
                    >= float(cfg.get("oversized_person_width", 0.62))
                ):
                    score *= float(cfg.get("oversized_person_score_scale", 0.35))

            if is_current:
                score += float(cfg.get("current_character_bonus", 0.10))
            candidate = FocusCandidate(
                key=f"character:{anchor_id}",
                label=label,
                score=min(1.0, score),
                center_x=clamp(center_x, 0.0, 1.0),
                boxes=tuple(boxes),
                scene_state="movie_character",
                evidence_kind=evidence_kind,
                smoothing_regime="compose",
            )
            evidence.append(
                _CharacterEvidence(candidate, person, region, anchor_id)
            )

        orphan_min_area = float(cfg.get("orphan_face_min_area", 0.012))
        for face in faces:
            if face.track_id in used_regions:
                continue
            if box_area(face.box) < orphan_min_area or face.hits < int(
                cfg.get("orphan_face_min_hits", 4)
            ):
                continue
            candidate = FocusCandidate(
                key=f"character:{face.track_id}",
                label="primary_face",
                score=min(0.70, 0.72 * self._movie_face_score(face, is_current=False)),
                center_x=box_center(face.box)[0],
                boxes=(self._focus_box(face),),
                scene_state="movie_character",
                evidence_kind="unpaired_face",
                smoothing_regime="compose",
            )
            evidence.append(_CharacterEvidence(candidate, None, face, face.track_id))
        return evidence

    def _continuation_anchor(self, person: TrackView) -> Optional[str]:
        """Relink a fragmented person track to the current character.

        Detection cadence and brief occlusion can recreate the same on-screen
        character under a new numeric track ID.  Treat only a very close,
        strongly overlapping, similarly sized person box as the same anchor.
        The gate is intentionally too strict to bridge an actual shot change.
        """

        if self._current is None:
            return None
        current_anchor = self._anchor_from_key(self._current.key)
        if current_anchor is None or current_anchor == person.track_id:
            return None
        prior_person = next(
            (box.box for box in self._current.boxes if box.label == "person"),
            None,
        )
        if prior_person is None:
            return None

        cfg = self.selection
        overlap = iou(prior_person, person.box)
        prior_center = box_center(prior_person)
        new_center = box_center(person.box)
        center_gap = math.hypot(
            prior_center[0] - new_center[0], prior_center[1] - new_center[1]
        )
        prior_area = max(box_area(prior_person), 1e-8)
        area_ratio = box_area(person.box) / prior_area
        if overlap < float(cfg.get("same_character_relink_iou", 0.55)):
            return None
        if center_gap > float(
            cfg.get("same_character_relink_center_distance", 0.045)
        ):
            return None
        if not (
            float(cfg.get("same_character_relink_min_area_ratio", 0.65))
            <= area_ratio
            <= float(cfg.get("same_character_relink_max_area_ratio", 1.55))
        ):
            return None
        return current_anchor

    def _pair_regions(
        self,
        regions: Sequence[TrackView],
        persons: Sequence[TrackView],
    ) -> Dict[str, TrackView]:
        """Pair face/head regions to people with a one-to-one assignment.

        Pose and detection person boxes frequently overlap in dialogue and group
        shots.  Choosing the smallest overlapping person independently for each
        region can attach several faces to the same person and leave the actual
        main character as a person-only candidate.  Rank all geometrically
        plausible pairs, then greedily accept the best non-conflicting pairs.
        Region containment dominates tiny floating-point differences in person
        area, while normalized center distance breaks ties.
        """

        pair_rows: List[tuple[float, float, float, str, str, TrackView, TrackView]] = []
        for region in regions:
            cx, cy = box_center(region.box)
            region_area = max(box_area(region.box), 1e-8)
            for person in persons:
                px, _ = box_center(person.box)
                person_width = max(0.01, person.box[2] - person.box[0])
                person_height = max(0.01, person.box[3] - person.box[1])
                margin_x = 0.04 * person_width
                margin_y = 0.04 * person_height
                contains_center = (
                    person.box[0] - margin_x <= cx <= person.box[2] + margin_x
                    and person.box[1] - margin_y <= cy <= person.box[3] + margin_y
                )

                ix1 = max(region.box[0], person.box[0])
                iy1 = max(region.box[1], person.box[1])
                ix2 = min(region.box[2], person.box[2])
                iy2 = min(region.box[3], person.box[3])
                intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                containment = intersection / region_area
                if not contains_center and containment < 0.20:
                    continue

                x_distance = abs(cx - px) / person_width
                expected_head_y = person.box[1] + 0.18 * person_height
                y_distance = abs(cy - expected_head_y) / person_height
                outside_penalty = 0.0 if contains_center else 0.35
                cost = (
                    2.50 * (1.0 - clamp(containment, 0.0, 1.0))
                    + 0.90 * x_distance
                    + 0.35 * y_distance
                    + outside_penalty
                    + 0.04 * math.sqrt(max(box_area(person.box), 0.0))
                    - 0.03 * clamp(region.score, 0.0, 1.0)
                )
                pair_rows.append(
                    (
                        cost,
                        -containment,
                        x_distance,
                        region.track_id,
                        person.track_id,
                        region,
                        person,
                    )
                )

        paired: Dict[str, TrackView] = {}
        used_regions: set[str] = set()
        used_persons: set[str] = set()
        for _, _, _, _, _, region, person in sorted(pair_rows):
            if region.track_id in used_regions or person.track_id in used_persons:
                continue
            paired[person.track_id] = region
            used_regions.add(region.track_id)
            used_persons.add(person.track_id)
        return paired

    def _movie_person_score(self, person: TrackView, *, is_current: bool) -> float:
        area = box_area(person.box)
        center_x, center_y = box_center(person.box)
        area_score = min(1.0, math.sqrt(max(area, 0.0) / 0.11))
        center_prior = 1.0 - min(1.0, abs(center_x - 0.5) * 1.5)
        vertical_prior = 1.0 - min(1.0, abs(center_y - 0.56) * 1.6)
        stable = min(1.0, person.hits / 4.0)
        score = (
            0.30 * person.score
            + 0.34 * area_score
            + 0.16 * center_prior
            + 0.12 * stable
            + 0.08 * vertical_prior
        )
        if is_current:
            score += 0.06
        return min(1.0, score)

    def _movie_face_score(self, face: TrackView, *, is_current: bool) -> float:
        area = box_area(face.box)
        center_x, _ = box_center(face.box)
        area_score = min(1.0, math.sqrt(max(area, 0.0) / 0.028))
        center_prior = 1.0 - min(1.0, abs(center_x - 0.5) * 1.5)
        stable = min(1.0, face.hits / 4.0)
        score = (
            0.38 * face.score
            + 0.34 * area_score
            + 0.16 * stable
            + 0.12 * center_prior
        )
        if face.label == "head":
            score *= 0.78
        if is_current:
            score += 0.05
        return min(1.0, score)

    def _interaction_group(
        self,
        characters: Sequence[_CharacterEvidence],
        crop_width: float,
    ) -> Optional[FocusCandidate]:
        if len(characters) < 2:
            return None
        ordered = sorted(
            characters, key=lambda item: item.candidate.score, reverse=True
        )
        top_score = ordered[0].candidate.score
        comparable = [
            item
            for item in ordered[:3]
            if item.candidate.score
            >= top_score * float(self.selection.get("group_score_ratio", 0.74))
        ]
        if len(comparable) < 2:
            return None
        combined = union_box(
            box.box
            for item in comparable
            for box in item.candidate.boxes[:1]
        )
        max_width = max(
            crop_width * float(self.selection.get("group_crop_width_factor", 1.08)),
            0.18,
        )
        if combined[2] - combined[0] > max_width:
            return None

        current_anchor = self._anchor_from_key(self._current.key) if self._current else None
        anchors = {item.anchor_id for item in comparable}
        anchor = current_anchor if current_anchor in anchors else comparable[0].anchor_id
        boxes: List[FocusBox] = []
        seen = set()
        for item in comparable:
            for box in item.candidate.boxes:
                if box.focus_id not in seen:
                    seen.add(box.focus_id)
                    boxes.append(box)
        center_x = sum(
            item.candidate.center_x * max(0.05, item.candidate.score)
            for item in comparable
        ) / sum(max(0.05, item.candidate.score) for item in comparable)
        score = (
            sum(item.candidate.score for item in comparable) / len(comparable)
            + float(self.selection.get("group_score_bonus", 0.08))
        )
        return FocusCandidate(
            key=f"interaction_group:{anchor}",
            label="interaction_group",
            score=min(1.0, score),
            center_x=clamp(center_x, 0.0, 1.0),
            boxes=tuple(boxes),
            scene_state="movie_interaction",
            evidence_kind="interaction_group",
            smoothing_regime="compose",
        )


class FocusPolicy:
    """Backward-compatible facade with distinct sports and movie policies."""

    def __init__(self, mode: str, config: Dict) -> None:
        if mode == "sports":
            self._implementation: _PolicyBase = SportsFocusPolicy(config)
        elif mode == "movie":
            self._implementation = MovieFocusPolicy(config)
        else:
            raise ValueError(f"Unsupported focus policy: {mode!r}")

    @property
    def implementation(self) -> _PolicyBase:
        return self._implementation

    def reset(self) -> None:
        self._implementation.reset()

    def select(
        self,
        tracks: Sequence[TrackView],
        motion: Optional[Detection],
        *,
        crop_width: float,
        time_s: float,
    ) -> SelectedFocus:
        return self._implementation.select(
            tracks, motion, crop_width=crop_width, time_s=time_s
        )
