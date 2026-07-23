from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from loguru import logger

from .geometry import crop_width_normalized
from .motion import MotionEstimator
from .policy import FocusPolicy
from .shots import LocalShotDetector, ShotRange, TagStoreShots
from .smoothing import SmoothCamera
from .tracker import TrackManager
from .types import Detection, FileProcessResult, FrameDecision, ShotOutput, TrackView
from .video import VideoReader


class VerticalFocusEngine:
    """Frame-level YOLO26 focus processor.

    A detector may be injected for deterministic unit tests. Production
    execution accepts only the YOLO26 backend; there is no alternate inference
    path or runtime fallback.
    """

    def __init__(
        self,
        *,
        mode: str,
        policy_config: Dict,
        object_model: str = "",
        face_model: str = "",
        delegate: str = "cpu",
        detector_backend: str = "yolo26",
        yolo_detect_model: str = "",
        yolo_pose_model: str = "",
        yolo_device: str = "0",
        yolo_imgsz: Optional[int] = None,
        yolo_half: bool = True,
        yolo_end2end: bool = False,
        shots: TagStoreShots,
        progress_log_interval_seconds: float,
        debug_jsonl_path: str = "",
        detector: Optional[Any] = None,
    ) -> None:
        del object_model, face_model, delegate
        backend = str(detector_backend).strip().lower()
        if backend != "yolo26":
            raise ValueError("VerticalFocusEngine supports only detector_backend='yolo26'")

        self.mode = mode
        self.config = policy_config
        self.shots = shots
        self.progress_log_interval_s = max(0.25, float(progress_log_interval_seconds))
        if detector is not None:
            self.detector = detector
        else:
            from .yolo26_detector import Yolo26FocusDetector

            self.detector = Yolo26FocusDetector(
                mode=mode,
                detect_model_path=yolo_detect_model,
                pose_model_path=yolo_pose_model,
                device=yolo_device,
                config=policy_config,
                imgsz=yolo_imgsz,
                half=yolo_half,
                end2end=yolo_end2end,
            )

        tracking = policy_config.get("tracking", {})
        self.tracker = TrackManager(
            max_missed_updates=int(tracking.get("max_missed_updates", 4)),
            max_missed_updates_by_label={
                "ball": int(
                    tracking.get(
                        "ball_max_missed_updates",
                        tracking.get("max_missed_updates", 4),
                    )
                ),
                "person": int(
                    tracking.get(
                        "person_max_missed_updates",
                        tracking.get("max_missed_updates", 4),
                    )
                ),
                "face": int(
                    tracking.get(
                        "face_max_missed_updates",
                        tracking.get("max_missed_updates", 4),
                    )
                ),
                "head": int(
                    tracking.get(
                        "head_max_missed_updates",
                        tracking.get("max_missed_updates", 4),
                    )
                ),
            },
            match_center_distance=float(tracking.get("match_center_distance", 0.18)),
            ball_match_center_distance=float(
                tracking.get("ball_match_center_distance", 0.18)
            ),
            ball_reacquire_center_distance=float(
                tracking.get("ball_reacquire_center_distance", 0.28)
            ),
            ball_reacquire_min_confidence=float(
                tracking.get("ball_reacquire_min_confidence", 0.55)
            ),
            ball_reacquire_min_hits=int(tracking.get("ball_reacquire_min_hits", 2)),
            min_iou=float(tracking.get("min_iou", 0.03)),
            box_alpha=float(tracking.get("box_alpha", 0.68)),
            ball_box_alpha=float(tracking.get("ball_box_alpha", 0.90)),
            box_alpha_by_label={
                "person": float(
                    tracking.get("person_box_alpha", tracking.get("box_alpha", 0.68))
                ),
                "face": float(
                    tracking.get("face_box_alpha", tracking.get("box_alpha", 0.68))
                ),
                "head": float(
                    tracking.get("head_box_alpha", tracking.get("box_alpha", 0.68))
                ),
                "hand": float(
                    tracking.get("hand_box_alpha", tracking.get("box_alpha", 0.68))
                ),
                "ball": float(tracking.get("ball_box_alpha", 0.90)),
            },
            velocity_alpha_by_label={
                "person": float(tracking.get("person_velocity_alpha", 0.68)),
                "face": float(tracking.get("face_velocity_alpha", 0.72)),
                "head": float(tracking.get("head_velocity_alpha", 0.68)),
                "hand": float(tracking.get("hand_velocity_alpha", 0.78)),
                "ball": float(tracking.get("ball_velocity_alpha", 0.82)),
            },
            prediction_max_frames=int(tracking.get("prediction_max_frames", 10)),
            ball_prediction_max_frames=int(
                tracking.get("ball_prediction_max_frames", 22)
            ),
            prediction_max_distance=float(
                tracking.get("prediction_max_distance", 0.10)
            ),
        )
        self.policy = FocusPolicy(mode, policy_config)
        self.motion = MotionEstimator()
        self.smoother: Optional[SmoothCamera] = None
        self.active_shot_id: Optional[str] = None
        self.last_global_time_ms: Optional[int] = None
        self.next_detection_global_ms = -1
        self.file_sequence = 0
        self.global_frame_idx = 0
        self._debug_handle: Optional[TextIO] = None
        if debug_jsonl_path:
            debug_path = Path(debug_jsonl_path)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            self._debug_handle = debug_path.open("a", encoding="utf-8")

    def _reset_for_shot(self, shot_id: str, crop_width: float) -> None:
        self.tracker.reset()
        self.policy.reset()
        self.motion.reset()
        temporal = self.config.get("temporal", {})
        self.smoother = SmoothCamera(
            crop_width=crop_width,
            response_time_s=float(temporal.get("response_time_seconds", 0.35)),
            deadband=float(temporal.get("deadband", 0.006)),
            max_speed_per_s=float(
                temporal.get("max_speed_normalized_per_second", 0.25)
            ),
            max_accel_per_s2=float(
                temporal.get("max_acceleration_normalized_per_second2", 0.75)
            ),
            fast_response_time_s=float(
                temporal.get("fast_response_time_seconds", 0.16)
            ),
            fast_max_speed_per_s=float(
                temporal.get("fast_max_speed_normalized_per_second", 0.55)
            ),
            fast_max_accel_per_s2=float(
                temporal.get("fast_max_acceleration_normalized_per_second2", 2.0)
            ),
            fast_error_threshold=float(
                temporal.get("fast_error_threshold", 0.07)
            ),
            fast_boost_s=float(temporal.get("fast_boost_seconds", 0.30)),
            target_filter_s=float(temporal.get("target_filter_seconds", 0.14)),
            fast_on_large_error=bool(temporal.get("fast_on_large_error", False)),
            profiles=temporal.get("camera_profiles", {}),
        )
        self.active_shot_id = shot_id
        self.last_global_time_ms = None
        # Detect immediately after a hard cut rather than inheriting the prior
        # shot's cadence deadline.
        self.next_detection_global_ms = -1

    @staticmethod
    def _serialize_detection(detection: Detection) -> Dict[str, Any]:
        return {
            "label": detection.label,
            "score": round(float(detection.score), 6),
            "box": [round(float(value), 6) for value in detection.box],
            "source": detection.source,
        }

    @staticmethod
    def _serialize_track(track: TrackView) -> Dict[str, Any]:
        return {
            "id": track.track_id,
            "label": track.label,
            "score": round(float(track.score), 6),
            "box": [round(float(value), 6) for value in track.box],
            "hits": track.hits,
            "misses": track.misses,
            "observed": track.observed,
            "last_detection_frame": track.last_detection_frame,
            "vx_per_frame": round(float(track.vx_per_frame), 8),
            "vy_per_frame": round(float(track.vy_per_frame), 8),
        }

    def _write_debug_frame(
        self,
        *,
        source_media: str,
        frame_idx: int,
        tracking_frame_idx: int,
        time_ms: int,
        global_time_ms: int,
        shot_id: str,
        crop_width: float,
        detection_update: bool,
        detections: List[Detection],
        tracks: List[TrackView],
        selected: Any,
        output_x: float,
    ) -> None:
        if self._debug_handle is None:
            return
        row = {
            "source": source_media,
            "mode": self.mode,
            "frame": frame_idx,
            "tracking_frame": tracking_frame_idx,
            "time_ms": time_ms,
            "global_time_ms": global_time_ms,
            "shot_id": shot_id,
            "crop_width": round(float(crop_width), 8),
            "detection_update": detection_update,
            "detections": [self._serialize_detection(item) for item in detections],
            "tracks": [self._serialize_track(item) for item in tracks],
            "selected": {
                "key": selected.key,
                "label": selected.label,
                "score": round(float(selected.score), 6),
                "center_x": round(float(selected.center_x), 6),
                "scene_state": selected.scene_state,
                "evidence_kind": selected.evidence_kind,
                "smoothing_regime": selected.smoothing_regime,
                "focus_ids": [box.focus_id for box in selected.boxes],
                "boxes": [
                    {
                        "id": box.focus_id,
                        "label": box.label,
                        "score": round(float(box.score), 6),
                        "box": [round(float(value), 6) for value in box.box],
                    }
                    for box in selected.boxes
                ],
            },
            "output_x": round(float(output_x), 6),
        }
        self._debug_handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    def process_file(self, source_media: str, content_offset_ms: int) -> FileProcessResult:
        # A failed input must not leak tracker, policy, cadence, or smoother state
        # into the next common-ml stdin item. Reserve a unique local-shot
        # namespace before opening the file so recovery is deterministic.
        file_sequence = self.file_sequence
        self.file_sequence += 1
        self.active_shot_id = None
        self.last_global_time_ms = None
        self.next_detection_global_ms = -1

        reader = VideoReader(source_media)
        info = reader.info
        crop_width = crop_width_normalized(info.width, info.height)
        detection_fps = float(
            self.config.get("detection", {}).get("detection_fps", 8.0)
        )
        detection_period_ms = max(1, int(round(1000.0 / detection_fps)))
        local_cfg = self.config.get("local_shots", {})
        local_shots = LocalShotDetector(
            threshold=float(local_cfg.get("histogram_threshold", 0.5)),
            gray_mad_threshold=float(local_cfg.get("gray_mad_threshold", 0.18)),
            edge_change_threshold=float(
                local_cfg.get("edge_change_threshold", 0.20)
            ),
            alignment_response_threshold=float(
                local_cfg.get("alignment_response_threshold", 0.45)
            ),
            moderate_histogram_threshold=(
                float(local_cfg["moderate_histogram_threshold"])
                if "moderate_histogram_threshold" in local_cfg
                else None
            ),
            moderate_gray_mad_threshold=float(
                local_cfg.get("moderate_gray_mad_threshold", 0.14)
            ),
            moderate_edge_change_threshold=float(
                local_cfg.get("moderate_edge_change_threshold", 0.50)
            ),
            moderate_alignment_response_threshold=float(
                local_cfg.get(
                    "moderate_alignment_response_threshold", 0.40
                )
            ),
            minimum_seconds=float(local_cfg.get("minimum_shot_seconds", 0.25)),
            fps=info.fps,
        )

        outputs: List[ShotOutput] = []
        current_decisions: List[FrameDecision] = []
        current_shot_id: Optional[str] = None
        current_shot_range: Optional[ShotRange] = None
        current_start_hint_ms = 0
        last_log_time_ms = -1

        def finalize(end_hint_ms: int) -> None:
            nonlocal current_decisions, current_shot_id, current_start_hint_ms
            if not current_decisions or current_shot_id is None:
                current_decisions = []
                return
            start_ms = max(0, int(current_start_hint_ms))
            end_ms = max(start_ms + 1, int(end_hint_ms))
            outputs.append(
                ShotOutput(
                    shot_id=current_shot_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    start_frame_idx=current_decisions[0].frame_idx,
                    crop_width=crop_width,
                    decisions=current_decisions,
                )
            )
            current_decisions = []

        try:
            for frame in reader:
                global_time_ms = int(content_offset_ms + frame.time_ms)
                tagstore_shot = (
                    self.shots.at(global_time_ms) if self.shots.available else None
                )
                local_cut = False
                if tagstore_shot is not None:
                    shot_id = f"tagstore:{tagstore_shot.shot_id}"
                    start_hint_ms = max(
                        0, tagstore_shot.start_ms - content_offset_ms
                    )
                else:
                    local_cut = local_shots.update(frame.rgb, frame.frame_idx)
                    shot_id = f"local:{file_sequence}:{local_shots.shot_index}"
                    start_hint_ms = (
                        frame.time_ms
                        if local_cut or current_shot_id is None
                        else current_start_hint_ms
                    )

                if current_shot_id is None:
                    current_shot_id = shot_id
                    current_shot_range = tagstore_shot
                    current_start_hint_ms = start_hint_ms
                    if self.active_shot_id != shot_id or self.smoother is None:
                        self._reset_for_shot(shot_id, crop_width)
                elif shot_id != current_shot_id:
                    previous_end = frame.time_ms
                    if current_shot_range is not None:
                        previous_end = min(
                            max(previous_end, current_start_hint_ms + 1),
                            max(
                                current_start_hint_ms + 1,
                                current_shot_range.end_ms - content_offset_ms,
                            ),
                        )
                    finalize(previous_end)
                    current_shot_id = shot_id
                    current_shot_range = tagstore_shot
                    current_start_hint_ms = start_hint_ms
                    self._reset_for_shot(shot_id, crop_width)

                tracking_frame_idx = self.global_frame_idx
                self.global_frame_idx += 1
                detection_update = global_time_ms >= self.next_detection_global_ms
                detections: List[Detection] = []
                if detection_update:
                    detections = list(self.detector.detect(frame.rgb, global_time_ms))
                    tracks = self.tracker.update(detections, tracking_frame_idx)
                    self.next_detection_global_ms = global_time_ms + detection_period_ms
                else:
                    tracks = self.tracker.snapshot(tracking_frame_idx)

                motion = self.motion.update(frame.rgb)
                selected = self.policy.select(
                    tracks,
                    motion,
                    crop_width=crop_width,
                    time_s=global_time_ms / 1000.0,
                )
                assert self.smoother is not None
                effective_regime = selected.smoothing_regime
                if not current_decisions and not self.smoother.initialized:
                    x_center = self.smoother.reset(selected.center_x)
                else:
                    if self.last_global_time_ms is None:
                        dt_s = 1.0 / info.fps
                    else:
                        dt_s = max(
                            1.0 / 240.0,
                            (global_time_ms - self.last_global_time_ms) / 1000.0,
                        )
                    if selected.smoothing_regime == "cut_reset":
                        x_center = self.smoother.reset(selected.center_x)
                    elif selected.smoothing_regime == "locked":
                        x_center = self.smoother.hold()
                    elif selected.smoothing_regime == "settle_then_lock":
                        reframe_profile = "confirmed_reframe"
                        if (
                            abs(selected.center_x - self.smoother.x)
                            <= self.smoother.profile_deadband(reframe_profile)
                        ):
                            x_center = self.smoother.hold()
                            effective_regime = "locked"
                        else:
                            # The policy has already confirmed a real subject
                            # reframe and frozen the target. Continue the movie
                            # reframe profile until the camera reaches it, then
                            # convert to an exact hold.
                            previous_x = self.smoother.x
                            x_center = self.smoother.update(
                                selected.center_x,
                                dt_s,
                                profile_name=reframe_profile,
                            )
                            if (
                                not self.smoother.reframing
                                and (
                                    abs(selected.center_x - self.smoother.x)
                                    <= self.smoother.profile_deadband(
                                        reframe_profile
                                    )
                                    + 1e-9
                                    or abs(self.smoother.x - previous_x) <= 1e-12
                                )
                            ):
                                x_center = self.smoother.hold()
                                effective_regime = "locked"
                    else:
                        urgent = selected.smoothing_regime in {
                            "fast_reacquire",
                            "confirmed_reframe",
                        }
                        x_center = self.smoother.update(
                            selected.center_x,
                            dt_s,
                            urgent=urgent,
                            profile_name=selected.smoothing_regime,
                        )
                selected = replace(selected, smoothing_regime=effective_regime)
                self.last_global_time_ms = global_time_ms
                current_decisions.append(
                    FrameDecision(
                        frame_idx=frame.frame_idx,
                        time_ms=frame.time_ms,
                        x_center=x_center,
                        label=selected.label,
                        confidence=selected.score,
                        boxes=selected.boxes,
                        selected_key=selected.key,
                        scene_state=selected.scene_state,
                        evidence_kind=selected.evidence_kind,
                        smoothing_regime=selected.smoothing_regime,
                    )
                )
                assert current_shot_id is not None
                self._write_debug_frame(
                    source_media=source_media,
                    frame_idx=frame.frame_idx,
                    tracking_frame_idx=tracking_frame_idx,
                    time_ms=frame.time_ms,
                    global_time_ms=global_time_ms,
                    shot_id=current_shot_id,
                    crop_width=crop_width,
                    detection_update=detection_update,
                    detections=detections,
                    tracks=tracks,
                    selected=selected,
                    output_x=x_center,
                )

                if frame.time_ms - last_log_time_ms >= int(
                    self.progress_log_interval_s * 1000.0
                ):
                    last_log_time_ms = frame.time_ms
                    logger.info(
                        "focus progress source={} frame={} time_ms={} shot={} label={} state={} evidence={} regime={} x={:.4f}",
                        source_media,
                        frame.frame_idx,
                        frame.time_ms,
                        current_shot_id,
                        selected.label,
                        selected.scene_state,
                        selected.evidence_kind,
                        selected.smoothing_regime,
                        x_center,
                    )
        finally:
            effective_duration_ms = reader.effective_duration_ms
            reader.close()
            if self._debug_handle is not None:
                self._debug_handle.flush()

        if current_decisions:
            final_end = effective_duration_ms
            if current_shot_range is not None:
                final_end = min(
                    final_end,
                    max(
                        current_start_hint_ms + 1,
                        current_shot_range.end_ms - content_offset_ms,
                    ),
                )
            finalize(final_end)

        logger.info(
            "focus complete source={} duration_ms={} frames={} shot_segments={} labels={}",
            source_media,
            effective_duration_ms,
            sum(len(shot.decisions) for shot in outputs),
            len(outputs),
            dict(
                Counter(
                    decision.label
                    for shot in outputs
                    for decision in shot.decisions
                )
            ),
        )
        return FileProcessResult(source_media, effective_duration_ms, outputs)

    def close(self) -> None:
        self.detector.close()
        if self._debug_handle is not None:
            self._debug_handle.flush()
            self._debug_handle.close()
            self._debug_handle = None
