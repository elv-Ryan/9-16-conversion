from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from loguru import logger

from .detector import MediaPipeFocusDetector
from .geometry import crop_width_normalized
from .motion import MotionEstimator
from .policy import FocusPolicy
from .shots import TagStoreShots
from .smoothing import SmoothCamera
from .tracker import TrackManager
from .types import FileProcessResult, FrameDecision, ShotOutput
from .video import VideoReader


class VerticalFocusEngine:
    """Frame-level focus processor retaining state across contiguous fragments."""

    def __init__(
        self,
        *,
        mode: str,
        policy_config: Dict,
        object_model: str,
        delegate: str,
        shots: TagStoreShots,
        progress_log_interval_seconds: float,
        detector: Optional[MediaPipeFocusDetector] = None,
    ) -> None:
        self.mode = mode
        self.config = policy_config
        self.shots = shots
        self.progress_log_interval_s = max(0.25, float(progress_log_interval_seconds))
        self.detector = detector or MediaPipeFocusDetector(
            model_path=object_model,
            delegate=delegate,
            config=policy_config,
        )
        tracking = policy_config.get("tracking", {})
        self.tracker = TrackManager(
            max_missed_updates=int(tracking.get("max_missed_updates", 4)),
            match_center_distance=float(tracking.get("match_center_distance", 0.18)),
            min_iou=float(tracking.get("min_iou", 0.03)),
            box_alpha=float(tracking.get("box_alpha", 0.68)),
        )
        self.policy = FocusPolicy(mode, policy_config)
        self.motion = MotionEstimator()
        self.smoother: Optional[SmoothCamera] = None
        self.active_shot_id: Optional[str] = None
        self.last_global_time_ms: Optional[int] = None
        self.next_detection_global_ms = -1
        self.file_sequence = 0
        self.global_frame_idx = 0

    def _reset_for_shot(self, shot_id: str, crop_width: float) -> None:
        self.tracker.reset()
        self.policy.reset()
        self.motion.reset()
        temporal = self.config.get("temporal", {})
        self.smoother = SmoothCamera(
            crop_width=crop_width,
            response_time_s=float(temporal.get("response_time_seconds", 0.35)),
            deadband=float(temporal.get("deadband", 0.006)),
            max_speed_per_s=float(temporal.get("max_speed_normalized_per_second", 0.25)),
            max_accel_per_s2=float(temporal.get("max_acceleration_normalized_per_second2", 0.75)),
        )
        self.active_shot_id = shot_id
        self.last_global_time_ms = None

    def process_file(self, source_media: str, content_offset_ms: int) -> FileProcessResult:
        reader = VideoReader(source_media)
        info = reader.info
        crop_width = crop_width_normalized(info.width, info.height)
        detection_fps = float(self.config.get("detection", {}).get("detection_fps", 8.0))
        detection_period_ms = max(1, int(round(1000.0 / detection_fps)))

        outputs: List[ShotOutput] = []
        current_decisions: List[FrameDecision] = []
        current_shot_id: Optional[str] = None
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
                # The vertical_video track is a single, uncut focus trajectory
                # covering the whole input file. We deliberately do not consult
                # the tagstore, run local shot detection, or infer anything from
                # the filename: every file yields exactly one shot that starts at
                # 0 and ends at the full media duration.
                shot_id = f"file:{self.file_sequence}"
                if current_shot_id is None:
                    current_shot_id = shot_id
                    current_start_hint_ms = 0
                    if self.active_shot_id != shot_id or self.smoother is None:
                        self._reset_for_shot(shot_id, crop_width)

                tracking_frame_idx = self.global_frame_idx
                self.global_frame_idx += 1
                if global_time_ms >= self.next_detection_global_ms:
                    detections = self.detector.detect(frame.rgb, global_time_ms)
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
                if not current_decisions and self.active_shot_id == current_shot_id and not self.smoother.initialized:
                    x_center = self.smoother.reset(selected.center_x)
                else:
                    if self.last_global_time_ms is None:
                        dt_s = 1.0 / info.fps
                    else:
                        dt_s = max(1.0 / 240.0, (global_time_ms - self.last_global_time_ms) / 1000.0)
                    x_center = self.smoother.update(selected.center_x, dt_s)
                self.last_global_time_ms = global_time_ms
                current_decisions.append(
                    FrameDecision(
                        frame_idx=frame.frame_idx,
                        time_ms=frame.time_ms,
                        x_center=x_center,
                        label=selected.label,
                        confidence=selected.score,
                        boxes=selected.boxes,
                    )
                )

                if frame.time_ms - last_log_time_ms >= int(self.progress_log_interval_s * 1000.0):
                    last_log_time_ms = frame.time_ms
                    if info.duration_ms > 0:
                        percent = min(100.0, 100.0 * frame.time_ms / info.duration_ms)
                        logger.info(
                            "focus progress source={} frame={} time_ms={} percent={:.1f} shot={}",
                            source_media,
                            frame.frame_idx,
                            frame.time_ms,
                            percent,
                            current_shot_id,
                        )
                    else:
                        logger.info(
                            "focus progress source={} frame={} time_ms={} shot={}",
                            source_media,
                            frame.frame_idx,
                            frame.time_ms,
                            current_shot_id,
                        )
        finally:
            effective_duration_ms = reader.effective_duration_ms
            reader.close()

        if current_decisions:
            # A single shot spanning the whole file: end at the full duration.
            finalize(effective_duration_ms)

        self.file_sequence += 1
        logger.info(
            "focus complete source={} duration_ms={} frames={} shot_segments={} labels={}",
            source_media,
            effective_duration_ms,
            sum(len(shot.decisions) for shot in outputs),
            len(outputs),
            dict(Counter(decision.label for shot in outputs for decision in shot.decisions)),
        )
        return FileProcessResult(source_media, effective_duration_ms, outputs)

    def close(self) -> None:
        self.detector.close()
