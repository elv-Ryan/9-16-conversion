from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger
import numpy as np

from .geometry import sanitize_box
from .types import Detection


class MediaPipeFocusDetector:
    """Person/ball object detection plus face detection.

    The detector is lazy so policy and contract unit tests do not need to load
    TensorFlow Lite or MediaPipe native libraries.
    """

    def __init__(self, *, model_path: str, delegate: str, config: Dict) -> None:
        self.model_path = model_path
        self.delegate = delegate
        self.config = config
        self._object_detector = None
        self._face_detector = None
        self._mp = None
        self._last_timestamp_ms = -1
        self._face_warning_emitted = False

    def _ensure_loaded(self) -> None:
        if self._object_detector is not None:
            return
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        delegate = (
            mp_python.BaseOptions.Delegate.GPU
            if self.delegate == "gpu"
            else mp_python.BaseOptions.Delegate.CPU
        )
        base_options = mp_python.BaseOptions(model_asset_path=self.model_path, delegate=delegate)
        detection_cfg = self.config.get("detection", {})
        self._object_detector = vision.ObjectDetector.create_from_options(
            vision.ObjectDetectorOptions(
                base_options=base_options,
                max_results=int(detection_cfg.get("max_results", 50)),
                score_threshold=min(
                    float(detection_cfg.get("person_min_score", 0.35)),
                    float(detection_cfg.get("ball_min_score", 0.25)),
                ),
                running_mode=vision.RunningMode.VIDEO,
            )
        )
        self._mp = mp

        # The Solutions API bundles its own full-range face model. Some future
        # MediaPipe builds may remove this API, in which case movie policy still
        # degrades to tracked people/action rather than failing the container.
        try:
            self._face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=float(detection_cfg.get("face_min_score", 0.45)),
            )
        except Exception as exc:  # pragma: no cover - depends on mediapipe build
            logger.warning("Face detector unavailable; using person/action fallback: {}", exc)
            self._face_detector = None

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> List[Detection]:
        self._ensure_loaded()
        assert self._mp is not None
        height, width = rgb.shape[:2]
        detections: List[Detection] = []
        detection_cfg = self.config.get("detection", {})

        # VIDEO mode requires strictly increasing timestamps across all files.
        timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._object_detector.detect_for_video(image, timestamp_ms)
        for item in result.detections:
            if not item.categories:
                continue
            category = item.categories[0]
            raw_name = (
                getattr(category, "category_name", None)
                or getattr(category, "display_name", None)
                or ""
            ).strip().lower()
            if raw_name == "person":
                label = "person"
                threshold = float(detection_cfg.get("person_min_score", 0.35))
            elif raw_name in {"sports ball", "ball"}:
                label = "ball"
                threshold = float(detection_cfg.get("ball_min_score", 0.25))
            else:
                continue
            score = float(category.score)
            if score < threshold:
                continue
            box = item.bounding_box
            detections.append(
                Detection(
                    label=label,
                    score=score,
                    box=sanitize_box(
                        (
                            box.origin_x / width,
                            box.origin_y / height,
                            (box.origin_x + box.width) / width,
                            (box.origin_y + box.height) / height,
                        )
                    ),
                    source="efficientdet",
                )
            )

        if self._face_detector is not None:
            try:
                face_result = self._face_detector.process(rgb)
                for face in face_result.detections or []:
                    score = float(face.score[0]) if face.score else 0.0
                    if score < float(detection_cfg.get("face_min_score", 0.45)):
                        continue
                    relative = face.location_data.relative_bounding_box
                    detections.append(
                        Detection(
                            label="face",
                            score=score,
                            box=sanitize_box(
                                (
                                    relative.xmin,
                                    relative.ymin,
                                    relative.xmin + relative.width,
                                    relative.ymin + relative.height,
                                )
                            ),
                            source="mediapipe_face",
                        )
                    )
            except Exception as exc:  # pragma: no cover - native model edge cases
                if not self._face_warning_emitted:
                    logger.warning("Face inference failed; continuing with other evidence: {}", exc)
                    self._face_warning_emitted = True
        return detections

    def close(self) -> None:
        if self._object_detector is not None:
            self._object_detector.close()
        if self._face_detector is not None:
            self._face_detector.close()
