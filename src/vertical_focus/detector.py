from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
from loguru import logger
import numpy as np

from .geometry import sanitize_box
from .types import Detection


class MediaPipeFocusDetector:
    """EfficientDet person/ball detection plus MediaPipe Tasks face detection.

    The detector is loaded lazily. If the MediaPipe face task cannot be created,
    OpenCV's bundled frontal-face cascade is used as a non-fatal fallback.
    """

    def __init__(
        self,
        *,
        model_path: str,
        face_model_path: str,
        delegate: str,
        config: Dict,
    ) -> None:
        self.model_path = model_path
        self.face_model_path = face_model_path
        self.delegate = delegate
        self.config = config
        self._object_detector = None
        self._face_detector = None
        self._face_cascade: Optional[cv2.CascadeClassifier] = None
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
        detection_cfg = self.config.get("detection", {})
        object_base_options = mp_python.BaseOptions(
            model_asset_path=self.model_path,
            delegate=delegate,
        )
        self._object_detector = vision.ObjectDetector.create_from_options(
            vision.ObjectDetectorOptions(
                base_options=object_base_options,
                max_results=int(detection_cfg.get("max_results", 50)),
                score_threshold=min(
                    float(detection_cfg.get("person_min_score", 0.35)),
                    float(detection_cfg.get("ball_min_score", 0.25)),
                ),
                category_allowlist=["person", "sports ball"],
                running_mode=vision.RunningMode.VIDEO,
            )
        )
        self._mp = mp

        face_path = Path(self.face_model_path)
        if face_path.is_file():
            try:
                face_base_options = mp_python.BaseOptions(
                    model_asset_path=str(face_path),
                    delegate=delegate,
                )
                self._face_detector = vision.FaceDetector.create_from_options(
                    vision.FaceDetectorOptions(
                        base_options=face_base_options,
                        running_mode=vision.RunningMode.VIDEO,
                        min_detection_confidence=float(detection_cfg.get("face_min_score", 0.45)),
                        min_suppression_threshold=float(
                            detection_cfg.get("face_min_suppression", 0.30)
                        ),
                    )
                )
                logger.info("face detector loaded model={}", face_path)
            except Exception as exc:  # pragma: no cover - native library/model dependent
                logger.warning(
                    "MediaPipe Tasks face detector failed; using OpenCV cascade: {}",
                    exc,
                )
                self._face_detector = None
        else:
            logger.warning(
                "Face model missing at {}; using OpenCV cascade",
                face_path,
            )

        if self._face_detector is None:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml"
            cascade = cv2.CascadeClassifier(str(cascade_path))
            if not cascade.empty():
                self._face_cascade = cascade
                logger.info("face detector fallback loaded model={}", cascade_path)

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> List[Detection]:
        self._ensure_loaded()
        assert self._mp is not None
        assert self._object_detector is not None

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
                    source="efficientdet_lite0",
                )
            )

        if self._face_detector is not None:
            try:
                face_result = self._face_detector.detect_for_video(image, timestamp_ms)
                for face in face_result.detections:
                    score = (
                        float(face.categories[0].score)
                        if getattr(face, "categories", None)
                        else 1.0
                    )
                    if score < float(detection_cfg.get("face_min_score", 0.45)):
                        continue
                    box = face.bounding_box
                    detections.append(
                        Detection(
                            label="face",
                            score=score,
                            box=sanitize_box(
                                (
                                    box.origin_x / width,
                                    box.origin_y / height,
                                    (box.origin_x + box.width) / width,
                                    (box.origin_y + box.height) / height,
                                )
                            ),
                            source="blazeface_short_range",
                        )
                    )
            except Exception as exc:  # pragma: no cover - native model edge cases
                if not self._face_warning_emitted:
                    logger.warning(
                        "Face task inference failed; continuing with cascade/person evidence: {}",
                        exc,
                    )
                    self._face_warning_emitted = True
        elif self._face_cascade is not None:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            minimum_fraction = float(detection_cfg.get("face_min_size_fraction", 0.03))
            minimum_pixels = max(20, int(round(min(width, height) * minimum_fraction)))
            boxes = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.10,
                minNeighbors=5,
                minSize=(minimum_pixels, minimum_pixels),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for x, y, w, h in boxes:
                detections.append(
                    Detection(
                        label="face",
                        score=max(0.50, float(detection_cfg.get("face_min_score", 0.45))),
                        box=sanitize_box((x / width, y / height, (x + w) / width, (y + h) / height)),
                        source="opencv_haar_fallback",
                    )
                )

        return detections

    def close(self) -> None:
        if self._object_detector is not None:
            self._object_detector.close()
        if self._face_detector is not None:
            self._face_detector.close()
