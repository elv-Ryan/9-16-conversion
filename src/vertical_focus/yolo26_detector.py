from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger
import numpy as np

from .geometry import sanitize_box
from .types import BBox, Detection


class Yolo26FocusDetector:
    """YOLO26-only person, sports-ball, and head/face evidence.

    The COCO detection model supplies person and sports-ball boxes. The YOLO26
    pose model supplies facial keypoints; those keypoints are converted into a
    conservative head box so movie focus does not depend on MediaPipe.
    """

    PERSON_CLASS = 0
    SPORTS_BALL_CLASS = 32
    NOSE_INDEX = 0
    LEFT_EYE_INDEX = 1
    RIGHT_EYE_INDEX = 2
    LEFT_EAR_INDEX = 3
    RIGHT_EAR_INDEX = 4
    FACE_KEYPOINTS = (NOSE_INDEX, LEFT_EYE_INDEX, RIGHT_EYE_INDEX, LEFT_EAR_INDEX, RIGHT_EAR_INDEX)

    def __init__(
        self,
        *,
        mode: str,
        detect_model_path: str,
        pose_model_path: str,
        device: str,
        config: Dict,
        imgsz: Optional[int] = None,
        half: bool = True,
        end2end: bool = False,
    ) -> None:
        self.mode = mode
        self.detect_model_path = detect_model_path
        self.pose_model_path = pose_model_path
        self.device = str(device)
        self.config = config
        yolo_cfg = config.get("yolo26", {})
        self.imgsz = int(imgsz or yolo_cfg.get("imgsz", 960 if mode == "movie" else 1280))
        self.pose_imgsz = int(yolo_cfg.get("pose_imgsz", min(self.imgsz, 960)))
        self.quantize = 16 if bool(half and self.device != "cpu") else 32
        self.end2end = bool(end2end)
        self._detect_model = None
        self._pose_model = None
        self._warned_end2end = False

    def _ensure_loaded(self) -> None:
        if self._detect_model is not None:
            return
        from ultralytics import YOLO

        for path in (self.detect_model_path, self.pose_model_path):
            if not Path(path).is_file():
                raise FileNotFoundError(f"YOLO26 model missing: {path}")
        self._detect_model = YOLO(self.detect_model_path)
        self._pose_model = YOLO(self.pose_model_path)
        logger.info(
            "YOLO26 detector loaded detect={} pose={} device={} imgsz={} pose_imgsz={} quantize={} end2end={}",
            self.detect_model_path,
            self.pose_model_path,
            self.device,
            self.imgsz,
            self.pose_imgsz,
            self.quantize,
            self.end2end,
        )

    def _predict(self, model: Any, rgb: np.ndarray, **kwargs: Any) -> Any:
        params = dict(
            source=rgb,
            device=self.device,
            quantize=self.quantize,
            verbose=False,
            save=False,
            stream=False,
        )
        params.update(kwargs)
        while True:
            try:
                return model.predict(**params)[0]
            except TypeError as exc:
                message = str(exc).lower()
                if "end2end" in params and "end2end" in message:
                    params.pop("end2end", None)
                    if not self._warned_end2end:
                        logger.warning(
                            "Ultralytics predict rejected end2end; retrying default head: {}",
                            exc,
                        )
                        self._warned_end2end = True
                    continue
                raise

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if value is None:
            return np.empty((0,), dtype=np.float32)
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @staticmethod
    def _normalized_box(xyxy: Sequence[float], width: int, height: int) -> BBox:
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        return sanitize_box((x1 / width, y1 / height, x2 / width, y2 / height))

    @classmethod
    def _visible_face_points(
        cls,
        keypoints: np.ndarray,
        *,
        min_confidence: float,
    ) -> Dict[int, tuple[float, float]]:
        if keypoints.ndim != 2 or keypoints.shape[0] < 5 or keypoints.shape[1] < 2:
            return {}
        visible: Dict[int, tuple[float, float]] = {}
        for index in cls.FACE_KEYPOINTS:
            confidence = float(keypoints[index, 2]) if keypoints.shape[1] >= 3 else 1.0
            if confidence >= min_confidence:
                visible[index] = (float(keypoints[index, 0]), float(keypoints[index, 1]))
        return visible

    @classmethod
    def head_box_from_keypoints(
        cls,
        keypoints: np.ndarray,
        person_box: BBox,
        *,
        min_confidence: float,
    ) -> Optional[BBox]:
        """Convert visible YOLO26-pose facial keypoints into a head box."""
        visible = cls._visible_face_points(keypoints, min_confidence=min_confidence)
        if len(visible) < 2:
            return None
        xs = [point[0] for point in visible.values()]
        ys = [point[1] for point in visible.values()]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        person_width = person_box[2] - person_box[0]
        person_height = person_box[3] - person_box[1]
        observed_width = max(xs) - min(xs)
        observed_height = max(ys) - min(ys)
        head_width = max(observed_width * 2.2, person_width * 0.20, 0.025)
        head_height = max(observed_height * 3.0, person_height * 0.18, 0.035)
        center_y -= 0.08 * head_height
        return sanitize_box((
            center_x - head_width / 2.0,
            center_y - head_height / 2.0,
            center_x + head_width / 2.0,
            center_y + head_height / 2.0,
        ))

    @classmethod
    def head_evidence_from_keypoints(
        cls,
        keypoints: np.ndarray,
        person_box: BBox,
        *,
        min_confidence: float,
    ) -> Optional[Tuple[str, BBox]]:
        """Return face only for frontal evidence; otherwise return head."""
        visible = cls._visible_face_points(keypoints, min_confidence=min_confidence)
        box = cls.head_box_from_keypoints(
            keypoints, person_box, min_confidence=min_confidence
        )
        if box is None:
            return None
        # A strong frontal face requires both eyes. Nose plus one eye is
        # useful profile/head evidence, but treating it as a primary frontal
        # face lets side views and foreground backs outscore the actual subject.
        frontal = (
            cls.NOSE_INDEX in visible
            and cls.LEFT_EYE_INDEX in visible
            and cls.RIGHT_EYE_INDEX in visible
        )
        return ("face" if frontal else "head", box)

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> List[Detection]:
        del timestamp_ms
        self._ensure_loaded()
        assert self._detect_model is not None
        assert self._pose_model is not None

        height, width = rgb.shape[:2]
        yolo_cfg = self.config.get("yolo26", {})
        person_conf = float(yolo_cfg.get("person_confidence", 0.20))
        ball_conf = float(yolo_cfg.get("ball_confidence", 0.06))
        detect_conf = min(person_conf, ball_conf) if self.mode == "sports" else person_conf
        detect_classes = [self.PERSON_CLASS, self.SPORTS_BALL_CLASS] if self.mode == "sports" else [self.PERSON_CLASS]

        detect_result = self._predict(
            self._detect_model,
            rgb,
            imgsz=self.imgsz,
            conf=detect_conf,
            iou=float(yolo_cfg.get("iou", 0.60)),
            classes=detect_classes,
            max_det=int(yolo_cfg.get("max_det", 100)),
            end2end=bool(yolo_cfg.get("end2end", self.end2end)),
        )

        detections: List[Detection] = []
        boxes_obj = getattr(detect_result, "boxes", None)
        if boxes_obj is not None:
            xyxy = self._to_numpy(getattr(boxes_obj, "xyxy", None))
            classes = self._to_numpy(getattr(boxes_obj, "cls", None)).reshape(-1)
            scores = self._to_numpy(getattr(boxes_obj, "conf", None)).reshape(-1)
            for box, class_id, score in zip(xyxy, classes, scores):
                class_id = int(class_id)
                score = float(score)
                if class_id == self.PERSON_CLASS and score >= person_conf:
                    label = "person"
                elif class_id == self.SPORTS_BALL_CLASS and score >= ball_conf:
                    label = "ball"
                else:
                    continue
                detections.append(
                    Detection(
                        label=label,
                        score=score,
                        box=self._normalized_box(box, width, height),
                        source="yolo26_detect",
                    )
                )

        # Pose-derived heads replace MediaPipe face detection. This runs for both
        # modes so close sports shots can still identify announcers/interviews.
        pose_result = self._predict(
            self._pose_model,
            rgb,
            imgsz=self.pose_imgsz,
            conf=float(yolo_cfg.get("pose_confidence", 0.18)),
            iou=float(yolo_cfg.get("pose_iou", 0.65)),
            classes=[self.PERSON_CLASS],
            max_det=int(yolo_cfg.get("pose_max_det", 40)),
            end2end=bool(yolo_cfg.get("pose_end2end", self.end2end)),
        )
        pose_boxes_obj = getattr(pose_result, "boxes", None)
        pose_keypoints_obj = getattr(pose_result, "keypoints", None)
        if pose_boxes_obj is not None and pose_keypoints_obj is not None:
            pose_xyxy = self._to_numpy(getattr(pose_boxes_obj, "xyxy", None))
            pose_scores = self._to_numpy(getattr(pose_boxes_obj, "conf", None)).reshape(-1)
            keypoint_data = self._to_numpy(getattr(pose_keypoints_obj, "data", None))
            detected_people = [
                det for det in detections if det.label == "person"
            ]
            for index, (box, score) in enumerate(zip(pose_xyxy, pose_scores)):
                person_box = self._normalized_box(box, width, height)
                evidence: Optional[Tuple[str, BBox]] = None
                if index < len(keypoint_data):
                    keypoints = np.asarray(
                        keypoint_data[index], dtype=np.float32
                    ).copy()
                    if keypoints.ndim == 2:
                        # Ultralytics returns pixel coordinates. Normalize before
                        # deriving a normalized head box.
                        keypoints[:, 0] /= max(1, width)
                        keypoints[:, 1] /= max(1, height)
                        evidence = self.head_evidence_from_keypoints(
                            keypoints,
                            person_box,
                            min_confidence=float(
                                yolo_cfg.get("keypoint_confidence", 0.25)
                            ),
                        )

                # Pose-person boxes are only a bounded rescue path. Large or
                # duplicate pose boxes can cover several characters and then win
                # movie scoring purely by area, producing the observed giant,
                # lagging focus boxes.
                duplicate = any(
                    self._box_overlap(det.box, person_box)
                    >= float(yolo_cfg.get("pose_person_merge_iou", 0.25))
                    or self._intersection_over_smaller(det.box, person_box)
                    >= float(
                        yolo_cfg.get(
                            "pose_person_merge_intersection_over_smaller", 0.65
                        )
                    )
                    for det in detected_people
                )
                person_area = self._box_area(person_box)
                person_width = person_box[2] - person_box[0]
                person_height = max(1e-6, person_box[3] - person_box[1])
                require_head_evidence = bool(
                    yolo_cfg.get("pose_person_rescue_require_head_evidence", True)
                )
                standard_size = (
                    person_area
                    <= float(yolo_cfg.get("pose_person_rescue_max_area", 0.32))
                    and person_width
                    <= float(yolo_cfg.get("pose_person_rescue_max_width", 0.62))
                )
                strong_frontal_size = (
                    evidence is not None
                    and evidence[0] == "face"
                    and float(score)
                    >= float(
                        yolo_cfg.get(
                            "pose_person_rescue_large_frontal_confidence", 0.50
                        )
                    )
                    and person_area
                    <= float(
                        yolo_cfg.get("pose_person_rescue_large_frontal_max_area", 0.60)
                    )
                    and person_width
                    <= float(
                        yolo_cfg.get("pose_person_rescue_large_frontal_max_width", 0.84)
                    )
                )
                rescue_allowed = (
                    bool(yolo_cfg.get("pose_person_rescue", True))
                    and not duplicate
                    and (evidence is not None or not require_head_evidence)
                    and float(score)
                    >= float(yolo_cfg.get("pose_person_rescue_confidence", 0.24))
                    and (standard_size or strong_frontal_size)
                    and person_width / person_height
                    <= float(yolo_cfg.get("pose_person_rescue_max_aspect", 1.10))
                )
                if rescue_allowed:
                    rescued = Detection(
                        label="person",
                        score=float(score) * 0.92,
                        box=person_box,
                        source="yolo26_pose",
                    )
                    detections.append(rescued)
                    detected_people.append(rescued)

                if evidence is not None:
                    label, head_box = evidence
                    score_scale = 1.0 if label == "face" else 0.68
                    detections.append(
                        Detection(
                            label=label,
                            score=min(1.0, float(score) * score_scale),
                            box=head_box,
                            source=f"yolo26_pose_{label}",
                        )
                    )
        return detections

    @staticmethod
    def _box_area(box: BBox) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    @classmethod
    def _intersection_over_smaller(cls, a: BBox, b: BBox) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        smaller = min(cls._box_area(a), cls._box_area(b))
        return intersection / smaller if smaller > 0.0 else 0.0

    @staticmethod
    def _box_overlap(a: BBox, b: BBox) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0.0 else 0.0

    def close(self) -> None:
        self._detect_model = None
        self._pose_model = None
