from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .types import Candidate, FrameEvidence, ModelIdentity


EXPECTED_FAMILIES = (
    "active_speaker",
    "gameplay_follow",
    "graphic_text_lock",
    "person_subject",
    "safe_center",
    "split_screen",
    "static_composition",
)


class YoloStudentModel:
    """One custom YOLO stream that jointly emits family and focus bbox."""

    def __init__(
        self,
        *,
        model_path: str,
        manifest_path: str,
        verify_sha256: bool,
        device: str,
        imgsz: int,
        min_confidence: float,
        max_detections: int,
        use_fp16: bool,
        iou: float = 0.7,
        top_k: int = 20,
    ) -> None:
        self.model_path = Path(model_path)
        self.manifest_path = Path(manifest_path)
        self.verify_sha256 = bool(verify_sha256)
        self.device = str(device)
        self.imgsz = int(imgsz)
        self.min_confidence = float(min_confidence)
        self.iou = float(iou)
        self.max_detections = int(max_detections)
        self.top_k = int(top_k)
        self.use_fp16 = bool(use_fp16)
        self._model = None
        self._manifest = self._load_manifest()
        self._sha256 = self._hash_file(self.model_path)
        expected_hash = str(self._manifest.get("sha256", "")).strip().lower()
        if self.verify_sha256:
            if not expected_hash:
                raise ValueError(f"model manifest has no sha256: {self.manifest_path}")
            if expected_hash != self._sha256:
                raise ValueError(
                    f"model sha256 mismatch: expected={expected_hash} actual={self._sha256}"
                )

    @staticmethod
    def _hash_file(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"YOLO Student checkpoint missing: {path}")
        digest = sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"model manifest missing: {self.manifest_path}")
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("model manifest must be a JSON object")
        names = payload.get("class_names")
        if names != list(EXPECTED_FAMILIES):
            raise ValueError(
                "model manifest class_names do not match the locked seven-family contract: "
                f"{names!r}"
            )
        return payload

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            path=str(self.model_path),
            sha256=self._sha256,
            artifact_version=str(self._manifest.get("artifact_version", "unknown")),
            class_names=tuple(self._manifest["class_names"]),
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        model = YOLO(str(self.model_path))
        names_raw = model.names
        names = (
            [str(names_raw[index]) for index in sorted(names_raw)]
            if isinstance(names_raw, Mapping)
            else [str(name) for name in names_raw]
        )
        if names != list(EXPECTED_FAMILIES):
            raise ValueError(
                "checkpoint class names do not match the locked seven-family contract: "
                f"{names!r}"
            )
        self._model = model

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

    def _predict(self, frames: Sequence[np.ndarray]) -> Sequence[Any]:
        self._ensure_loaded()
        assert self._model is not None
        common = dict(
            source=list(frames),
            imgsz=self.imgsz,
            conf=self.min_confidence,
            iou=self.iou,
            # The end-to-end head hands back its top-k regardless of quality,
            # and its own default k is 300. Without both of these the whole
            # tail arrives and every downstream consumer has to carry it.
            max_det=self.max_detections,
            device=self.device,
            verbose=False,
            save=False,
            stream=False,
        )
        # Ultralytics renamed this control from ``half`` to ``quantize`` and
        # now warns on every call that ``half`` is going away. On the pinned
        # build the two are bit-identical over real frames, so prefer the name
        # that has a future and keep ``half`` only for older runtimes.
        #
        # The order matters more than it looks: Ultralytics rejects an unknown
        # argument with SyntaxError from its own argument checker, not
        # TypeError, so the removal this deprecation promises would not have
        # been caught by the previous handler at all -- it would have taken the
        # daemon down on the first frame after an upgrade.
        try:
            return self._model.predict(
                quantize=16 if self.use_fp16 and self.device != "cpu" else 32,
                **common,
            )
        except (TypeError, ValueError, SyntaxError) as error:
            if "quantize" not in str(error).lower():
                raise
            return self._model.predict(half=self.use_fp16, **common)

    def infer_batch(
        self,
        *,
        frame_indices: Sequence[int],
        frames: Sequence[np.ndarray],
        source_fps: float,
    ) -> List[FrameEvidence]:
        if len(frame_indices) != len(frames):
            raise ValueError("frame_indices and frames must have equal length")
        results = self._predict(frames)
        evidence: List[FrameEvidence] = []
        for frame_index, result in zip(frame_indices, results):
            candidates: List[Candidate] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes):
                xyxyn = self._to_numpy(getattr(boxes, "xyxyn", None))
                class_ids = self._to_numpy(getattr(boxes, "cls", None)).reshape(-1)
                confidences = self._to_numpy(getattr(boxes, "conf", None)).reshape(-1)
                for box, class_id, confidence in zip(xyxyn, class_ids, confidences):
                    class_index = int(class_id)
                    if class_index < 0 or class_index >= len(EXPECTED_FAMILIES):
                        continue
                    x1, y1, x2, y2 = [float(value) for value in box]
                    normalized = (
                        min(1.0, max(0.0, x1)),
                        min(1.0, max(0.0, y1)),
                        min(1.0, max(0.0, x2)),
                        min(1.0, max(0.0, y2)),
                    )
                    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
                        continue
                    candidates.append(
                        Candidate(
                            family=EXPECTED_FAMILIES[class_index],
                            confidence=float(confidence),
                            bbox=normalized,
                        )
                    )
            candidates.sort(key=lambda item: item.confidence, reverse=True)
            candidates = candidates[: self.top_k]
            evidence.append(
                FrameEvidence(
                    frame_index=int(frame_index),
                    timestamp_ms=int(round(1000.0 * frame_index / source_fps)),
                    candidates=tuple(candidates),
                )
            )
        return evidence
