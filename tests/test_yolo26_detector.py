import unittest

import numpy as np

from vertical_focus.yolo26_detector import Yolo26FocusDetector


class _FakePredictModel:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [object()]


class _RejectQuantizeModel:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        raise TypeError("unexpected keyword argument 'quantize'")


class _Boxes:
    def __init__(self, xyxy, classes, scores):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.cls = np.asarray(classes, dtype=np.float32)
        self.conf = np.asarray(scores, dtype=np.float32)


class _Keypoints:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)


class _Result:
    def __init__(self, boxes, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


class _StaticResultModel:
    def __init__(self, result):
        self.result = result

    def predict(self, **kwargs):
        return [self.result]


class Yolo26DetectorTests(unittest.TestCase):
    def test_predict_uses_quantize_instead_of_deprecated_half(self):
        detector = Yolo26FocusDetector(
            mode="sports",
            detect_model_path="unused",
            pose_model_path="unused",
            device="0",
            config={},
            half=True,
        )
        model = _FakePredictModel()
        detector._predict(model, np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(16, model.calls[-1]["quantize"])
        self.assertNotIn("half", model.calls[-1])

    def test_predict_never_falls_back_to_deprecated_half(self):
        detector = Yolo26FocusDetector(
            mode="sports",
            detect_model_path="unused",
            pose_model_path="unused",
            device="0",
            config={},
            half=True,
        )
        model = _RejectQuantizeModel()
        with self.assertRaises(TypeError):
            detector._predict(model, np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(1, len(model.calls))
        self.assertNotIn("half", model.calls[0])

    def test_predict_uses_fp32_quantize_on_cpu(self):
        detector = Yolo26FocusDetector(
            mode="movie",
            detect_model_path="unused",
            pose_model_path="unused",
            device="cpu",
            config={},
            half=True,
        )
        model = _FakePredictModel()
        detector._predict(model, np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(32, model.calls[-1]["quantize"])
        self.assertNotIn("half", model.calls[-1])

    @staticmethod
    def person_box():
        return (0.35, 0.18, 0.65, 0.92)

    def test_frontal_keypoints_emit_face(self):
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[0] = (0.50, 0.22, 0.95)  # nose
        keypoints[1] = (0.47, 0.20, 0.90)  # left eye
        keypoints[2] = (0.53, 0.20, 0.90)  # right eye
        evidence = Yolo26FocusDetector.head_evidence_from_keypoints(
            keypoints, self.person_box(), min_confidence=0.25
        )
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual("face", evidence[0])
        self.assertAlmostEqual((evidence[1][0] + evidence[1][2]) / 2.0, 0.50, places=2)

    def test_nose_plus_one_eye_emits_head_not_primary_face(self):
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[0] = (0.50, 0.22, 0.95)
        keypoints[1] = (0.47, 0.20, 0.90)
        evidence = Yolo26FocusDetector.head_evidence_from_keypoints(
            keypoints, self.person_box(), min_confidence=0.25
        )
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual("head", evidence[0])

    def test_ear_only_or_nonfrontal_keypoints_emit_head_not_face(self):
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[3] = (0.44, 0.22, 0.90)
        keypoints[4] = (0.56, 0.22, 0.90)
        evidence = Yolo26FocusDetector.head_evidence_from_keypoints(
            keypoints, self.person_box(), min_confidence=0.25
        )
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual("head", evidence[0])

    def test_single_keypoint_returns_no_head_region(self):
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[0] = (0.5, 0.2, 0.9)
        self.assertIsNone(
            Yolo26FocusDetector.head_evidence_from_keypoints(
                keypoints, self.person_box(), min_confidence=0.25
            )
        )

    def test_head_box_is_normalized(self):
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[0] = (0.50, 0.22, 0.95)
        keypoints[1] = (0.47, 0.20, 0.90)
        box = Yolo26FocusDetector.head_box_from_keypoints(
            keypoints, self.person_box(), min_confidence=0.25
        )
        self.assertIsNotNone(box)
        assert box is not None
        self.assertTrue(0.0 <= box[0] < box[2] <= 1.0)
        self.assertTrue(0.0 <= box[1] < box[3] <= 1.0)

    @staticmethod
    def _frontal_keypoints_pixels(center_x=40.0):
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[0] = (center_x, 20.0, 0.95)
        keypoints[1] = (center_x - 3.0, 18.0, 0.90)
        keypoints[2] = (center_x + 3.0, 18.0, 0.90)
        return keypoints

    def test_duplicate_pose_person_is_not_emitted_twice(self):
        detector = Yolo26FocusDetector(
            mode="movie",
            detect_model_path="unused",
            pose_model_path="unused",
            device="cpu",
            config={"yolo26": {"person_confidence": 0.18}},
        )
        detector._detect_model = _StaticResultModel(
            _Result(_Boxes([[20, 10, 60, 95]], [0], [0.90]))
        )
        detector._pose_model = _StaticResultModel(
            _Result(
                _Boxes([[19, 9, 61, 96]], [0], [0.92]),
                _Keypoints([self._frontal_keypoints_pixels()]),
            )
        )
        rows = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), 0)
        people = [row for row in rows if row.label == "person"]
        faces = [row for row in rows if row.label == "face"]
        self.assertEqual(1, len(people))
        self.assertEqual("yolo26_detect", people[0].source)
        self.assertEqual(1, len(faces))

    def test_pose_person_rescue_requires_head_evidence(self):
        detector = Yolo26FocusDetector(
            mode="movie",
            detect_model_path="unused",
            pose_model_path="unused",
            device="cpu",
            config={"yolo26": {"person_confidence": 0.18}},
        )
        detector._detect_model = _StaticResultModel(
            _Result(_Boxes([], [], []))
        )
        detector._pose_model = _StaticResultModel(
            _Result(
                _Boxes([[25, 10, 60, 90]], [0], [0.90]),
                _Keypoints([np.zeros((17, 3), dtype=np.float32)]),
            )
        )
        rows = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), 0)
        self.assertFalse(any(row.label == "person" for row in rows))

    def test_bounded_pose_person_with_frontal_evidence_can_rescue_miss(self):
        detector = Yolo26FocusDetector(
            mode="movie",
            detect_model_path="unused",
            pose_model_path="unused",
            device="cpu",
            config={"yolo26": {"person_confidence": 0.18}},
        )
        detector._detect_model = _StaticResultModel(
            _Result(_Boxes([], [], []))
        )
        detector._pose_model = _StaticResultModel(
            _Result(
                _Boxes([[25, 10, 60, 90]], [0], [0.90]),
                _Keypoints([self._frontal_keypoints_pixels(center_x=42.0)]),
            )
        )
        rows = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), 0)
        self.assertTrue(any(row.label == "person" for row in rows))
        self.assertTrue(any(row.label == "face" for row in rows))

    def test_oversized_pose_person_rescue_is_rejected(self):
        detector = Yolo26FocusDetector(
            mode="movie",
            detect_model_path="unused",
            pose_model_path="unused",
            device="cpu",
            config={
                "yolo26": {
                    "person_confidence": 0.18,
                    "pose_person_rescue_max_area": 0.42,
                    "pose_person_rescue_max_width": 0.72,
                }
            },
        )
        detector._detect_model = _StaticResultModel(
            _Result(_Boxes([], [], []))
        )
        detector._pose_model = _StaticResultModel(
            _Result(
                _Boxes([[0, 0, 95, 95]], [0], [0.95]),
                _Keypoints([self._frontal_keypoints_pixels(center_x=50.0)]),
            )
        )
        rows = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), 0)
        self.assertFalse(any(row.label == "person" for row in rows))
        self.assertTrue(any(row.label == "face" for row in rows))



if __name__ == "__main__":
    unittest.main()
