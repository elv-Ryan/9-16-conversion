from pathlib import Path
import re
import unittest

from vertical_focus.config import runtime_params_from_dict


class ConfigTests(unittest.TestCase):
    def test_default_backend_is_yolo26(self):
        runtime = runtime_params_from_dict({})
        self.assertEqual("yolo26", runtime.detector_backend)
        self.assertTrue(runtime.yolo_detect_model.endswith("yolo26s.pt"))
        self.assertTrue(runtime.yolo_pose_model.endswith("yolo26n-pose.pt"))

    def test_non_yolo_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly 'yolo26'"):
            runtime_params_from_dict({"detector_backend": "mediapipe"})

    def test_legacy_model_params_are_accepted_but_do_not_change_backend(self):
        runtime = runtime_params_from_dict(
            {"object_model": "legacy.tflite", "face_model": "legacy-face.tflite"}
        )
        self.assertEqual("yolo26", runtime.detector_backend)
        self.assertEqual("legacy.tflite", runtime.object_model)

    def test_behavior_policy_cannot_key_on_evaluation_clip_identity(self):
        root = Path(__file__).resolve().parents[1]
        policy_text = (root / "src" / "vertical_focus" / "policy.py").read_text(
            encoding="utf-8"
        ).lower()
        config_text = (root / "configs" / "policies.yml").read_text(
            encoding="utf-8"
        ).lower()
        behavior_text = policy_text + "\n" + config_text
        self.assertNotIn("source_media", policy_text)
        self.assertNotIn("/home/", behavior_text)
        self.assertIsNone(re.search(r"\b(?:sports|movie)_\d+\b", behavior_text))

    def test_runtime_source_has_no_mediapipe_import_or_dependency(self):
        root = Path(__file__).resolve().parents[1]
        runtime_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((root / "src" / "vertical_focus").glob("*.py"))
        ).lower()
        requirements = (root / "requirements.txt").read_text(encoding="utf-8").lower()
        setup_text = (root / "setup.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("import mediapipe", runtime_text)
        self.assertNotIn("from mediapipe", runtime_text)
        self.assertNotIn("mediapipe>=", requirements)
        self.assertNotIn('"mediapipe', setup_text)


if __name__ == "__main__":
    unittest.main()
