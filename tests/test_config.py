import unittest

from nba_yolo_shot_tagger.config import RuntimeConfig, config_from_params


class RuntimeConfigTests(unittest.TestCase):
    def test_defaults_are_shot_first(self):
        config = config_from_params({"continue_on_error": True})
        self.assertEqual(config.input_mode, "shot_file")
        self.assertEqual(config.output_track, "vertical_video")
        self.assertEqual(config.imgsz, 1280)
        self.assertEqual(config.inference_fps, 10.0)

    def test_unknown_param_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            config_from_params({"not_a_real_param": 1})

    def test_manifest_requires_path(self):
        with self.assertRaisesRegex(ValueError, "shot_manifest_path"):
            config_from_params({"input_mode": "shot_manifest"})

    def test_json_types_are_strict(self):
        with self.assertRaisesRegex(ValueError, "batch_size"):
            config_from_params({"batch_size": "8"})
        with self.assertRaisesRegex(ValueError, "use_fp16"):
            config_from_params({"use_fp16": 1})


if __name__ == "__main__":
    unittest.main()
