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

    def test_family_determination_default_depends_on_input_mode(self):
        shot_file = config_from_params({"input_mode": "shot_file"})
        segment_file = config_from_params({"input_mode": "segment_file"})
        # shot_file inputs are whole shots, so the whole file feeds the vote.
        self.assertEqual(shot_file.family_determination_max_seconds, 999999.0)
        self.assertEqual(segment_file.family_determination_max_seconds, 3.0)

    def test_explicit_family_determination_wins_over_the_mode_default(self):
        config = config_from_params(
            {"input_mode": "segment_file", "family_determination_max_seconds": 5.0}
        )
        self.assertEqual(config.family_determination_max_seconds, 5.0)

    def test_trajectory_commit_lag_must_clear_the_detector_lookahead(self):
        # Committing inside shot detection's lookahead could be invalidated by
        # a cut reported late.
        with self.assertRaisesRegex(ValueError, "trajectory_commit_lag_frames"):
            config_from_params({"trajectory_commit_lag_frames": 10})
        config = config_from_params({"trajectory_commit_lag_frames": 240})
        self.assertEqual(config.trajectory_commit_lag_frames, 240)

    def test_shot_manifest_mode_is_gone(self):
        with self.assertRaisesRegex(ValueError, "input_mode"):
            config_from_params({"input_mode": "shot_manifest"})
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            config_from_params({"shot_manifest_path": "/tmp/shots.json"})

    def test_segment_file_input_mode_is_accepted(self):
        config = config_from_params({"input_mode": "segment_file"})
        self.assertEqual(config.input_mode, "segment_file")

    def test_unknown_input_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "input_mode"):
            config_from_params({"input_mode": "not_a_real_mode"})

    def test_json_types_are_strict(self):
        with self.assertRaisesRegex(ValueError, "batch_size"):
            config_from_params({"batch_size": "8"})
        with self.assertRaisesRegex(ValueError, "use_fp16"):
            config_from_params({"use_fp16": 1})


if __name__ == "__main__":
    unittest.main()
