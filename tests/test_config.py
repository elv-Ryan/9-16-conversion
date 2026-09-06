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

    def test_detection_gate_has_both_a_floor_and_a_cap(self):
        # The checkpoint's head is NMS-free and returns its top-k whatever the
        # scores are, so a floor without a cap still admits the whole tail.
        config = config_from_params({})
        self.assertEqual(config.min_detection_confidence, 0.05)
        self.assertEqual(config.max_detections, 20)

    def test_max_detections_is_bounded_by_the_checkpoint_head(self):
        with self.assertRaisesRegex(ValueError, "max_detections"):
            config_from_params({"max_detections": 0})
        with self.assertRaisesRegex(ValueError, "max_detections"):
            config_from_params({"max_detections": 301})
        with self.assertRaisesRegex(ValueError, "max_detections"):
            config_from_params({"max_detections": 20.0})
        self.assertEqual(config_from_params({"max_detections": 300}).max_detections, 300)

    def test_min_shot_seconds_is_bounded_and_below_max(self):
        self.assertEqual(config_from_params({}).min_shot_seconds, 0.25)
        with self.assertRaisesRegex(ValueError, "min_shot_seconds"):
            config_from_params({"min_shot_seconds": -1.0})
        with self.assertRaisesRegex(ValueError, "min_shot_seconds"):
            config_from_params({"min_shot_seconds": 11.0})
        with self.assertRaisesRegex(ValueError, "below max_shot_seconds"):
            config_from_params({"min_shot_seconds": 5.0, "max_shot_seconds": 1.0})

    def test_json_types_are_strict(self):
        with self.assertRaisesRegex(ValueError, "batch_size"):
            config_from_params({"batch_size": "8"})
        with self.assertRaisesRegex(ValueError, "use_fp16"):
            config_from_params({"use_fp16": 1})


if __name__ == "__main__":
    unittest.main()
