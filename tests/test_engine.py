import json
import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from vertical_focus.config import RuntimeParams, load_configuration
from vertical_focus.engine import VerticalFocusEngine
from vertical_focus.shots import TagStoreShots
from vertical_focus.types import Detection


CONFIG = {
    "detection": {"detection_fps": 10.0},
    "tracking": {
        "max_missed_updates": 4,
        "match_center_distance": 0.2,
        "min_iou": 0.01,
        "box_alpha": 0.7,
    },
    "selection": {
        "min_track_hits": 1,
        "min_face_area": 0.001,
        "group_score_ratio": 0.72,
        "group_crop_width_factor": 1.05,
        "min_motion_score": 0.1,
    },
    "temporal": {
        "min_hold_seconds": 0.2,
        "lost_hold_seconds": 0.2,
        "switch_margin": 0.1,
        "deadband": 0.004,
        "response_time_seconds": 0.3,
        "max_speed_normalized_per_second": 0.25,
        "max_acceleration_normalized_per_second2": 0.8,
    },
    "local_shots": {"histogram_threshold": 0.5, "minimum_shot_seconds": 0.2},
}


class FakeDetector:
    def detect(self, rgb, timestamp_ms):
        if timestamp_ms < 1000:
            box = (0.18, 0.15, 0.35, 0.90)
        else:
            box = (0.65, 0.15, 0.82, 0.90)
        return [Detection("person", 0.9, box)]

    def close(self):
        pass


class GradualMoveDetector:
    def detect(self, rgb, timestamp_ms):
        del rgb
        if timestamp_ms < 800:
            center = 0.28
        elif timestamp_ms < 1400:
            center = 0.28 + 0.40 * ((timestamp_ms - 800) / 600.0)
        else:
            center = 0.68
        return [Detection("person", 0.92, (center - 0.09, 0.15, center + 0.09, 0.92))]

    def close(self):
        pass


class FailOnceDetector:
    def __init__(self):
        self.failed = False

    def detect(self, rgb, timestamp_ms):
        del rgb, timestamp_ms
        if not self.failed:
            self.failed = True
            raise RuntimeError("synthetic detector failure")
        return [Detection("person", 0.90, (0.35, 0.15, 0.55, 0.92))]

    def close(self):
        pass


class EngineTests(unittest.TestCase):
    def test_decodes_every_frame_and_emits_shot_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "synthetic.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 180))
            self.assertTrue(writer.isOpened())
            for frame_idx in range(60):
                frame = np.zeros((180, 320, 3), dtype=np.uint8)
                if frame_idx >= 30:
                    frame[:, :] = (0, 0, 180)
                x = 60 if frame_idx < 30 else 220
                cv2.rectangle(frame, (x, 30), (x + 40, 165), (255, 255, 255), -1)
                writer.write(frame)
            writer.release()

            debug_path = str(Path(directory) / "debug.jsonl")
            engine = VerticalFocusEngine(
                mode="movie",
                policy_config=CONFIG,
                object_model="unused",
                face_model="unused",
                delegate="cpu",
                shots=TagStoreShots([]),
                progress_log_interval_seconds=10.0,
                debug_jsonl_path=debug_path,
                detector=FakeDetector(),
            )
            result = engine.process_file(path, 0)
            engine.close()

            self.assertEqual(60, sum(len(shot.decisions) for shot in result.shots))
            self.assertGreaterEqual(len(result.shots), 2)
            self.assertTrue(all(0.0 <= decision.x_center <= 1.0 for shot in result.shots for decision in shot.decisions))
            self.assertGreater(result.duration_ms, 1900)

            debug_rows = [
                json.loads(line)
                for line in Path(debug_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(60, len(debug_rows))
            self.assertTrue(all(row["mode"] == "movie" for row in debug_rows))
            self.assertTrue(all(0.0 < row["crop_width"] <= 1.0 for row in debug_rows))
            self.assertTrue(
                all(
                    {
                        "key",
                        "label",
                        "scene_state",
                        "evidence_kind",
                        "smoothing_regime",
                        "focus_ids",
                        "boxes",
                    }
                    <= set(row["selected"])
                    for row in debug_rows
                )
            )
            self.assertTrue(
                all(
                    all(
                        set(box) == {"id", "label", "score", "box"}
                        and len(box["box"]) == 4
                        and 0.0 <= box["box"][0] < box["box"][2] <= 1.0
                        and 0.0 <= box["box"][1] < box["box"][3] <= 1.0
                        for box in row["selected"]["boxes"]
                    )
                    for row in debug_rows
                )
            )

    def test_failed_input_does_not_leak_state_into_next_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "recover.avi")
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 180)
            )
            self.assertTrue(writer.isOpened())
            for _ in range(18):
                writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
            writer.release()

            engine = VerticalFocusEngine(
                mode="movie",
                policy_config=CONFIG,
                shots=TagStoreShots([]),
                progress_log_interval_seconds=10.0,
                detector=FailOnceDetector(),
            )
            with self.assertRaisesRegex(RuntimeError, "synthetic detector failure"):
                engine.process_file(path, 0)
            result = engine.process_file(path, 0)
            engine.close()

            self.assertEqual(
                18, sum(len(shot.decisions) for shot in result.shots)
            )
            self.assertTrue(result.shots[0].shot_id.startswith("local:1:"))

    def test_movie_settles_to_moved_subject_before_exact_hold(self):
        movie_config, _, _ = load_configuration(RuntimeParams(mode="movie"))
        movie_config["selection"]["min_track_hits"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "continuous.avi")
            debug_path = str(Path(directory) / "continuous-debug.jsonl")
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 180)
            )
            self.assertTrue(writer.isOpened())
            for _ in range(120):
                writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
            writer.release()

            engine = VerticalFocusEngine(
                mode="movie",
                policy_config=movie_config,
                shots=TagStoreShots([]),
                progress_log_interval_seconds=10.0,
                debug_jsonl_path=debug_path,
                detector=GradualMoveDetector(),
            )
            result = engine.process_file(path, 0)
            engine.close()
            self.assertEqual(120, sum(len(shot.decisions) for shot in result.shots))

            rows = [
                json.loads(line)
                for line in Path(debug_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreater(rows[-1]["output_x"], 0.60)
            self.assertEqual("locked", rows[-1]["selected"]["smoothing_regime"])
            final_values = [row["output_x"] for row in rows[-10:]]
            self.assertLess(max(final_values) - min(final_values), 1e-9)


if __name__ == "__main__":
    unittest.main()
