import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

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


class EngineTests(unittest.TestCase):
    def test_decodes_every_frame_and_emits_single_shot(self):
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

            engine = VerticalFocusEngine(
                mode="movie",
                policy_config=CONFIG,
                object_model="unused",
                delegate="cpu",
                shots=TagStoreShots([]),
                progress_log_interval_seconds=10.0,
                detector=FakeDetector(),
            )
            result = engine.process_file(path, 0)
            engine.close()

            self.assertEqual(60, sum(len(shot.decisions) for shot in result.shots))
            # Regardless of scene cuts within the file, the vertical_video track
            # is a single uncut trajectory: exactly one shot per file, spanning
            # the whole media from start_ms 0 to the full duration.
            self.assertEqual(1, len(result.shots))
            shot = result.shots[0]
            self.assertEqual(0, shot.start_ms)
            self.assertEqual(0, shot.start_frame_idx)
            self.assertEqual(result.duration_ms, shot.end_ms)
            self.assertEqual(60, len(shot.decisions))
            self.assertTrue(all(0.0 <= decision.x_center <= 1.0 for shot in result.shots for decision in shot.decisions))
            self.assertGreater(result.duration_ms, 1900)


if __name__ == "__main__":
    unittest.main()
